import os
import json
import time
import math
import queue
import signal
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Optional, Deque, Tuple
from collections import deque

import numpy as np
import sounddevice as sd
import wave


# ----------------------------
# Config
# ----------------------------

@dataclass
class VoxConfig:
    device: Optional[int] = None          # None = default input device
    samplerate: int = 48000
    channels: int = 2
    block_ms: int = 20                    # analysis/IO block size
    open_threshold_db: float = -35.0      # start recording if louder than this
    close_threshold_db: float = -42.0     # stop after hangover if quieter than this
    hangover_ms: int = 800                # how long it must stay quiet before closing
    pre_roll_ms: int = 250                # include this much audio before trigger
    min_clip_ms: int = 300                # discard tiny accidental clips
    out_dir: str = "recordings"


def rms_dbfs(block: np.ndarray) -> float:
    """
    block: shape (frames, channels), float32 in [-1, 1]
    returns RMS level in dBFS (0 dBFS = full-scale sine-ish)
    """
    # Avoid NaNs for silence
    rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)) + 1e-12)
    return 20.0 * math.log10(rms)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_paths(base_dir: str) -> Tuple[str, str]:
    now = dt.datetime.now()
    day_dir = os.path.join(base_dir, now.strftime("%Y-%m-%d"))
    ensure_dir(day_dir)
    stem = now.strftime("%H%M%S")
    wav_path = os.path.join(day_dir, f"{stem}.wav")
    json_path = os.path.join(day_dir, f"{stem}.json")
    return wav_path, json_path


# ----------------------------
# Recorder
# ----------------------------

class WavWriter:
    def __init__(self, path: str, samplerate: int, channels: int):
        self.path = path
        self.samplerate = samplerate
        self.channels = channels
        self.wav = wave.open(path, "wb")
        self.wav.setnchannels(channels)
        self.wav.setsampwidth(2)          # int16
        self.wav.setframerate(samplerate)
        self.frames_written = 0

    def write_float_block(self, block: np.ndarray) -> None:
        # Convert float32 [-1,1] to int16
        clipped = np.clip(block, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        self.wav.writeframes(pcm16.tobytes())
        self.frames_written += len(pcm16)

    def close(self) -> None:
        self.wav.close()


def main():
    cfg = VoxConfig()

    # Optional: allow quick edit via env vars without touching code
    # (handy when iterating)
    cfg.device = int(os.getenv("FR_DEVICE")) if os.getenv("FR_DEVICE") else cfg.device

    blocksize = int(cfg.samplerate * (cfg.block_ms / 1000.0))
    hangover_blocks = max(1, int(cfg.hangover_ms / cfg.block_ms))
    preroll_blocks = max(0, int(cfg.pre_roll_ms / cfg.block_ms))
    min_clip_blocks = max(1, int(cfg.min_clip_ms / cfg.block_ms))

    print("Flight Recorder V0 (VOX)")
    print(f" device={cfg.device} sr={cfg.samplerate} ch={cfg.channels} block={cfg.block_ms}ms")
    print(f" open={cfg.open_threshold_db:.1f} dBFS close={cfg.close_threshold_db:.1f} dBFS "
          f"hangover={cfg.hangover_ms}ms preroll={cfg.pre_roll_ms}ms")

    q: "queue.Queue[np.ndarray]" = queue.Queue()
    stopping = False

    def on_sigint(signum, frame):
        nonlocal stopping
        stopping = True
        print("\nStopping...")

    signal.signal(signal.SIGINT, on_sigint)

    def callback(indata, frames, time_info, status):
        if status:
            # underrun/overrun warnings show up here
            print(f"[audio] {status}", flush=True)
        # Copy because sounddevice reuses buffers
        q.put(indata.copy())

    prebuf: Deque[np.ndarray] = deque(maxlen=preroll_blocks if preroll_blocks > 0 else 1)

    recording = False
    quiet_count = 0
    clip_blocks = 0
    writer: Optional[WavWriter] = None
    clip_start_wall: Optional[float] = None
    clip_wav_path: Optional[str] = None
    clip_json_path: Optional[str] = None

    with sd.InputStream(
        device=cfg.device,
        samplerate=cfg.samplerate,
        channels=cfg.channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        last_print = 0.0
        while not stopping:
            try:
                block = q.get(timeout=0.2)
            except queue.Empty:
                continue

            level = rms_dbfs(block)
            now = time.time()

            # lightweight meter print (2x/sec)
            if now - last_print > 0.5:
                state = "REC" if recording else "IDLE"
                print(f"\r[{state}] {level:6.1f} dBFS", end="", flush=True)
                last_print = now

            # Keep pre-roll buffer always
            if preroll_blocks > 0:
                prebuf.append(block)

            if not recording:
                if level >= cfg.open_threshold_db:
                    # Start new clip
                    clip_wav_path, clip_json_path = make_paths(cfg.out_dir)
                    writer = WavWriter(clip_wav_path, cfg.samplerate, cfg.channels)
                    clip_start_wall = time.time()
                    recording = True
                    quiet_count = 0
                    clip_blocks = 0

                    # dump preroll first
                    if preroll_blocks > 0:
                        for b in prebuf:
                            writer.write_float_block(b)
                            clip_blocks += 1

                    print(f"\n--> VOX OPEN @ {level:.1f} dBFS  -> {clip_wav_path}")
                continue

            # recording
            assert writer is not None
            writer.write_float_block(block)
            clip_blocks += 1

            if level <= cfg.close_threshold_db:
                quiet_count += 1
            else:
                quiet_count = 0

            if quiet_count >= hangover_blocks:
                # close
                recording = False
                writer.close()
                clip_end_wall = time.time()
                duration_s = writer.frames_written / cfg.samplerate

                # discard too-short clips
                if clip_blocks < min_clip_blocks:
                    try:
                        os.remove(clip_wav_path)
                    except OSError:
                        pass
                    print(f"\n<-- VOX CLOSE (discarded short clip: {duration_s:.2f}s)")
                else:
                    # rename to include duration
                    dur_tag = f"dur_{duration_s:.1f}s"
                    base, ext = os.path.splitext(clip_wav_path)
                    new_wav = f"{base}__{dur_tag}{ext}"
                    new_json = f"{base}__{dur_tag}.json"
                    try:
                        os.rename(clip_wav_path, new_wav)
                        clip_wav_path = new_wav
                    except OSError:
                        # if rename fails, keep original filename
                        new_json = clip_json_path

                    meta = {
                        "created_local": dt.datetime.now().isoformat(timespec="seconds"),
                        "wav_path": clip_wav_path,
                        "duration_s": round(duration_s, 3),
                        "samplerate": cfg.samplerate,
                        "channels": cfg.channels,
                        "device": cfg.device,
                        "thresholds_dbfs": {
                            "open": cfg.open_threshold_db,
                            "close": cfg.close_threshold_db,
                        },
                        "timing": {
                            "hangover_ms": cfg.hangover_ms,
                            "pre_roll_ms": cfg.pre_roll_ms,
                            "block_ms": cfg.block_ms,
                        },
                        "wallclock": {
                            "start_unix": clip_start_wall,
                            "end_unix": clip_end_wall,
                        },
                        "notes": "V0 VOX clip",
                    }
                    ensure_dir(os.path.dirname(new_json))
                    with open(new_json, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                    print(f"\n<-- VOX CLOSE  ({duration_s:.1f}s)  saved: {clip_wav_path}")

                writer = None
                clip_start_wall = None
                clip_wav_path = None
                clip_json_path = None
                quiet_count = 0
                clip_blocks = 0

    print("\nExited cleanly.")


if __name__ == "__main__":
    main()
