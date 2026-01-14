import os
import re
import json
import wave
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

RE_DUR = re.compile(r"__dur_(\d+(?:\.\d+)?)s", re.IGNORECASE)

def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)

def parse_duration_from_name(name: str) -> Optional[float]:
    m = RE_DUR.search(name)
    return float(m.group(1)) if m else None

def scan_recordings(root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not root.exists():
        return items

    for wav_path in root.rglob("*.wav"):
        # Skip partial/in-progress if you ever write those; your V0 uses dur_ in finished names
        if "__dur_" not in wav_path.name:
            continue

        json_path = wav_path.with_suffix(".json")
        dur = parse_duration_from_name(wav_path.name)
        if dur is None:
            try:
                dur = wav_duration_seconds(wav_path)
            except Exception:
                dur = None

        stat = wav_path.stat()
        items.append({
            "rel": str(wav_path.relative_to(root)).replace("\\", "/"),
            "name": wav_path.name,
            "date_dir": wav_path.parent.name,  # typically YYYY-MM-DD
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "duration": dur,
            "has_json": json_path.exists(),
            "json_rel": str(json_path.relative_to(root)).replace("\\", "/") if json_path.exists() else None,
        })

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items

def human_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

ROOT = Path(os.environ.get("FR_RECORDINGS_DIR", "recordings")).resolve()

app = FastAPI(title="Flight Recorder Web UI")
app.mount("/files", StaticFiles(directory=str(ROOT)), name="files")

@app.get("/", response_class=HTMLResponse)
def index():
    clips = scan_recordings(ROOT)

    # Minimal, dependency-free HTML
    rows = []
    for c in clips[:500]:  # cap to keep it snappy
        dur = f"{c['duration']:.1f}s" if isinstance(c["duration"], (int, float)) else "?"
        js = f'<a href="/meta/{c["json_rel"]}">meta</a>' if c["has_json"] else ""
        rows.append(f"""
        <div style="padding:10px; border-bottom:1px solid #ddd;">
          <div><b>{c["rel"]}</b> — {dur} — {human_size(c["size"])} {js}</div>
          <audio controls preload="none" style="width: 100%;">
            <source src="/files/{c["rel"]}" type="audio/wav">
          </audio>
        </div>
        """)

    body = "\n".join(rows) if rows else "<p>No clips found yet.</p>"

    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Flight Recorder</title>
      </head>
      <body style="font-family: sans-serif; max-width: 900px; margin: 20px auto;">
        <h2>Flight Recorder</h2>
        <p>Root: <code>{ROOT}</code></p>
        {body}
      </body>
    </html>
    """

@app.get("/meta/{rel_path:path}", response_class=HTMLResponse)
def meta(rel_path: str):
    p = (ROOT / rel_path).resolve()
    if not str(p).startswith(str(ROOT)) or not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if p.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Not a json file")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        pretty = json.dumps(data, indent=2)
    except Exception:
        pretty = p.read_text(encoding="utf-8", errors="replace")

    return f"""
    <html>
      <head><meta name="viewport" content="width=device-width, initial-scale=1" /></head>
      <body style="font-family: monospace; max-width: 900px; margin: 20px auto;">
        <p><a href="/">← back</a></p>
        <pre>{pretty}</pre>
      </body>
    </html>
    """

