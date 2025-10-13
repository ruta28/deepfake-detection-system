import subprocess, shutil, hashlib, json
from pathlib import Path
from datetime import datetime, timezone

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def ffprobe_metadata(path: str):
    try:
        out = subprocess.check_output(
            ['ffprobe','-v','error','-show_format','-show_streams', path],
            stderr=subprocess.STDOUT, text=True, timeout=10
        )
        return {'ffprobe_raw': out}
    except Exception as e:
        return {'ffprobe_raw': None, 'note': f'ffprobe not available or failed: {e}'}

def iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def write_json(path: str, data: dict):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding='utf-8')
