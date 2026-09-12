from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/stephanmatzka/condition-monitoring-dataset-ai4i-2021"
SOURCE_PAGE = "https://www.kaggle.com/datasets/stephanmatzka/condition-monitoring-dataset-ai4i-2021"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.zip"
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=90) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(RAW)

    files = {}
    for path in sorted(p for p in RAW.rglob("*") if p.is_file() and p.name != "manifest.json"):
        rel = path.relative_to(RAW).as_posix()
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "source_page": SOURCE_PAGE,
        "download_endpoint": URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC-SA 4.0",
        "files": files,
        "redistribution_note": "Raw source files are downloaded at use time and are not redistributed from this repository.",
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(files), "manifest": str(RAW / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
