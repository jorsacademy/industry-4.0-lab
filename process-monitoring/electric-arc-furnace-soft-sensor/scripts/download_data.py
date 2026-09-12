from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET_PAGE = "https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-arc-furnace"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/yuriykatser/industrial-data-from-the-arc-furnace"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "source.zip"
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)

    extracted: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".csv"):
                continue
            name = Path(member.filename).name
            destination = RAW / name
            with zf.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted[name] = {"bytes": destination.stat().st_size, "sha256": sha256(destination)}

    archive.unlink(missing_ok=True)
    manifest = {
        "source_page": DATASET_PAGE,
        "download_endpoint": DOWNLOAD_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "MIT",
        "files": extracted,
        "redistribution_note": "Raw source files are downloaded at use time and are not committed by this project.",
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(extracted), "manifest": str(RAW / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
