from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/adorigueto/cnc-turning-roughness-forces-and-tool-wear"
SOURCE_PAGE = "https://www.kaggle.com/datasets/adorigueto/cnc-turning-roughness-forces-and-tool-wear"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EXPECTED = ["Exp1.csv", "Exp2.csv", "Prep.csv"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "cnc-turning.zip"
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink(missing_ok=True)

    for expected in EXPECTED:
        target = RAW_DIR / expected
        if target.exists():
            continue
        matches = [p for p in RAW_DIR.rglob("*.csv") if p.name.lower() == expected.lower()]
        if not matches:
            raise FileNotFoundError(f"Downloaded dataset does not contain {expected}")
        shutil.move(str(matches[0]), target)

    manifest = {
        "source_page": SOURCE_PAGE,
        "download_endpoint": DATASET_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC-SA 4.0",
        "files": {
            name: {"bytes": (RAW_DIR / name).stat().st_size, "sha256": sha256(RAW_DIR / name)}
            for name in EXPECTED
        },
        "redistribution_note": "Source files are downloaded at use time; this repository does not redistribute the raw dataset.",
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
