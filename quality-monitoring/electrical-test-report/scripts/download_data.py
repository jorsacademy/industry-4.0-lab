from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/joshipranjal5/electrical-test-report"
SOURCE_PAGE = "https://www.kaggle.com/datasets/joshipranjal5/electrical-test-report"
RAW_DIR = Path("data/raw")
EXPECTED_NAME = "Electrical_Test_Report.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "electrical-test-report.zip"
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink(missing_ok=True)

    matches = list(RAW_DIR.rglob(EXPECTED_NAME))
    if not matches:
        csvs = list(RAW_DIR.rglob("*.csv"))
        if len(csvs) == 1:
            matches = csvs
    if not matches:
        raise FileNotFoundError(f"Dataset downloaded, but {EXPECTED_NAME} was not found in {RAW_DIR}.")

    source = matches[0]
    target = RAW_DIR / EXPECTED_NAME
    if source != target:
        shutil.move(str(source), target)

    manifest = {
        "source_page": SOURCE_PAGE,
        "download_endpoint": DATASET_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {target.name: {"bytes": target.stat().st_size, "sha256": sha256(target)}},
        "redistribution_note": "Raw data is downloaded from Kaggle at use time because the dataset license was not independently verified when this project was assembled.",
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
