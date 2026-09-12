from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/179/secom"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data/raw"
EXPECTED = ("secom.data", "secom_labels.data", "secom.names")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "secom.zip"
    request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0 industry-4.0-lab/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink(missing_ok=True)
    for name in EXPECTED:
        matches = list(RAW_DIR.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Downloaded UCI archive does not contain {name}.")
        source, target = matches[0], RAW_DIR / name
        if source != target:
            shutil.move(str(source), target)
    manifest = {
        "source_page": SOURCE_PAGE, "archive_url": ARCHIVE_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "license": "CC BY 4.0",
        "files": {name: {"bytes": (RAW_DIR / name).stat().st_size, "sha256": sha256(RAW_DIR / name)} for name in EXPECTED},
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
