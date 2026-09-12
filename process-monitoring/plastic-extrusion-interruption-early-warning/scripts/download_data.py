from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import kagglehub

HANDLE = "podsyp/find-a-defect-in-the-production-extrusion-line"
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
    source = Path(kagglehub.dataset_download(HANDLE))
    files: dict[str, dict[str, object]] = {}
    for src in sorted(source.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(source)
        dst = RAW / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files[str(rel)] = {"bytes": dst.stat().st_size, "sha256": sha256(dst)}
    manifest = {
        "source": f"https://www.kaggle.com/datasets/{HANDLE}",
        "handle": HANDLE,
        "license": "CC BY-NC-ND 4.0",
        "files": files,
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
