from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HANDLE = "supergus/multistage-continuousflow-manufacturing-process"
FILES = ["continuous_factory_process.csv", "notes_on_dataset.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": "https://www.kaggle.com/datasets/supergus/multistage-continuousflow-manufacturing-process",
        "handle": HANDLE,
        "data_file_label": "Data files © Original Authors",
        "files": {},
    }
    for filename in FILES:
        resolved = Path(kagglehub.dataset_download(HANDLE, path=filename))
        source = resolved if resolved.is_file() else resolved / filename
        if not source.exists():
            matches = list(resolved.rglob(filename)) if resolved.exists() else []
            if not matches:
                raise FileNotFoundError(f"KaggleHub did not return {filename}: {resolved}")
            source = matches[0]
        target = RAW / filename
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        manifest["files"][filename] = {
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
