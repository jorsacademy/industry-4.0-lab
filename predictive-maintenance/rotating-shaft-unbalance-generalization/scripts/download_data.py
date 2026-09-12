from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import write_manifest


def download_with_resume(url: str, destination: Path, retries: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size > 1_000_000_000:
        print(f"Archive already present: {destination} ({destination.stat().st_size} bytes)")
        return

    for attempt in range(1, retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 180), headers=headers) as response:
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                if existing and not append:
                    existing = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    downloaded = existing
                    last_report = time.time()
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_report >= 15:
                            print(f"downloaded {downloaded / (1024**3):.2f} GiB", flush=True)
                            last_report = time.time()
            partial.replace(destination)
            return
        except Exception as exc:
            print(f"Download attempt {attempt}/{retries} failed: {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(min(30, attempt * 5))


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    source = cfg["source"]
    destination = ROOT / "data" / "raw" / source["archive_name"]
    download_with_resume(source["archive_url"], destination)
    manifest = write_manifest(
        ROOT / "data" / "raw" / "manifest.json",
        source["archive_url"],
        destination,
    )
    manifest.update({"source_page": source["page"], "doi": source["doi"], "license": source["license"]})
    (ROOT / "data" / "raw" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
