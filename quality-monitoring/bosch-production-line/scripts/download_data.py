from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "bosch-production-line-performance"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        raise SystemExit("Kaggle CLI is not installed. Install project requirements first.")

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "This competition requires your own Kaggle account and acceptance of the applicable competition rules.\n"
        "The script does not bypass those requirements."
    )
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(args.output),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            "Kaggle download failed. Confirm that your Kaggle credentials are configured and that you have accepted "
            "the Bosch competition rules in your own account.",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode) from exc

    zip_candidates = sorted(args.output.glob(f"{COMPETITION}*.zip"))
    if not zip_candidates:
        zip_candidates = sorted(args.output.glob("*.zip"))
    if not zip_candidates:
        raise FileNotFoundError("Kaggle CLI completed but no competition ZIP was found")

    archive = zip_candidates[-1]
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(args.output)
    if not args.keep_zip:
        archive.unlink()
    print(f"Extracted Bosch competition files to {args.output}")


if __name__ == "__main__":
    main()
