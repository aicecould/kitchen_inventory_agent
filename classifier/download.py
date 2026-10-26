"""Download MASSIVE v1.1 and extract its Chinese partition."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parent
SOURCE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.1.tar.gz"
)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    archive = ROOT.parent / ".run" / "amazon-massive-dataset-1.1.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        temporary = archive.with_suffix(".download")
        print("Downloading MASSIVE v1.1", flush=True)
        urlretrieve(SOURCE_URL, temporary)
        temporary.replace(archive)
    destination = ROOT / "dataset" / "massive"
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member, name in (
            ("1.1/data/zh-CN.jsonl", "zh-CN.jsonl"),
            ("1.1/LICENSE", "LICENSE"),
        ):
            with source.extractfile(member) as handle:
                (destination / name).write_bytes(handle.read())
    print(f"Chinese data ready: {destination / 'zh-CN.jsonl'}")


if __name__ == "__main__":
    main()
