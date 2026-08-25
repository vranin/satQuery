"""Stream selected BigEarthNet v2 Sentinel-2 patches from Zenodo.

This does not save the full archive. Because the source is one compressed
stream, bytes before the requested patches still need to be read.

Usage:
    python scripts/download_bigearthnet_subset.py PATCH_ID [PATCH_ID ...]
"""
from __future__ import annotations

import argparse
import logging
import os
import tarfile
from pathlib import Path, PurePosixPath

import requests
import zstandard

S2_ARCHIVE_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "bigearthnet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _safe_destination(output_dir: Path, member_name: str) -> Path | None:
    """Return a safe local path, rejecting archive path traversal."""
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    destination = output_dir.joinpath(*relative.parts)
    try:
        destination.relative_to(output_dir)
    except ValueError:
        return None
    return destination


def download_subset(sample_ids: set[str], output_dir: Path, archive_url: str) -> set[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    log.info("Streaming official BigEarthNet-S2 archive")
    log.info("Target patches: %s", ", ".join(sorted(sample_ids)))

    with requests.get(archive_url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(response.raw) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    matched_id = next(
                        (sample_id for sample_id in sample_ids if sample_id in member.name),
                        None,
                    )
                    if matched_id is None:
                        if found == sample_ids:
                            log.info("Finished requested patch directory; stopping stream")
                            break
                        continue
                    archive_path = _safe_destination(output_dir, member.name)
                    if archive_path is None:
                        raise RuntimeError(f"Unsafe archive member: {member.name}")
                    destination = output_dir / matched_id / archive_path.name
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(f"Could not read archive member: {member.name}")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("wb") as target:
                            while chunk := source.read(1024 * 1024):
                                target.write(chunk)
                        found.add(matched_id)
                        log.info("Extracted %s", destination)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_ids", nargs="+", help="BigEarthNet S2 patch IDs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive-url", default=S2_ARCHIVE_URL)
    args = parser.parse_args()

    found = download_subset(set(args.sample_ids), args.output, args.archive_url)
    missing = set(args.sample_ids) - found
    if missing:
        raise SystemExit(f"Patches not found in archive: {', '.join(sorted(missing))}")
    log.info("Saved selected patches under %s", args.output)


if __name__ == "__main__":
    main()