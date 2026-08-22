#!/usr/bin/env python3
"""Download the public MOVi-A 128×128 validation TFRecord shards directly from GCS."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


BUCKET = "kubric-public"
PREFIX = "tfds/movi_a/128x128/1.0.0/"


def list_validation_shards() -> list[dict]:
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
        f"prefix={urllib.parse.quote(PREFIX, safe='')}&maxResults=1000"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        items = json.load(response)["items"]
    return [item for item in items if "/movi_a-validation.tfrecord-" in item["name"]]


def download(item: dict, destination: Path) -> str:
    target = destination / Path(item["name"]).name
    expected_size = int(item["size"])
    if target.exists() and target.stat().st_size == expected_size:
        return f"kept {target.name} ({expected_size} bytes)"
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(f"https://storage.googleapis.com/{BUCKET}/{item['name']}")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download for {target.name}: got {temporary.stat().st_size}, expected {expected_size}")
    os.replace(temporary, target)
    return f"downloaded {target.name} ({expected_size} bytes)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    shards = list_validation_shards()
    if len(shards) != 16:
        raise RuntimeError(f"Expected 16 validation shards; found {len(shards)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, shard, destination): shard for shard in shards}
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)
    (destination / "download_manifest.json").write_text(
        json.dumps({"bucket": BUCKET, "prefix": PREFIX, "shards": shards}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
