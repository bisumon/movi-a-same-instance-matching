#!/usr/bin/env python3
"""Extract the small selection inventory from already-downloaded MOVi-A TFRecords.

Install the lightweight reader first: `python -m pip install tfrecord`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as exc:
        raise SystemExit("Install the `tfrecord` package to use this extractor.") from exc

    paths = sorted(Path(args.tfrecord_dir).glob("movi_a-validation.tfrecord-*-of-00016"))
    if len(paths) != 16:
        raise SystemExit(f"Expected 16 complete validation shards; found {len(paths)}")
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    source_index = 0
    with output.open("w", encoding="utf-8") as handle:
        for path in paths:
            for example in tfrecord_loader(str(path), None):
                video_id = example["metadata/video_name"]
                if isinstance(video_id, bytes):
                    video_id = video_id.decode("utf-8")
                if str(video_id) in seen:
                    raise RuntimeError(f"Duplicate video ID {video_id}")
                seen.add(str(video_id))
                shape = example["instances/shape_label"].tolist()
                color = example["instances/color_label"].tolist()
                material = example["instances/material_label"].tolist()
                visibility_flat = example["instances/visibility"].tolist()
                instance_count = len(shape)
                if len(visibility_flat) != instance_count * 24:
                    raise RuntimeError(f"Unexpected visibility shape for {video_id}")
                visibility = [visibility_flat[index * 24 : (index + 1) * 24] for index in range(instance_count)]
                record = {
                    "video_id": str(video_id),
                    "source_index": source_index,
                    "visibility": visibility,
                    "shape_label": shape,
                    "color_label": color,
                    "material_label": material,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                source_index += 1
    if source_index != 250:
        raise RuntimeError(f"Expected 250 validation videos; extracted {source_index}")
    print(f"Wrote {source_index} video records to {output}")


if __name__ == "__main__":
    main()
