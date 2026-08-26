#!/usr/bin/env python3
"""Download seeded MOVi-D/E validation shards and lock 20 pilot videos each."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from movi_de_dataset_adapter import (
    camera_motion_summary,
    get_dataset_spec,
    iter_tfrecords,
    record_video_id,
    scalar,
    sha256,
    validate_record_schema,
)


BUCKET = "kubric-public"
VERSION = "1.0.0"
RESOLUTION = "128x128"
DATASETS = ("movi_d", "movi_e")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--videos-per-dataset", type=int, default=20)
    parser.add_argument("--shards-per-dataset", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def list_objects(prefix: str) -> list[dict[str, Any]]:
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
        f"prefix={urllib.parse.quote(prefix, safe='')}&maxResults=1000"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response).get("items", [])


def list_dataset_files(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = f"tfds/{dataset}/{RESOLUTION}/{VERSION}/"
    shards = list_objects(f"{base}{dataset}-validation.tfrecord-")
    metadata_names = {"dataset_info.json", "features.json", "instances-category.labels.txt"}
    metadata = [item for item in list_objects(base) if Path(item["name"]).name in metadata_names]
    if not shards:
        raise RuntimeError(f"No public validation shards found for {dataset}")
    declared = {int(Path(item["name"]).name.rsplit("-of-", 1)[1]) for item in shards}
    if len(declared) != 1 or len(shards) != declared.pop():
        raise RuntimeError(f"Unexpected public shard listing for {dataset}: {len(shards)} files")
    if {Path(item["name"]).name for item in metadata} != metadata_names:
        raise RuntimeError(f"Missing schema metadata files for {dataset}")
    return sorted(shards, key=lambda item: item["name"]), sorted(metadata, key=lambda item: item["name"])


def seeded_rank(seed: int, dataset: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{value}".encode("utf-8")).hexdigest()


def choose_shards(shards: list[dict[str, Any]], dataset: str, seed: int, count: int) -> list[dict[str, Any]]:
    if count < 1 or count > len(shards):
        raise ValueError(f"--shards-per-dataset must be between 1 and {len(shards)}")
    ranked = sorted(shards, key=lambda item: seeded_rank(seed, dataset, item["name"]))
    return sorted(ranked[:count], key=lambda item: item["name"])


def local_md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode("ascii")


def download(item: dict[str, Any], destination: Path, retries: int = 3) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / Path(item["name"]).name
    expected_size = int(item["size"])
    expected_md5 = item.get("md5Hash")
    if target.is_file() and target.stat().st_size == expected_size:
        if expected_md5 is None or local_md5_base64(target) == expected_md5:
            status = "kept"
        else:
            raise RuntimeError(f"Existing file has the wrong MD5: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".part")
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(
                    f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(item['name'], safe='/')}"
                )
                with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
                    while chunk := response.read(4 * 1024 * 1024):
                        handle.write(chunk)
                if temporary.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"Incomplete download for {target.name}: {temporary.stat().st_size}/{expected_size}"
                    )
                if expected_md5 is not None and local_md5_base64(temporary) != expected_md5:
                    raise RuntimeError(f"MD5 mismatch for {target.name}")
                os.replace(temporary, target)
                status = "downloaded"
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(2**attempt)
    return {
        "filename": target.name,
        "gcs_name": item["name"],
        "size_bytes": expected_size,
        "gcs_md5_base64": expected_md5,
        "sha256": sha256(target),
        "status": status,
    }


def inventory_shards(dataset: str, paths: list[Path], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard, record in iter_tfrecords(paths):
        validate_record_schema(record, dataset)
        video_id = record_video_id(record)
        if video_id in seen:
            raise RuntimeError(f"Duplicate {dataset} video ID {video_id} in downloaded shards")
        seen.add(video_id)
        num_instances = scalar(record, "metadata/num_instances", int)
        is_dynamic = np.asarray(record["instances/is_dynamic"], dtype=bool).reshape(-1)
        visibility = np.asarray(record["instances/visibility"], dtype=np.float64).reshape(num_instances, -1)
        positions = np.asarray(record["camera/positions"], dtype=np.float64).reshape(-1, 3)
        quaternions = np.asarray(record["camera/quaternions"], dtype=np.float64).reshape(-1, 4)
        motion = camera_motion_summary(positions, quaternions)
        rows.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "source_shard": shard.name,
                "source_record_index": sum(row["source_shard"] == shard.name for row in rows),
                "num_instances": num_instances,
                "dynamic_instance_count": int(is_dynamic.sum()),
                "static_instance_count": int((~is_dynamic).sum()),
                "mean_visibility_pixels": float(np.mean(visibility)),
                "camera_translation_scene_units": motion["translation_start_to_end_scene_units"],
                "camera_rotation_degrees": motion["rotation_start_to_end_degrees"],
                "normalized_camera_translation": motion["normalized_start_to_end_translation"],
                "seeded_tie_break": seeded_rank(seed, dataset, video_id),
            }
        )
    return rows


def normalized_feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    fields = (
        "num_instances",
        "dynamic_instance_count",
        "mean_visibility_pixels",
        "camera_translation_scene_units",
        "camera_rotation_degrees",
    )
    values = np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=np.float64)
    minimum = np.min(values, axis=0)
    span = np.max(values, axis=0) - minimum
    span[span == 0] = 1.0
    return (values - minimum) / span


def select_diverse_pilot(
    inventory: list[dict[str, Any]], dataset: str, seed: int, count: int
) -> list[dict[str, Any]]:
    """Seeded maximin selection covering scene complexity and camera motion."""
    if len(inventory) < count:
        raise RuntimeError(
            f"Downloaded {len(inventory)} {dataset} candidates but need {count}; increase --shards-per-dataset"
        )
    features = normalized_feature_matrix(inventory)
    tie_order = sorted(range(len(inventory)), key=lambda i: inventory[i]["seeded_tie_break"])
    selected = [tie_order[0]]
    remaining = set(range(len(inventory))) - set(selected)
    while len(selected) < count:
        best_index = None
        best_distance = -1.0
        for index in remaining:
            distance = float(np.min(np.linalg.norm(features[index] - features[selected], axis=1)))
            if distance > best_distance + 1e-12:
                best_index, best_distance = index, distance
            elif abs(distance - best_distance) <= 1e-12:
                assert best_index is not None
                if inventory[index]["seeded_tie_break"] < inventory[best_index]["seeded_tie_break"]:
                    best_index = index
        assert best_index is not None
        selected.append(best_index)
        remaining.remove(best_index)
    result = []
    for rank, index in enumerate(selected, start=1):
        source = inventory[index]
        result.append(
            {
                "dataset": dataset,
                "video_id": source["video_id"],
                "split": "pilot",
                "pilot_selection_rank": rank,
                "seed": seed,
                "source_shard": source["source_shard"],
                "confirmatory_test_eligible": False,
                "selection_rule": "seeded maximin coverage of object count, dynamic count, visibility, camera translation, and camera rotation",
            }
        )
    return sorted(result, key=lambda row: row["video_id"])


def main() -> int:
    args = parse_args()
    if args.videos_per_dataset < 1:
        raise ValueError("--videos-per-dataset must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifests_dir.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {
        "pipeline": "MOVi-D/E seeded pilot download",
        "version": VERSION,
        "resolution": RESOLUTION,
        "seed": args.seed,
        "videos_per_dataset": args.videos_per_dataset,
        "shards_per_dataset": args.shards_per_dataset,
        "datasets": {},
    }

    for dataset in DATASETS:
        spec = get_dataset_spec(dataset)
        all_shards, metadata = list_dataset_files(dataset)
        chosen_shards = choose_shards(all_shards, dataset, args.seed, args.shards_per_dataset)
        selected_indices = [
            int(Path(item["name"]).name.split(".tfrecord-", 1)[1].split("-of-", 1)[0])
            for item in chosen_shards
        ]
        destination = args.output_root / f"{dataset}_validation_partial"
        items = metadata + chosen_shards
        print(
            f"{spec.display_name}: selected validation shard indices {selected_indices}; "
            f"download={sum(int(item['size']) for item in items) / 2**20:.1f} MiB",
            flush=True,
        )
        downloads: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(items))) as executor:
            futures = {executor.submit(download, item, destination): item for item in items}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                downloads.append(result)
                print(f"{dataset}: {result['status']} {result['filename']}", flush=True)
        shard_paths = [destination / Path(item["name"]).name for item in chosen_shards]
        inventory = inventory_shards(dataset, shard_paths, args.seed)
        pilot = select_diverse_pilot(inventory, dataset, args.seed, args.videos_per_dataset)
        inventory_path = args.manifests_dir / f"pilot_{dataset}_candidate_inventory.jsonl"
        pilot_path = args.manifests_dir / f"pilot_{dataset}_{args.videos_per_dataset}.jsonl"
        write_jsonl(inventory_path, sorted(inventory, key=lambda row: row["video_id"]))
        write_jsonl(pilot_path, pilot)
        dataset_manifest = {
            "dataset": dataset,
            "display_name": spec.display_name,
            "bucket": BUCKET,
            "gcs_prefix": f"tfds/{dataset}/{RESOLUTION}/{VERSION}/",
            "public_validation_shard_count": len(all_shards),
            "selected_shard_indices": selected_indices,
            "downloaded_candidate_videos": len(inventory),
            "selected_pilot_videos": len(pilot),
            "confirmatory_test_eligible": False,
            "files": sorted(downloads, key=lambda row: row["filename"]),
            "candidate_inventory": {
                "path": str(inventory_path.resolve()),
                "sha256": sha256(inventory_path),
            },
            "pilot_manifest": {
                "path": str(pilot_path.resolve()),
                "sha256": sha256(pilot_path),
            },
        }
        write_json(destination / "pilot_download_manifest.json", dataset_manifest)
        overall["datasets"][dataset] = dataset_manifest

    overall_path = args.output_root / "pilot_download_manifest.json"
    write_json(overall_path, overall)
    print(f"Complete: {overall_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
