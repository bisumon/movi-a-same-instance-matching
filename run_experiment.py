#!/usr/bin/env python3
"""Run the locked MOVi-A experiment from public TFRecords through Phase 5."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/main"))
    parser.add_argument(
        "--tfrecord-dir",
        type=Path,
        help="Use an existing directory containing all 16 MOVi-A validation shards.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def run_if_missing(marker: Path, command: list[str]) -> None:
    if marker.is_file():
        print(f"kept completed stage: {marker}", flush=True)
        return
    run(command)
    if not marker.is_file():
        raise RuntimeError(f"Stage completed without expected marker: {marker}")


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parent
    source = repository / "src"
    config = json.loads((repository / "configs" / "protocol.json").read_text(encoding="utf-8"))
    if args.seed != int(config["seed"]):
        raise ValueError(f"This release is locked to seed {config['seed']}; received {args.seed}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    tfrecord_dir = args.tfrecord_dir or args.run_dir / "data" / "movi_a_validation"
    if args.tfrecord_dir is None:
        run_if_missing(
            tfrecord_dir / "download_manifest.json",
            [
                sys.executable,
                str(source / "download_movi_a_validation.py"),
                "--output-dir",
                str(tfrecord_dir),
                "--workers",
                str(args.download_workers),
            ],
        )
    shards = sorted(tfrecord_dir.glob("movi_a-validation.tfrecord-*-of-00016"))
    if len(shards) != 16:
        raise FileNotFoundError(f"Expected 16 MOVi-A validation shards in {tfrecord_dir}; found {len(shards)}")

    phase1 = args.run_dir / "phase1"
    run_if_missing(
        phase1 / "phase1_manifest.json",
        [
            sys.executable,
            str(source / "extract_movi_a_phase1.py"),
            "--tfrecord-dir",
            str(tfrecord_dir),
            "--video-splits",
            str(repository / "manifests" / "selection" / "video_splits.jsonl"),
            "--output-dir",
            str(phase1),
            "--crop-size",
            str(config["phase1"]["crop_size"]),
            "--crop-padding",
            str(config["phase1"]["crop_padding"]),
            "--min-visibility-pixels",
            str(config["phase1"]["min_visibility_pixels"]),
            "--min-mask-area",
            str(config["phase1"]["min_mask_area"]),
        ],
    )

    phase2_split = args.run_dir / "phase2_split"
    run_if_missing(
        phase2_split / "phase2_split_manifest.json",
        [
            sys.executable,
            str(source / "prepare_movi_a_phase2_split.py"),
            "--selected-videos",
            str(repository / "manifests" / "selection" / "selected_50.jsonl"),
            "--video-splits",
            str(repository / "manifests" / "selection" / "video_splits.jsonl"),
            "--observation-index",
            str(phase1 / "observation_index.jsonl"),
            "--output-dir",
            str(phase2_split),
        ],
    )

    phase2_pairs = args.run_dir / "phase2_pairs"
    run_if_missing(
        phase2_pairs / "phase2_pair_manifest.json",
        [
            sys.executable,
            str(source / "generate_movi_a_phase2_pairs.py"),
            "--locked-video-splits",
            str(phase2_split / "locked_video_splits.jsonl"),
            "--observation-index",
            str(phase1 / "observation_index.jsonl"),
            "--model-inputs",
            str(phase1 / "model_inputs.jsonl"),
            "--instance-metadata",
            str(phase1 / "instance_metadata.jsonl"),
            "--output-dir",
            str(phase2_pairs),
            "--seed",
            str(args.seed),
        ],
    )

    phase3 = args.run_dir / "phase3"
    run_if_missing(
        phase3 / "baselines" / "phase3_baseline_manifest.json",
        [
            sys.executable,
            str(source / "run_movi_a_phase3_pipeline.py"),
            "--phase1-dir",
            str(phase1),
            "--phase2-pairs-dir",
            str(phase2_pairs),
            "--output-dir",
            str(phase3),
            "--torch-cache",
            str(args.run_dir / "torch_cache"),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ],
    )

    phase4 = args.run_dir / "phase4"
    run_if_missing(
        phase4 / "phase4_evaluation_manifest.json",
        [
            sys.executable,
            str(source / "evaluate_movi_a_phase4.py"),
            "--predictions",
            str(phase3 / "baselines" / "phase3_predictions.jsonl"),
            "--locked-config",
            str(phase3 / "baselines" / "phase3_locked_config.json"),
            "--phase3-results",
            str(phase3 / "baselines" / "phase3_results.json"),
            "--rgb-manifest",
            str(phase3 / "rgb_embeddings" / "rgb_embedding_manifest.json"),
            "--pairs",
            str(phase2_pairs / "all_pairs.jsonl"),
            "--model-inputs",
            str(phase1 / "model_inputs.jsonl"),
            "--diagnostics",
            str(phase1 / "diagnostics.jsonl"),
            "--output-dir",
            str(phase4),
            "--bootstrap-replicates",
            str(config["phase4"]["bootstrap_replicates"]),
            "--seed",
            str(args.seed),
        ],
    )

    phase5 = args.run_dir / "phase5"
    run_if_missing(
        phase5 / "phase5_selection_manifest.json",
        [
            sys.executable,
            str(source / "select_movi_a_phase5_errors.py"),
            "--predictions",
            str(phase3 / "baselines" / "phase3_predictions.jsonl"),
            "--locked-config",
            str(phase3 / "baselines" / "phase3_locked_config.json"),
            "--phase4-results",
            str(phase4 / "phase4_results.json"),
            "--pairs",
            str(phase2_pairs / "all_pairs.jsonl"),
            "--model-inputs",
            str(phase1 / "model_inputs.jsonl"),
            "--diagnostics",
            str(phase1 / "diagnostics.jsonl"),
            "--phase1-dir",
            str(phase1),
            "--output-dir",
            str(phase5),
        ],
    )
    print(f"Complete: locked experiment outputs are in {args.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
