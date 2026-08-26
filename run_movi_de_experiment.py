#!/usr/bin/env python3
"""Reproduce the locked MOVi-D/E camera-pose experiments from frozen inputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/movi_de_reproduction"))
    parser.add_argument("--source-run", type=Path, help="Reuse a completed Phase 7 directory instead of regenerating it.")
    parser.add_argument("--tfrecord-root", type=Path, help="Directory containing movi_d_validation/ and movi_e_validation/. If omitted, all public validation shards are downloaded.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    source = root / "src"
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config = root / "configs/movi_de_phase6_systems.json"

    if args.run_tests:
        run([sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-p", "test_*.py", "-v"])

    if args.source_run:
        source_run = args.source_run if args.source_run.is_absolute() else root / args.source_run
    else:
        source_run = run_dir / "phase7"
        tfrecord_root = args.tfrecord_root
        if tfrecord_root is not None:
            tfrecord_root = tfrecord_root if tfrecord_root.is_absolute() else root / tfrecord_root
        else:
            tfrecord_root = run_dir / "data"
            download_manifests = run_dir / "download_manifests"
            run_if_missing(download_manifests / "pilot_download_manifest.json", [
                sys.executable, str(source / "download_movi_de_pilot.py"),
                "--output-root", str(tfrecord_root), "--manifests-dir", str(download_manifests),
                "--videos-per-dataset", "20", "--shards-per-dataset", "16",
                "--workers", str(args.download_workers), "--seed", "20260825",
            ])
        for dataset in ("movi_d", "movi_e"):
            raw_dir = tfrecord_root / f"{dataset}_validation"
            if len(list(raw_dir.glob(f"{dataset}-validation.tfrecord-*-of-00016"))) != 16:
                raise FileNotFoundError(f"Expected all 16 {dataset} validation shards in {raw_dir}")
            phase1 = source_run / f"{dataset}_phase1"
            run_if_missing(phase1 / "phase1_adapter_manifest.json", [
                sys.executable, str(source / "extract_movi_de_phase1.py"), "--dataset", dataset,
                "--tfrecord-dir", str(raw_dir),
                "--video-manifest", str(root / f"manifests/movi_de/confirmatory_{dataset}_150.jsonl"),
                "--output-dir", str(phase1),
            ])
            pairs = root / f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl"
            observations = source_run / f"{dataset}_pair_observations"
            run_if_missing(observations / "phase7_observation_filter_manifest.json", [
                sys.executable, str(source / "prepare_movi_de_phase7_observations.py"),
                "--phase1-dir", str(phase1), "--pairs", str(pairs), "--output-dir", str(observations),
            ])
            embeddings = source_run / f"{dataset}_rgb_embeddings"
            run_if_missing(embeddings / "rgb_embedding_manifest.json", [
                sys.executable, str(source / "extract_movi_a_phase3_rgb_embeddings.py"),
                "--model-inputs", str(observations / "pair_model_inputs.jsonl"),
                "--phase1-dir", str(phase1), "--output-dir", str(embeddings),
                "--torch-cache", str(run_dir / "torch_cache"), "--batch-size", str(args.batch_size),
                "--device", args.device,
            ])
            pose = source_run / f"{dataset}_pose_noise"
            run_if_missing(pose / f"{dataset}_phase7_pose_noise_results.json", [
                sys.executable, str(source / "run_movi_de_phase7_pose_noise.py"), "--dataset", dataset,
                "--model-inputs", str(observations / "pair_model_inputs.jsonl"),
                "--observation-index", str(observations / "pair_observation_index.jsonl"),
                "--pairs", str(pairs), "--embeddings", str(embeddings / "rgb_embeddings.npz"),
                "--system-config", str(config), "--output-dir", str(pose),
            ])

    datasets = {
        "movi_e": ("regime1", "build_movi_e_phase8_in_domain_features.py", "run_movi_e_phase8_in_domain.py"),
        "movi_d": ("regime2", "build_movi_d_phase8_in_domain_features.py", "run_movi_d_phase8_in_domain.py"),
    }
    outputs: dict[str, Path] = {}
    for dataset, (regime, builder, runner) in datasets.items():
        pairs = root / f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl"
        observation_dir = source_run / f"{dataset}_pair_observations"
        pose_dir = source_run / f"{dataset}_pose_noise"
        features_dir = run_dir / regime / "features"
        prefix = "movi_e_phase8" if dataset == "movi_e" else "movi_d_phase8"
        feature_file = features_dir / f"{prefix}_in_domain_features.npz"
        feature_manifest = features_dir / f"{prefix}_feature_manifest.json"
        run_if_missing(
            feature_manifest,
            [
                sys.executable, str(source / builder),
                "--phase7-features", str(pose_dir / f"{dataset}_phase7_pose_noise_features.npz"),
                "--model-inputs", str(observation_dir / "pair_model_inputs.jsonl"),
                "--observation-index", str(observation_dir / "pair_observation_index.jsonl"),
                "--pairs", str(pairs), "--system-config", str(config), "--output-dir", str(features_dir),
            ],
        )
        output_dir = run_dir / regime / f"in_domain_{dataset}"
        output_prefix = f"{dataset}_phase8_{regime}"
        run_if_missing(
            output_dir / f"{output_prefix}_manifest.json",
            [
                sys.executable, str(source / runner), "--features", str(feature_file),
                "--feature-manifest", str(feature_manifest), "--pairs", str(pairs),
                "--system-config", str(config),
                "--phase7-d-lock", str(pose_dir / f"{dataset}_phase7_clean_D_locked_config.json"),
                "--phase7-predictions", str(pose_dir / f"{dataset}_phase7_pose_noise_predictions.jsonl"),
                "--output-dir", str(output_dir), "--bootstrap-replicates", str(args.bootstrap_replicates),
            ],
        )
        outputs[dataset] = output_dir

    transfer = run_dir / "regime3" / "d_to_e_transfer"
    run_if_missing(
        transfer / "movi_d_to_e_phase8_regime3_manifest.json",
        [
            sys.executable, str(source / "run_movi_d_to_e_phase8_transfer.py"),
            "--source-models", str(outputs["movi_d"] / "movi_d_phase8_regime2_models.joblib"),
            "--source-lock", str(outputs["movi_d"] / "movi_d_phase8_regime2_locked_config.json"),
            "--source-freeze", str(root / "manifests/movi_de/phase8_regime2_in_domain_movi_d_freeze.json"),
            "--target-features", str(run_dir / "regime1/features/movi_e_phase8_in_domain_features.npz"),
            "--target-feature-manifest", str(run_dir / "regime1/features/movi_e_phase8_feature_manifest.json"),
            "--target-pairs", str(root / "manifests/pairs/movi_de/movi_e_all_pairs.jsonl"),
            "--system-config", str(config),
            "--target-in-domain-predictions", str(outputs["movi_e"] / "movi_e_phase8_regime1_predictions.jsonl"),
            "--target-in-domain-lock", str(outputs["movi_e"] / "movi_e_phase8_regime1_locked_config.json"),
            "--output-dir", str(transfer), "--bootstrap-replicates", str(args.bootstrap_replicates),
        ],
    )
    print(f"Complete: MOVi-D/E in-domain and transfer outputs are in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
