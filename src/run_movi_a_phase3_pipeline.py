#!/usr/bin/env python3
"""Run all Phase 3 stages with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--phase2-pairs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    script_dir = Path(__file__).resolve().parent
    embeddings_dir = args.output_dir / "rgb_embeddings"
    features_dir = args.output_dir / "pair_features"
    baselines_dir = args.output_dir / "baselines"
    run(
        [
            sys.executable,
            str(script_dir / "extract_movi_a_phase3_rgb_embeddings.py"),
            "--model-inputs",
            str(args.phase1_dir / "model_inputs.jsonl"),
            "--phase1-dir",
            str(args.phase1_dir),
            "--output-dir",
            str(embeddings_dir),
            "--torch-cache",
            str(args.torch_cache),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "build_movi_a_phase3_pair_features.py"),
            "--model-inputs",
            str(args.phase1_dir / "model_inputs.jsonl"),
            "--observation-index",
            str(args.phase1_dir / "observation_index.jsonl"),
            "--pairs",
            str(args.phase2_pairs_dir / "all_pairs.jsonl"),
            "--embeddings",
            str(embeddings_dir / "rgb_embeddings.npz"),
            "--output-dir",
            str(features_dir),
        ]
    )
    run(
        [
            sys.executable,
            str(script_dir / "run_movi_a_phase3_baselines.py"),
            "--features",
            str(features_dir / "phase3_pair_features.npz"),
            "--feature-manifest",
            str(features_dir / "phase3_feature_manifest.json"),
            "--pairs",
            str(args.phase2_pairs_dir / "all_pairs.jsonl"),
            "--output-dir",
            str(baselines_dir),
            "--seed",
            str(args.seed),
        ]
    )
    print(f"Phase 3 complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
