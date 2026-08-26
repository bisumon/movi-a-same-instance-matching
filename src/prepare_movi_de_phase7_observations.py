#!/usr/bin/env python3
"""Filter confirmatory Phase 1 outputs to observations used by locked Phase 5 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(args.pairs)
    if len(pairs) != 10000 or len({row["pair_id"] for row in pairs}) != 10000:
        raise ValueError("Expected exactly 10,000 unique locked pairs")
    required = {str(row[key]) for row in pairs for key in ("observation_id_a", "observation_id_b")}
    model_source = args.phase1_dir / "model_inputs.jsonl"
    index_source = args.phase1_dir / "observation_index.jsonl"
    model = [row for row in read_jsonl(model_source) if str(row["observation_id"]) in required]
    index = [row for row in read_jsonl(index_source) if str(row["observation_id"]) in required]
    if {row["observation_id"] for row in model} != required or {row["observation_id"] for row in index} != required:
        raise RuntimeError("At least one locked pair endpoint is absent from Phase 1 outputs")
    model.sort(key=lambda row: row["observation_id"])
    index.sort(key=lambda row: row["observation_id"])
    model_output = args.output_dir / "pair_model_inputs.jsonl"
    index_output = args.output_dir / "pair_observation_index.jsonl"
    write_jsonl(model_output, model)
    write_jsonl(index_output, index)
    manifest = {
        "pipeline": "MOVi-D/E Phase 7 locked-pair observation filter", "version": "1.0.0",
        "counts": {"pairs": len(pairs), "unique_pair_observations": len(required)},
        "inputs": {
            "pairs_sha256": sha256(args.pairs), "model_inputs_sha256": sha256(model_source),
            "observation_index_sha256": sha256(index_source),
        },
        "outputs": {
            model_output.name: sha256(model_output), index_output.name: sha256(index_output),
        },
        "checks": {
            "all_pair_endpoints_present": True, "model_index_ids_match": True,
            "filter_uses_pair_membership_not_labels_or_scores": True,
        },
    }
    (args.output_dir / "phase7_observation_filter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Complete: {len(required)} unique observations used by 10,000 locked pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
