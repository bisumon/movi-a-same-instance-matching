#!/usr/bin/env python3
"""Extract deterministic frozen ResNet-18 embeddings for Phase 1 RGB crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.torch_cache.mkdir(parents=True, exist_ok=True)

    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(20260727)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(args.torch_cache.resolve()))
    if args.device == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(device_name)

    rows = [
        json.loads(line)
        for line in args.model_inputs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len({row["observation_id"] for row in rows}):
        raise ValueError("Duplicate observation IDs in model inputs")
    rows.sort(key=lambda row: row["observation_id"])

    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    class CropDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
            row = rows[index]
            path = args.phase1_dir / row["rgb_crop_path"]
            with Image.open(path) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, str(row["observation_id"])

    loader = DataLoader(
        CropDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device_name == "cuda",
        drop_last=False,
    )
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    all_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    forward_per_crop_ms: list[float] = []
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, observation_ids) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device_name == "cuda")
            if device_name == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            embeddings = model(images)
            if device_name == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            forward_per_crop_ms.extend([elapsed_ms / len(observation_ids)] * len(observation_ids))
            array = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
            all_embeddings.append(array)
            all_ids.extend(list(observation_ids))
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"batch {batch_index:03d}/{len(loader):03d}: {len(all_ids):04d}/{len(rows):04d} crops",
                    flush=True,
                )
    wall_seconds = time.perf_counter() - wall_start
    matrix = np.vstack(all_embeddings)
    if matrix.shape != (len(rows), 512) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Unexpected or non-finite embedding matrix: {matrix.shape}")
    if all_ids != [row["observation_id"] for row in rows]:
        raise RuntimeError("Embedding order does not match observation order")
    norms = np.linalg.norm(matrix, axis=1)

    embeddings_path = args.output_dir / "rgb_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        observation_ids=np.asarray(all_ids, dtype="U20"),
        embeddings=matrix,
    )
    checkpoint_name = Path(weights.url).name
    checkpoint_path = args.torch_cache / "checkpoints" / checkpoint_name
    manifest = {
        "pipeline": "MOVi-A Phase 3 frozen RGB embedding extraction",
        "version": VERSION,
        "input": {
            "model_inputs": str(args.model_inputs.resolve()),
            "model_inputs_sha256": sha256(args.model_inputs),
            "phase1_directory": str(args.phase1_dir.resolve()),
            "observation_count": len(rows),
        },
        "encoder": {
            "architecture": "torchvision.models.resnet18",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "weights_url": weights.url,
            "weights_sha256": sha256(checkpoint_path) if checkpoint_path.exists() else None,
            "output_layer": "global average pool; classification fc replaced by identity",
            "embedding_dimension": 512,
            "frozen": True,
            "preprocessing": {
                "resize_shorter_side": 256,
                "center_crop": [224, 224],
                "pixel_scale": "uint8 RGB to [0,1]",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "runtime": {
            "device": device_name,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "latency": {
            "wall_seconds": wall_seconds,
            "wall_ms_per_crop": wall_seconds * 1000.0 / len(rows),
            "forward_ms_per_crop_p50": percentile(forward_per_crop_ms, 0.50),
            "forward_ms_per_crop_p95": percentile(forward_per_crop_ms, 0.95),
            "forward_ms_per_crop_mean": statistics.fmean(forward_per_crop_ms),
            "note": "Forward timings divide batch duration by batch size; wall time includes decoding and preprocessing.",
        },
        "quality_checks": {
            "shape": list(matrix.shape),
            "all_finite": bool(np.isfinite(matrix).all()),
            "embedding_norm_min": float(norms.min()),
            "embedding_norm_median": float(np.median(norms)),
            "embedding_norm_max": float(norms.max()),
            "unique_observation_ids": len(set(all_ids)),
        },
        "output": {
            "filename": embeddings_path.name,
            "sha256": sha256(embeddings_path),
        },
    }
    write_json(args.output_dir / "rgb_embedding_manifest.json", manifest)
    print(f"Complete: {matrix.shape[0]} frozen 512-D embeddings on {device_name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
