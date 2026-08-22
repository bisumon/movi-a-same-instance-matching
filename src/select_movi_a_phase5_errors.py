#!/usr/bin/env python3
"""Select and render a balanced qualitative review of Phase 3 test errors."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import platform
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


VERSION = "1.0.0"
CONFIG_ORDER = ("A_rgb_only", "B_rgb_2d", "C_rgb_2d_3d", "geometry_only")
ERROR_ORDER = ("false_positive", "false_negative")
GAP_ORDER = ("short", "medium", "long")


@dataclass(frozen=True)
class Slot:
    configuration: str
    error_type: str
    temporal_gap_bin: str
    negative_difficulty: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    parser.add_argument("--phase4-results", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inline-fragment", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def classify_error(label: int, score: float, threshold: float) -> str | None:
    predicted = int(score >= threshold)
    if label == 0 and predicted == 1:
        return "false_positive"
    if label == 1 and predicted == 0:
        return "false_negative"
    return None


def assign_tertile(value: float, cutpoints: Sequence[float]) -> str:
    low, high = map(float, cutpoints)
    return "low" if value <= low else "medium" if value <= high else "high"


def selection_slots() -> list[Slot]:
    """Create 24 exact method/error/gap slots and 6/6 hard/easy FP slots."""
    slots: list[Slot] = []
    for gap_index, gap in enumerate(GAP_ORDER):
        for error_type in ERROR_ORDER:
            for method_index, configuration in enumerate(CONFIG_ORDER):
                difficulty = None
                if error_type == "false_positive":
                    difficulty = "hard" if (gap_index + method_index) % 2 == 0 else "easy"
                slots.append(Slot(configuration, error_type, gap, difficulty))
    return slots


def select_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_pair_ids: set[str] = set()
    visibility_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    for slot in selection_slots():
        eligible = [
            row
            for row in candidates
            if row["configuration"] == slot.configuration
            and row["error_type"] == slot.error_type
            and row["temporal_gap_bin"] == slot.temporal_gap_bin
            and (slot.negative_difficulty is None or row["negative_difficulty"] == slot.negative_difficulty)
            and row["pair_id"] not in selected_pair_ids
        ]
        if not eligible:
            raise ValueError(f"No unique candidate available for slot {slot}")

        def ranking(row: dict[str, Any]) -> tuple[Any, ...]:
            diversity_penalty = (
                visibility_counts[row["visibility_stratum"]]
                + motion_counts[row["motion_stratum"]]
                + video_counts[str(row["video_id"])]
            )
            maximum_reuse = max(
                visibility_counts[row["visibility_stratum"]],
                motion_counts[row["motion_stratum"]],
                video_counts[str(row["video_id"])],
            )
            return (diversity_penalty, maximum_reuse, -float(row["threshold_margin"]), str(row["pair_id"]))

        chosen = min(eligible, key=ranking).copy()
        chosen["selection_slot"] = {
            "configuration": slot.configuration,
            "error_type": slot.error_type,
            "temporal_gap_bin": slot.temporal_gap_bin,
            "negative_difficulty": slot.negative_difficulty,
        }
        selected.append(chosen)
        selected_pair_ids.add(str(chosen["pair_id"]))
        visibility_counts[chosen["visibility_stratum"]] += 1
        motion_counts[chosen["motion_stratum"]] += 1
        video_counts[str(chosen["video_id"])] += 1
    return selected


def edge_truncated(model_row: dict[str, Any]) -> bool:
    x0, y0, x1, y1 = map(int, model_row["tight_bbox_xyxy"])
    intrinsics = model_row["intrinsics"]
    return x0 <= 0 or y0 <= 0 or x1 >= int(intrinsics["image_width"]) or y1 >= int(intrinsics["image_height"])


def diagnose(row: dict[str, Any], depth_bias_cutpoint: float, low_fill_cutpoint: float) -> tuple[str, str]:
    method = str(row["configuration"])
    low_or_truncated = (
        row["visibility_stratum"] == "low"
        and float(row["min_mask_fill_fraction"]) <= low_fill_cutpoint
    ) or bool(row["edge_truncated"])
    biased_depth = float(row["max_abs_radial_depth_error"]) >= depth_bias_cutpoint
    if row["error_type"] == "false_positive":
        if method in ("C_rgb_2d_3d", "geometry_only") and biased_depth:
            return "biased_depth_estimate", "Biased depth estimate makes two different objects look geometrically compatible."
        if row["negative_difficulty"] == "hard":
            return "matched_attribute_distractor", "Matched-attribute distractor remains similar in appearance and pair controls."
        if low_or_truncated:
            return "occlusion_truncated_mask", "Low visibility or a truncated mask hides cues that separate this distractor."
        return "easy_distractor_similarity", "A different-attribute distractor still receives an unusually high similarity score."
    if low_or_truncated:
        return "occlusion_truncated_mask", "Occlusion- or image-edge-truncated mask removes reliable same-instance cues."
    if row["temporal_gap_bin"] == "long" or row["motion_stratum"] == "high":
        return "long_gap_motion", "Long-gap motion or viewpoint change lowers similarity for the same instance."
    if method in ("C_rgb_2d_3d", "geometry_only") and biased_depth:
        return "biased_depth_estimate", "Biased depth estimate shifts the recovered geometry of the same object."
    if method == "A_rgb_only":
        return "appearance_change", "Appearance change across frames overwhelms the RGB-only similarity signal."
    return "control_drift", "The same object's 2D or 3D controls drift enough to cross the fixed threshold."


def image_data_url(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=78, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def score_table(row: dict[str, Any]) -> str:
    cells = []
    for configuration in CONFIG_ORDER:
        score = float(row["scores"][configuration])
        threshold = float(row["thresholds"][configuration])
        prediction = "match" if score >= threshold else "non-match"
        correct = int(score >= threshold) == int(row["label"])
        marker = "correct" if correct else "error"
        assigned = " · selected" if configuration == row["configuration"] else ""
        cells.append(
            "<tr>"
            f"<td><code>{escape(configuration)}</code>{assigned}</td>"
            f"<td class=\"text-end\">{score:.3f}</td>"
            f"<td class=\"text-end\">{threshold:.3f}</td>"
            f"<td>{prediction} · {marker}</td>"
            "</tr>"
        )
    return "".join(cells)


def build_fragment(rows: Sequence[dict[str, Any]]) -> str:
    items = []
    for index, row in enumerate(rows, start=1):
        left_url = image_data_url(Path(row["source_crop_path_a"]))
        right_url = image_data_url(Path(row["source_crop_path_b"]))
        controls = row["controls"]
        item = f"""
<article class="p5-item" data-method="{escape(row['configuration'])}" data-error="{escape(row['error_type'])}" data-gap="{escape(row['temporal_gap_bin'])}">
  <div class="viz-row p5-heading">
    <strong>#{index} · <code>{escape(row['pair_id'])}</code></strong>
    <span class="viz-badge">{escape(row['configuration'])}</span>
    <span class="viz-badge">{escape(row['error_type'].replace('_', ' '))}</span>
    <span class="viz-badge">{escape(row['temporal_gap_bin'])} gap</span>
  </div>
  <p class="p5-diagnosis"><strong>{escape(row['diagnosis_category'].replace('_', ' '))}:</strong> {escape(row['diagnosis'])}</p>
  <div class="p5-layout">
    <div class="p5-crops">
      <figure><img src="{left_url}" alt="Left RGB crop for pair {escape(row['pair_id'])}, frame {row['frame_index_a']}"><figcaption>Frame {row['frame_index_a']} · instance {row['instance_index_a']}</figcaption></figure>
      <figure><img src="{right_url}" alt="Right RGB crop for pair {escape(row['pair_id'])}, frame {row['frame_index_b']}"><figcaption>Frame {row['frame_index_b']} · instance {row['instance_index_b']}</figcaption></figure>
    </div>
    <div class="table-responsive">
      <table class="table table-sm"><thead><tr><th>Method</th><th class="text-end">Score</th><th class="text-end">Threshold</th><th>Decision</th></tr></thead><tbody>{score_table(row)}</tbody></table>
    </div>
  </div>
  <div class="p5-controls text-small">
    <span>Truth: <strong>{'match' if row['label'] else 'non-match'}</strong></span>
    <span>Relation: {escape(str(row['attribute_relation']))}</span>
    <span>Gap: {row['temporal_gap']} frames</span>
    <span>Visibility: {controls['visibility_a']}/{controls['visibility_b']} ({escape(row['visibility_stratum'])})</span>
    <span>Mask area: {controls['mask_area_a']}/{controls['mask_area_b']} px</span>
    <span>Crop: {controls['padded_crop_width_a']}×{controls['padded_crop_height_a']} / {controls['padded_crop_width_b']}×{controls['padded_crop_height_b']} px</span>
    <span>Motion: {row['pair_motion_speed']:.2f} ({escape(row['motion_stratum'])})</span>
    <span>Max |depth error|: {row['max_abs_radial_depth_error']:.2f}</span>
  </div>
</article>"""
        items.append(item)
    return f"""<div id="phase5-error-gallery">
<style>
#phase5-error-gallery{{color:var(--foreground);background:transparent}}
#phase5-error-gallery .p5-toolbar{{margin-bottom:1rem}}
#phase5-error-gallery .p5-item{{padding:1rem 0;border-top:1px solid var(--border)}}
#phase5-error-gallery .p5-heading{{justify-content:flex-start}}
#phase5-error-gallery .p5-diagnosis{{margin:.65rem 0}}
#phase5-error-gallery .p5-layout{{display:grid;grid-template-columns:minmax(250px,.8fr) minmax(360px,1.2fr);gap:1rem;align-items:start}}
#phase5-error-gallery .p5-crops{{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}}
#phase5-error-gallery figure{{margin:0}}
#phase5-error-gallery img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:var(--muted)}}
#phase5-error-gallery figcaption{{margin-top:.3rem;color:var(--muted-foreground)}}
#phase5-error-gallery .p5-controls{{display:flex;flex-wrap:wrap;gap:.35rem 1rem;margin-top:.7rem;color:var(--muted-foreground)}}
#phase5-error-gallery .p5-empty{{display:none;padding:1rem 0;color:var(--muted-foreground)}}
@media(max-width:700px){{#phase5-error-gallery .p5-layout{{grid-template-columns:1fr}}}}
</style>
<div class="viz-controls p5-toolbar">
  <label class="form-label">Method<select id="p5-method" class="form-select"><option value="all">All methods</option>{''.join(f'<option value="{escape(c)}">{escape(c)}</option>' for c in CONFIG_ORDER)}</select></label>
  <label class="form-label">Error<select id="p5-error" class="form-select"><option value="all">Both errors</option><option value="false_positive">False positive</option><option value="false_negative">False negative</option></select></label>
  <label class="form-label">Gap<select id="p5-gap" class="form-select"><option value="all">All gaps</option>{''.join(f'<option value="{g}">{g.title()}</option>' for g in GAP_ORDER)}</select></label>
  <span id="p5-count" aria-live="polite">Showing {len(rows)} of {len(rows)}</span>
</div>
<div id="p5-items">{''.join(items)}</div>
<p id="p5-empty" class="p5-empty">No selected items match these filters.</p>
<script>
(() => {{
  const root = document.getElementById('phase5-error-gallery');
  const method = root.querySelector('#p5-method');
  const error = root.querySelector('#p5-error');
  const gap = root.querySelector('#p5-gap');
  const count = root.querySelector('#p5-count');
  const empty = root.querySelector('#p5-empty');
  const items = Array.from(root.querySelectorAll('.p5-item'));
  function update() {{
    let visible = 0;
    items.forEach(item => {{
      const show = (method.value === 'all' || item.dataset.method === method.value)
        && (error.value === 'all' || item.dataset.error === error.value)
        && (gap.value === 'all' || item.dataset.gap === gap.value);
      item.hidden = !show;
      if (show) visible += 1;
    }});
    count.textContent = `Showing ${{visible}} of ${{items.length}}`;
    empty.style.display = visible ? 'none' : 'block';
  }}
  [method, error, gap].forEach(control => control.addEventListener('change', update));
  update();
}})();
</script>
</div>
"""


def standalone_html(fragment: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-A Phase 5 error gallery</title>
<style>:root{{--background:#fff;--foreground:#1f2933;--card:#fff;--card-foreground:#1f2933;--primary:#253858;--primary-foreground:#fff;--secondary:#edf2f7;--secondary-foreground:#1f2933;--muted:#edf2f7;--muted-foreground:#52606d;--accent:#e6eef8;--accent-foreground:#1f2933;--border:#cbd2d9;--input:#cbd2d9;--ring:#4c78a8;font-family:system-ui,sans-serif}}body{{max-width:1180px;margin:0 auto;padding:20px;background:var(--background);color:var(--foreground)}}.viz-row,.viz-controls{{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center}}.viz-badge{{padding:.15rem .5rem;border-radius:999px;background:var(--secondary);color:var(--secondary-foreground)}}.form-label{{display:grid;gap:.2rem}}.form-select{{padding:.35rem;border:1px solid var(--input);background:var(--background);color:var(--foreground)}}.table{{border-collapse:collapse;width:100%}}.table th,.table td{{padding:.35rem;border-bottom:1px solid var(--border);text-align:left}}.text-end{{text-align:right!important}}.text-small,figcaption{{font-size:.85rem}}.table-responsive{{overflow-x:auto}}</style>
</head><body>{fragment}</body></html>"""


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir()

    predictions = read_jsonl(args.predictions)
    pairs = read_jsonl(args.pairs)
    model_inputs = read_jsonl(args.model_inputs)
    diagnostics = read_jsonl(args.diagnostics)
    locked = json.loads(args.locked_config.read_text(encoding="utf-8"))
    phase4 = json.loads(args.phase4_results.read_text(encoding="utf-8"))
    pair_by_id = {str(row["pair_id"]): row for row in pairs}
    model_by_id = {str(row["observation_id"]): row for row in model_inputs}
    diagnostic_by_id = {str(row["observation_id"]): row for row in diagnostics}
    if len(predictions) != 10_000 or len(pair_by_id) != 10_000:
        raise ValueError("Expected 10,000 predictions and pairs")

    thresholds = {
        configuration: float(locked["configurations"][configuration]["f1_threshold_selected_on_dev"])
        for configuration in CONFIG_ORDER
    }
    visibility_cutpoints = phase4["train_defined_strata"]["visibility"]["tertile_cutpoints"]
    motion_cutpoints = phase4["train_defined_strata"]["motion"]["tertile_cutpoints"]

    pair_diagnostics: dict[str, dict[str, Any]] = {}
    training_depth_errors: list[float] = []
    training_fill_fractions: list[float] = []
    for pair_id, pair in pair_by_id.items():
        endpoint_ids = (str(pair["observation_id_a"]), str(pair["observation_id_b"]))
        model_rows = [model_by_id[obs_id] for obs_id in endpoint_ids]
        diagnostic_rows = [diagnostic_by_id[obs_id] for obs_id in endpoint_ids]
        speed = float(np.mean([np.linalg.norm(row["gt_world_velocity_xyz"]) for row in diagnostic_rows]))
        values = {
            "pair_visibility": min(float(row["visibility"]) for row in model_rows),
            "pair_motion_speed": speed,
            "min_mask_fill_fraction": min(float(row["mask_fill_fraction"]) for row in model_rows),
            "max_abs_radial_depth_error": max(abs(float(row["radial_depth_error"])) for row in diagnostic_rows),
            "max_visible_surface_center_error": max(float(row["visible_surface_center_error_l2"]) for row in diagnostic_rows),
            "edge_truncated": any(edge_truncated(row) for row in model_rows),
            "model_rows": model_rows,
        }
        pair_diagnostics[pair_id] = values
        if pair["split"] == "train":
            training_depth_errors.append(values["max_abs_radial_depth_error"])
            training_fill_fractions.append(values["min_mask_fill_fraction"])
    depth_bias_cutpoint = float(np.quantile(training_depth_errors, 0.80))
    low_fill_cutpoint = float(np.quantile(training_fill_fractions, 0.20))

    candidates: list[dict[str, Any]] = []
    candidate_counts: Counter[tuple[str, str, str, str | None]] = Counter()
    for prediction in predictions:
        if prediction["split"] != "test":
            continue
        pair_id = str(prediction["pair_id"])
        pair = pair_by_id[pair_id]
        diagnostics_for_pair = pair_diagnostics[pair_id]
        visibility_stratum = assign_tertile(diagnostics_for_pair["pair_visibility"], visibility_cutpoints)
        motion_stratum = assign_tertile(diagnostics_for_pair["pair_motion_speed"], motion_cutpoints)
        for configuration in CONFIG_ORDER:
            threshold = thresholds[configuration]
            score = float(prediction["scores"][configuration])
            error_type = classify_error(int(prediction["label"]), score, threshold)
            if error_type is None:
                continue
            row = {
                **prediction,
                **diagnostics_for_pair,
                "configuration": configuration,
                "error_type": error_type,
                "threshold": threshold,
                "threshold_margin": abs(score - threshold),
                "visibility_stratum": visibility_stratum,
                "motion_stratum": motion_stratum,
            }
            row.pop("model_rows")
            candidates.append(row)
            candidate_counts[(configuration, error_type, str(row["temporal_gap_bin"]), row["negative_difficulty"])] += 1

    selected = select_candidates(candidates)
    for selection_index, row in enumerate(selected, start=1):
        pair = pair_by_id[str(row["pair_id"])]
        model_rows = pair_diagnostics[str(row["pair_id"])]["model_rows"]
        diagnosis_category, diagnosis = diagnose(row, depth_bias_cutpoint, low_fill_cutpoint)
        row.update(
            {
                "selection_index": selection_index,
                "diagnosis_category": diagnosis_category,
                "diagnosis": diagnosis,
                "thresholds": thresholds,
                "attribute_relation": pair["attribute_relation"],
                "controls": pair["controls"],
                "frame_index_a": int(pair["frame_index_a"]),
                "frame_index_b": int(pair["frame_index_b"]),
                "instance_index_a": int(pair["instance_index_a"]),
                "instance_index_b": int(pair["instance_index_b"]),
                "observation_id_a": str(pair["observation_id_a"]),
                "observation_id_b": str(pair["observation_id_b"]),
            }
        )
        for side, model_row in zip(("a", "b"), model_rows):
            source = args.phase1_dir / str(model_row["rgb_crop_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = assets_dir / f"{selection_index:02d}_{row['pair_id']}_{side}.png"
            shutil.copyfile(source, destination)
            row[f"source_crop_path_{side}"] = str(source.resolve())
            row[f"gallery_crop_path_{side}"] = str(destination.relative_to(args.output_dir))

    if len(selected) != 24 or len({row["pair_id"] for row in selected}) != 24:
        raise ValueError("Phase 5 must select 24 unique pairs")

    selected_path = args.output_dir / "selected_error_pairs.jsonl"
    review_path = args.output_dir / "phase5_error_review.csv"
    gallery_path = args.output_dir / "phase5_error_gallery.html"
    write_jsonl(selected_path, selected)
    review_fields = [
        "selection_index", "pair_id", "configuration", "error_type", "label", "video_id",
        "temporal_gap", "temporal_gap_bin", "negative_difficulty", "visibility_stratum",
        "motion_stratum", "threshold", "threshold_margin", "diagnosis_category", "diagnosis",
        "frame_index_a", "frame_index_b", "instance_index_a", "instance_index_b",
        "gallery_crop_path_a", "gallery_crop_path_b",
        *[f"score_{configuration}" for configuration in CONFIG_ORDER],
    ]
    review_rows = []
    for row in selected:
        flat = {field: row.get(field) for field in review_fields}
        for configuration in CONFIG_ORDER:
            flat[f"score_{configuration}"] = row["scores"][configuration]
        review_rows.append(flat)
    write_csv(review_path, review_rows, review_fields)
    fragment = build_fragment(selected)
    gallery_path.write_text(standalone_html(fragment), encoding="utf-8")
    if args.inline_fragment:
        args.inline_fragment.parent.mkdir(parents=True, exist_ok=True)
        args.inline_fragment.write_text(fragment, encoding="utf-8")
        if args.inline_fragment.stat().st_size >= 2_000_000:
            raise ValueError("Inline visualization exceeds 2 MB")

    method_counts = Counter(row["configuration"] for row in selected)
    error_counts = Counter(row["error_type"] for row in selected)
    gap_counts = Counter(row["temporal_gap_bin"] for row in selected)
    difficulty_counts = Counter(row["negative_difficulty"] for row in selected if row["error_type"] == "false_positive")
    visibility_counts = Counter(row["visibility_stratum"] for row in selected)
    motion_counts = Counter(row["motion_stratum"] for row in selected)
    video_counts = Counter(str(row["video_id"]) for row in selected)
    if method_counts != Counter({configuration: 6 for configuration in CONFIG_ORDER}):
        raise ValueError(f"Method balance failed: {method_counts}")
    if error_counts != Counter({"false_positive": 12, "false_negative": 12}):
        raise ValueError(f"Error balance failed: {error_counts}")
    if gap_counts != Counter({gap: 8 for gap in GAP_ORDER}):
        raise ValueError(f"Gap balance failed: {gap_counts}")
    if difficulty_counts != Counter({"hard": 6, "easy": 6}):
        raise ValueError(f"Negative difficulty balance failed: {difficulty_counts}")

    manifest_path = args.output_dir / "phase5_selection_manifest.json"
    asset_hashes = {str(path.relative_to(args.output_dir)): sha256(path) for path in sorted(assets_dir.glob("*.png"))}
    manifest = {
        "pipeline": "MOVi-A Phase 5 balanced qualitative error review",
        "version": VERSION,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": Image.__version__, "platform": platform.platform()},
        "protocol": {
            "operating_threshold": "Phase 3 dev-selected maximum-F1 threshold, applied unchanged to test",
            "selection_count": 24,
            "uniqueness": "one selected item per pair ID; a pair is never assigned to two methods",
            "exact_slots": "one item per method x error type x temporal-gap bin",
            "false_positive_difficulty": "alternating slot assignment produces exactly 6 hard and 6 easy negatives",
            "secondary_balance": "deterministic greedy minimization of visibility-stratum, motion-stratum, and video reuse; ties favor larger error margin then pair ID",
            "diagnosis_status": "rule-based one-line hypotheses for qualitative review, not adjudicated causal labels",
            "diagnostic_leakage_guard": "ground-truth velocity and reconstruction errors are used only after predictions are locked and only for selection/diagnosis",
        },
        "thresholds": thresholds,
        "strata": {
            "visibility_cutpoints_from_phase4_training_pairs": visibility_cutpoints,
            "motion_cutpoints_from_phase4_training_pairs": motion_cutpoints,
            "depth_bias_cutpoint_training_pair_p80": depth_bias_cutpoint,
            "low_mask_fill_cutpoint_training_pair_p20": low_fill_cutpoint,
        },
        "inputs": {
            "predictions_sha256": sha256(args.predictions),
            "locked_config_sha256": sha256(args.locked_config),
            "phase4_results_sha256": sha256(args.phase4_results),
            "pairs_sha256": sha256(args.pairs),
            "model_inputs_sha256": sha256(args.model_inputs),
            "diagnostics_sha256": sha256(args.diagnostics),
        },
        "candidate_counts": {"|".join(str(value) for value in key): count for key, count in sorted(candidate_counts.items(), key=lambda item: str(item[0]))},
        "selected_balance": {
            "methods": dict(method_counts),
            "error_types": dict(error_counts),
            "temporal_gap_bins": dict(gap_counts),
            "false_positive_difficulty": dict(difficulty_counts),
            "visibility_strata": dict(visibility_counts),
            "motion_strata": dict(motion_counts),
            "videos": dict(video_counts),
            "diagnoses": dict(Counter(row["diagnosis_category"] for row in selected)),
        },
        "checks": {
            "selected_pairs": len(selected),
            "unique_pair_ids": len({row["pair_id"] for row in selected}),
            "all_from_test": all(row["split"] == "test" for row in selected),
            "all_misclassified_at_assigned_locked_threshold": all(
                classify_error(int(row["label"]), float(row["scores"][row["configuration"]]), float(row["threshold"])) == row["error_type"]
                for row in selected
            ),
            "all_gallery_assets_exist": len(asset_hashes) == 48,
        },
        "outputs": {
            selected_path.name: sha256(selected_path),
            review_path.name: sha256(review_path),
            gallery_path.name: sha256(gallery_path),
            "assets": asset_hashes,
        },
    }
    write_json(manifest_path, manifest)
    print(f"Complete: selected {len(selected)} unique Phase 5 errors in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
