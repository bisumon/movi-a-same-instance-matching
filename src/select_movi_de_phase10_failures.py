#!/usr/bin/env python3
"""Select and render the frozen MOVi-D/E Phase 10 failure analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw


VERSION = "1.0.0"
SYSTEMS = ("C_camera_geometry", "D_pose_aligned_geometry")
DIAGNOSIS_PRIORITY = {
    "matched_asset_distractor": 0,
    "camera_motion_misalignment": 1,
    "pose_transform_error": 2,
    "occlusion_truncated_mask": 3,
    "biased_depth_estimate": 4,
    "long_gap_motion": 5,
    "matched_category_distractor": 6,
    "ambiguous_appearance": 7,
}


@dataclass(frozen=True)
class Slot:
    dataset: str
    system: str
    error_type: str
    dynamic_group: str
    motion_stratum: str
    negative_difficulty: str | None = None


def slots() -> list[Slot]:
    return [
        # MOVi-E false positives: the two medium-motion slots are capacity-constrained.
        Slot("movi_e", SYSTEMS[0], "false_positive", "static", "low", "easy"),
        Slot("movi_e", SYSTEMS[0], "false_positive", "dynamic", "medium", "easy"),
        Slot("movi_e", SYSTEMS[0], "false_positive", "static", "high", "hard"),
        Slot("movi_e", SYSTEMS[1], "false_positive", "dynamic", "medium", "easy"),
        Slot("movi_e", SYSTEMS[1], "false_positive", "dynamic", "low", "hard"),
        Slot("movi_e", SYSTEMS[1], "false_positive", "static", "high", "hard"),
        Slot("movi_e", SYSTEMS[0], "false_negative", "dynamic", "low"),
        Slot("movi_e", SYSTEMS[0], "false_negative", "dynamic", "high"),
        Slot("movi_e", SYSTEMS[0], "false_negative", "static", "high"),
        Slot("movi_e", SYSTEMS[1], "false_negative", "static", "low"),
        Slot("movi_e", SYSTEMS[1], "false_negative", "dynamic", "low"),
        Slot("movi_e", SYSTEMS[1], "false_negative", "static", "high"),
        # MOVi-D is structurally zero-motion; balance static/dynamic and hard/easy instead.
        Slot("movi_d", SYSTEMS[0], "false_positive", "dynamic", "zero", "easy"),
        Slot("movi_d", SYSTEMS[0], "false_positive", "dynamic", "zero", "hard"),
        Slot("movi_d", SYSTEMS[0], "false_positive", "static", "zero", "hard"),
        Slot("movi_d", SYSTEMS[1], "false_positive", "dynamic", "zero", "easy"),
        Slot("movi_d", SYSTEMS[1], "false_positive", "static", "zero", "easy"),
        Slot("movi_d", SYSTEMS[1], "false_positive", "static", "zero", "hard"),
        Slot("movi_d", SYSTEMS[0], "false_negative", "dynamic", "zero"),
        Slot("movi_d", SYSTEMS[0], "false_negative", "dynamic", "zero"),
        Slot("movi_d", SYSTEMS[0], "false_negative", "static", "zero"),
        Slot("movi_d", SYSTEMS[1], "false_negative", "dynamic", "zero"),
        Slot("movi_d", SYSTEMS[1], "false_negative", "static", "zero"),
        Slot("movi_d", SYSTEMS[1], "false_negative", "static", "zero"),
    ]


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


def classify_error(label: int, score: float, threshold: float) -> str | None:
    predicted = int(score >= threshold)
    if label == 0 and predicted == 1:
        return "false_positive"
    if label == 1 and predicted == 0:
        return "false_negative"
    return None


def assign_tertile(value: float, cutpoints: Sequence[float]) -> str:
    return "low" if value <= cutpoints[0] else "medium" if value <= cutpoints[1] else "high"


def dynamic_group(pair: dict[str, Any]) -> str:
    return "static" if pair["controls"]["object_dynamic_static_status"] == "static-static" else "dynamic"


def edge_truncated(row: dict[str, Any]) -> bool:
    x0, y0, x1, y1 = map(int, row["tight_bbox_xyxy"])
    intrinsics = row["intrinsics"]
    return x0 <= 0 or y0 <= 0 or x1 >= int(intrinsics["image_width"]) or y1 >= int(intrinsics["image_height"])


def pair_diagnostics(pair: dict[str, Any], observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = observations[str(pair["observation_id_a"])]
    second = observations[str(pair["observation_id_b"])]
    camera_first = np.asarray(first["camera_space_visible_surface_centroid_xyz"], dtype=np.float64)
    camera_second = np.asarray(second["camera_space_visible_surface_centroid_xyz"], dtype=np.float64)
    world_first = np.asarray(first["pose_aligned_world_visible_surface_centroid_xyz"], dtype=np.float64)
    world_second = np.asarray(second["pose_aligned_world_visible_surface_centroid_xyz"], dtype=np.float64)
    return {
        "observation_rows": (first, second),
        "edge_truncated": edge_truncated(first) or edge_truncated(second),
        "minimum_mask_fill_fraction": min(float(first["mask_fill_fraction"]), float(second["mask_fill_fraction"])),
        "absolute_depth_median_difference": abs(float(first["depth"]["median"]) - float(second["depth"]["median"])),
        "camera_centroid_distance": float(np.linalg.norm(camera_first - camera_second)),
        "world_centroid_distance": float(np.linalg.norm(world_first - world_second)),
    }


def diagnosis(row: dict[str, Any], cutpoints: dict[str, float]) -> tuple[str, str]:
    other = SYSTEMS[1] if row["system"] == SYSTEMS[0] else SYSTEMS[0]
    other_error = classify_error(int(row["label"]), float(row["scores"][other]), float(row["thresholds"][other]))
    low_or_truncated = row["visibility_stratum"] == "low" and (
        row["edge_truncated"] or row["minimum_mask_fill_fraction"] <= cutpoints["mask_fill_p20"]
    )
    depth_unstable = row["absolute_depth_median_difference"] >= cutpoints["positive_depth_difference_p80"]
    world_unstable = row["world_centroid_distance"] >= cutpoints["positive_world_distance_p80"]
    if row["error_type"] == "false_positive":
        if bool(row["controls"].get("very_hard_same_asset")) or row["controls"].get("asset_relation") == "same_asset":
            return "matched_asset_distractor", "Same-asset distractor remains nearly indistinguishable at the locked operating threshold."
        if row["negative_difficulty"] == "hard":
            return "matched_category_distractor", "Matched-category, similar-scale distractor remains compatible in appearance and geometry."
        if low_or_truncated:
            return "occlusion_truncated_mask", "Low support or an image-edge-truncated mask hides cues that separate these objects."
        return "ambiguous_appearance", "Different-category objects still produce an unusually match-like appearance and geometry score."
    if low_or_truncated:
        return "occlusion_truncated_mask", "Occlusion or image-edge truncation removes reliable same-instance surface evidence."
    if row["dataset"] == "movi_e" and row["motion_stratum"] == "high" and row["system"] == SYSTEMS[0] and other_error is None:
        return "camera_motion_misalignment", "Camera-space geometry drifts under high camera motion while pose-aligned D remains correct."
    if row["dataset"] == "movi_e" and row["motion_stratum"] == "high" and row["system"] == SYSTEMS[1] and other_error is None:
        return "pose_transform_error", "Possible pose-transform or visible-surface instability makes D miss a pair that C retains."
    if depth_unstable or world_unstable:
        return "biased_depth_estimate", "Visible-surface depth or centroid changes unusually strongly across the two observations."
    if row["temporal_gap_bin"].startswith("long") or row["dynamic_group"] == "dynamic":
        return "long_gap_motion", "Object motion or a long temporal gap shifts appearance and visible geometry below threshold."
    return "ambiguous_appearance", "The same instance changes enough in viewpoint or appearance to cross the fixed threshold."


def matches(row: dict[str, Any], slot: Slot) -> bool:
    return (
        row["dataset"] == slot.dataset
        and row["system"] == slot.system
        and row["error_type"] == slot.error_type
        and row["dynamic_group"] == slot.dynamic_group
        and row["motion_stratum"] == slot.motion_stratum
        and (slot.negative_difficulty is None or row["negative_difficulty"] == slot.negative_difficulty)
    )


def select(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    remaining = list(enumerate(slots()))
    selected: list[dict[str, Any]] = []
    used_pairs: set[str] = set()
    gap_counts: Counter[str] = Counter()
    diagnosis_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    initial_capacity = {str(index + 1): sum(matches(row, slot) for row in candidates) for index, slot in remaining}
    while remaining:
        capacity = []
        for original_index, slot in remaining:
            eligible = [row for row in candidates if row["pair_id"] not in used_pairs and matches(row, slot)]
            capacity.append((len(eligible), original_index, slot, eligible))
        count, original_index, slot, eligible = min(capacity, key=lambda value: (value[0], value[1]))
        if count == 0:
            raise ValueError(f"No unique candidate remains for slot {original_index + 1}: {slot}")
        chosen = min(
            eligible,
            key=lambda row: (
                diagnosis_counts[row["diagnosis_category"]],
                DIAGNOSIS_PRIORITY[row["diagnosis_category"]],
                gap_counts[row["temporal_gap_bin"]],
                video_counts[f"{row['dataset']}:{row['video_id']}"],
                -float(row["threshold_margin"]),
                str(row["pair_id"]),
            ),
        ).copy()
        chosen["selection_slot"] = {"slot_number": original_index + 1, **asdict(slot)}
        selected.append(chosen)
        used_pairs.add(str(chosen["pair_id"]))
        gap_counts[chosen["temporal_gap_bin"]] += 1
        diagnosis_counts[chosen["diagnosis_category"]] += 1
        video_counts[f"{chosen['dataset']}:{chosen['video_id']}"] += 1
        remaining = [(index, value) for index, value in remaining if index != original_index]
    selected.sort(key=lambda row: int(row["selection_slot"]["slot_number"]))
    return selected, initial_capacity


def score_rows(row: dict[str, Any]) -> str:
    output = []
    for system in SYSTEMS:
        score = float(row["scores"][system])
        threshold = float(row["thresholds"][system])
        prediction = "match" if score >= threshold else "non-match"
        correct = int(score >= threshold) == int(row["label"])
        selected = " · reviewed" if system == row["system"] else ""
        output.append(
            f"<tr><td><code>{escape(system)}</code>{selected}</td><td>{score:.4f}</td>"
            f"<td>{threshold:.4f}</td><td class=\"{'ok' if correct else 'bad'}\">{prediction} · {'correct' if correct else 'error'}</td></tr>"
        )
    return "".join(output)


def gallery_html(rows: Sequence[dict[str, Any]], balance: dict[str, Any]) -> str:
    cards = []
    for index, row in enumerate(rows, start=1):
        controls = row["controls"]
        truth = "match" if row["label"] else "non-match"
        cards.append(f"""
<article class="card" data-dataset="{row['dataset']}" data-system="{row['system']}" data-error="{row['error_type']}" data-motion="{row['motion_stratum']}">
  <div class="badges"><strong>#{index:02d}</strong><span>{row['dataset'].upper()}</span><span>{escape(row['system'].split('_')[0])}</span><span>{escape(row['error_type'].replace('_',' '))}</span><span>{escape(row['motion_stratum'])} motion</span></div>
  <h2>{escape(row['diagnosis_category'].replace('_',' ').title())}</h2>
  <p class="diagnosis">{escape(row['diagnosis'])}</p>
  <div class="content">
    <div class="crops">
      <figure><img src="{escape(row['gallery_crop_path_a'])}" alt="First crop for {row['pair_id']}"><figcaption>Frame {row['frame_index_a']} · instance {row['instance_index_a']}</figcaption></figure>
      <figure><img src="{escape(row['gallery_crop_path_b'])}" alt="Second crop for {row['pair_id']}"><figcaption>Frame {row['frame_index_b']} · instance {row['instance_index_b']}</figcaption></figure>
    </div>
    <div class="tablewrap"><table><thead><tr><th>System</th><th>Score</th><th>Threshold</th><th>Decision</th></tr></thead><tbody>{score_rows(row)}</tbody></table></div>
  </div>
  <div class="controls">
    <span>Truth: <strong>{truth}</strong></span><span>Pair: <code>{row['pair_id']}</code></span><span>Video: {row['video_id']}</span>
    <span>Gap: {row['temporal_gap']} ({row['temporal_gap_bin'].split('_')[0]})</span><span>Difficulty: {row['negative_difficulty'] or 'positive'}</span>
    <span>Dynamics: {escape(controls['object_dynamic_static_status'])}</span><span>Visibility: {controls['visibility_a']}/{controls['visibility_b']}</span>
    <span>Mask area: {controls['mask_area_a']}/{controls['mask_area_b']} px</span><span>Depth pixels: {controls['valid_depth_pixels_a']}/{controls['valid_depth_pixels_b']}</span>
    <span>Camera translation: {controls['camera_displacement_scene_units']:.3f}</span><span>Rotation: {controls['relative_camera_rotation_degrees']:.3f}°</span>
    <span>Normalized translation: {controls['normalized_camera_displacement']:.4f}</span>
  </div>
</article>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-D/E Phase 10 failure gallery</title><style>
:root{{--ink:#17212b;--muted:#52606d;--line:#d9e2ec;--soft:#f3f7fa;--accent:#245b78;--bad:#a23b32;--ok:#27724b;font-family:Inter,system-ui,-apple-system,sans-serif}}*{{box-sizing:border-box}}body{{max-width:1200px;margin:auto;padding:28px;color:var(--ink);background:#fff}}header{{border-bottom:3px solid var(--accent);padding-bottom:18px}}h1{{margin:.1rem 0 .5rem;font-size:2rem}}header p{{max-width:900px;color:var(--muted)}}.summary{{display:flex;gap:.5rem 1.2rem;flex-wrap:wrap;margin-top:1rem}}.filters{{position:sticky;top:0;background:#fff;padding:12px 0;border-bottom:1px solid var(--line);display:flex;gap:.75rem;align-items:end;z-index:2}}label{{display:grid;gap:.2rem;font-size:.85rem;color:var(--muted)}}select{{padding:.45rem;border:1px solid var(--line);border-radius:5px;background:#fff}}.card{{padding:24px 0;border-bottom:1px solid var(--line)}}.badges{{display:flex;gap:.4rem;align-items:center;flex-wrap:wrap}}.badges span{{background:#e6f0f5;color:#17475f;border-radius:999px;padding:.2rem .55rem;font-size:.82rem}}h2{{font-size:1.15rem;margin:.7rem 0 .25rem}}.diagnosis{{margin:.2rem 0 1rem}}.content{{display:grid;grid-template-columns:minmax(300px,.9fr) minmax(420px,1.1fr);gap:22px;align-items:start}}.crops{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0}}img{{width:100%;aspect-ratio:1;object-fit:contain;background:var(--soft);border:1px solid var(--line);image-rendering:auto}}figcaption{{font-size:.82rem;color:var(--muted);margin-top:.25rem}}table{{border-collapse:collapse;width:100%;font-size:.88rem}}th,td{{padding:.48rem;border-bottom:1px solid var(--line);text-align:left}}th{{background:var(--soft)}}.bad{{color:var(--bad);font-weight:700}}.ok{{color:var(--ok)}}.controls{{display:flex;gap:.35rem 1rem;flex-wrap:wrap;margin-top:12px;font-size:.82rem;color:var(--muted)}}.note{{background:var(--soft);border-left:4px solid var(--accent);padding:12px 16px;margin-top:14px}}[hidden]{{display:none!important}}@media(max-width:780px){{body{{padding:16px}}.content{{grid-template-columns:1fr}}.filters{{position:static;flex-wrap:wrap}}}}
</style></head><body><header><h1>Phase 10 · MOVi-D/E failure analysis</h1><p>Twenty-four unique locked-test errors at each system’s development-selected 90%-recall threshold. Selection is balanced across datasets, systems, false positives/negatives, dynamics, negative difficulty, and camera motion where capacity permits.</p><div class="summary"><strong>24 unique pairs</strong><span>12 MOVi-D / 12 MOVi-E</span><span>12 C / 12 D</span><span>12 FP / 12 FN</span><span>12 static / 12 dynamic</span></div><p class="note"><strong>Interpret diagnoses as review hypotheses.</strong> They are deterministic rule-based labels assigned after metrics were locked, not adjudicated causal ground truth. Gallery selection cannot change any reported result.</p></header>
<div class="filters"><label>Dataset<select id="dataset"><option value="all">Both</option><option value="movi_d">MOVi-D</option><option value="movi_e">MOVi-E</option></select></label><label>System<select id="system"><option value="all">C and D</option><option value="C_camera_geometry">C</option><option value="D_pose_aligned_geometry">D</option></select></label><label>Error<select id="error"><option value="all">FP and FN</option><option value="false_positive">False positive</option><option value="false_negative">False negative</option></select></label><label>Motion<select id="motion"><option value="all">All</option><option value="zero">Zero</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label><strong id="count">Showing 24 of 24</strong></div>
<main>{''.join(cards)}</main><script>(()=>{{const cards=[...document.querySelectorAll('.card')];const controls=['dataset','system','error','motion'].map(x=>document.getElementById(x));function update(){{let n=0;cards.forEach(c=>{{const show=controls.every(x=>x.value==='all'||c.dataset[x.id]===x.value);c.hidden=!show;if(show)n++}});document.getElementById('count').textContent=`Showing ${{n}} of ${{cards.length}}`;}}controls.forEach(x=>x.addEventListener('change',update));update();}})();</script></body></html>"""


def contact_sheet(rows: Sequence[dict[str, Any]], output_dir: Path, destination: Path) -> None:
    columns, cell_width, cell_height = 4, 220, 132
    sheet = Image.new("RGB", (columns * cell_width, 6 * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for number, row in enumerate(rows):
        x = (number % columns) * cell_width
        y = (number // columns) * cell_height
        for side, offset in (("a", 8), ("b", 108)):
            with Image.open(output_dir / row[f"gallery_crop_path_{side}"]) as crop:
                sheet.paste(crop.convert("RGB").resize((96, 96)), (x + offset, y + 26))
        system = "C" if row["system"] == SYSTEMS[0] else "D"
        error = "FP" if row["error_type"] == "false_positive" else "FN"
        label = f"#{number + 1:02d} {row['dataset'].upper()} {system} {error} {row['motion_stratum']}"
        draw.text((x + 8, y + 7), label, fill="#17212b")
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#d9e2ec")
    sheet.save(destination, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir()

    candidates: list[dict[str, Any]] = []
    input_paths: dict[str, Path] = {}
    dataset_audits: dict[str, Any] = {}
    for dataset, regime in (("movi_d", "regime2"), ("movi_e", "regime1")):
        prefix = "movi_d_phase8_regime2" if dataset == "movi_d" else "movi_e_phase8_regime1"
        pair_path = root / f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl"
        prediction_path = root / f"runs/movi_de_confirmatory/phase8_{regime}/in_domain_{dataset}/{prefix}_predictions.jsonl"
        lock_path = root / f"runs/movi_de_confirmatory/phase8_{regime}/in_domain_{dataset}/{prefix}_locked_config.json"
        model_path = root / f"runs/movi_de_confirmatory/phase7/{dataset}_phase1/model_inputs.jsonl"
        phase1_dir = model_path.parent
        input_paths.update({f"{dataset}_pairs": pair_path, f"{dataset}_predictions": prediction_path, f"{dataset}_lock": lock_path, f"{dataset}_model_inputs": model_path})
        pairs, predictions = read_jsonl(pair_path), read_jsonl(prediction_path)
        if [row["pair_id"] for row in pairs] != [row["pair_id"] for row in predictions]:
            raise ValueError(f"{dataset} pair/prediction order mismatch")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        thresholds = {system: float(lock["systems"][system]["recall_90_threshold"]) for system in SYSTEMS}
        observations = {row["observation_id"]: row for row in read_jsonl(model_path)}
        training = [row for row in pairs if row["split"] == "train"]
        motion_cuts = np.quantile([row["controls"]["camera_displacement_scene_units"] for row in training], [1 / 3, 2 / 3]) if dataset == "movi_e" else np.asarray([0.0, 0.0])
        visibility_cuts = np.quantile([row["controls"]["minimum_visibility"] for row in training], [1 / 3, 2 / 3])
        training_diagnostics = [pair_diagnostics(row, observations) for row in training if row["label"] == 1]
        diagnostic_cuts = {
            "mask_fill_p20": float(np.quantile([value["minimum_mask_fill_fraction"] for value in training_diagnostics], 0.20)),
            "positive_depth_difference_p80": float(np.quantile([value["absolute_depth_median_difference"] for value in training_diagnostics], 0.80)),
            "positive_world_distance_p80": float(np.quantile([value["world_centroid_distance"] for value in training_diagnostics], 0.80)),
        }
        errors: Counter[str] = Counter()
        for pair, prediction in zip(pairs, predictions, strict=True):
            if pair["split"] != "test":
                continue
            diagnostics = pair_diagnostics(pair, observations)
            motion = "zero" if dataset == "movi_d" else assign_tertile(float(pair["controls"]["camera_displacement_scene_units"]), motion_cuts)
            visibility = assign_tertile(float(pair["controls"]["minimum_visibility"]), visibility_cuts)
            for system in SYSTEMS:
                error = classify_error(int(pair["label"]), float(prediction["scores"][system]), thresholds[system])
                if error is None:
                    continue
                row = {
                    **pair, **diagnostics, "dataset": dataset, "system": system, "error_type": error,
                    "dynamic_group": dynamic_group(pair), "motion_stratum": motion, "visibility_stratum": visibility,
                    "scores": {name: float(prediction["scores"][name]) for name in SYSTEMS}, "thresholds": thresholds,
                    "threshold_margin": abs(float(prediction["scores"][system]) - thresholds[system]),
                    "phase1_dir": str(phase1_dir.resolve()), "diagnostic_cutpoints": diagnostic_cuts,
                }
                category, text = diagnosis(row, diagnostic_cuts)
                row["diagnosis_category"], row["diagnosis"] = category, text
                candidates.append(row)
                errors[f"{system}|{error}"] += 1
        dataset_audits[dataset] = {
            "thresholds": thresholds,
            "training_motion_tertile_cutpoints": [float(value) for value in motion_cuts],
            "training_visibility_tertile_cutpoints": [float(value) for value in visibility_cuts],
            "diagnostic_cutpoints": diagnostic_cuts,
            "same_asset_test_negative_pairs": sum(
                row["split"] == "test" and row["label"] == 0 and bool(row["controls"].get("very_hard_same_asset"))
                for row in pairs
            ),
            "test_error_capacity": dict(errors),
        }

    selected, slot_capacity = select(candidates)
    for index, row in enumerate(selected, start=1):
        first, second = row.pop("observation_rows")
        row.pop("phase1_dir")
        row["selection_index"] = index
        for side, observation in (("a", first), ("b", second)):
            source = root / f"runs/movi_de_confirmatory/phase7/{row['dataset']}_phase1" / observation["rgb_crop_path"]
            destination = assets / f"{index:02d}_{row['dataset']}_{row['pair_id']}_{side}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, destination)
            row[f"gallery_crop_path_{side}"] = str(destination.relative_to(args.output_dir))

    balance = {
        "datasets": dict(Counter(row["dataset"] for row in selected)),
        "systems": dict(Counter(row["system"] for row in selected)),
        "dataset_system_error": dict(Counter(f"{row['dataset']}|{row['system']}|{row['error_type']}" for row in selected)),
        "error_types": dict(Counter(row["error_type"] for row in selected)),
        "dynamic_groups": dict(Counter(row["dynamic_group"] for row in selected)),
        "MOVi_E_motion": dict(Counter(row["motion_stratum"] for row in selected if row["dataset"] == "movi_e")),
        "false_positive_difficulty": dict(Counter(row["negative_difficulty"] for row in selected if row["error_type"] == "false_positive")),
        "temporal_gap_bins": dict(Counter(row["temporal_gap_bin"] for row in selected)),
        "diagnoses": dict(Counter(row["diagnosis_category"] for row in selected)),
        "videos": dict(Counter(f"{row['dataset']}:{row['video_id']}" for row in selected)),
    }
    expected_cells = {f"{dataset}|{system}|{error}": 3 for dataset in ("movi_d", "movi_e") for system in SYSTEMS for error in ("false_positive", "false_negative")}
    checks = {
        "exact_24_unique_pairs": len(selected) == 24 and len({row["pair_id"] for row in selected}) == 24,
        "all_locked_test_errors": all(row["split"] == "test" and classify_error(row["label"], row["scores"][row["system"]], row["thresholds"][row["system"]]) == row["error_type"] for row in selected),
        "exact_dataset_balance": balance["datasets"] == {"movi_e": 12, "movi_d": 12},
        "exact_system_balance": balance["systems"] == {SYSTEMS[0]: 12, SYSTEMS[1]: 12},
        "exact_error_balance": balance["error_types"] == {"false_positive": 12, "false_negative": 12},
        "exact_three_per_dataset_system_error_cell": balance["dataset_system_error"] == expected_cells,
        "exact_static_dynamic_balance": balance["dynamic_groups"] == {"static": 12, "dynamic": 12},
        "exact_hard_easy_false_positive_balance": balance["false_positive_difficulty"] == {"easy": 6, "hard": 6},
        "MOVi_E_low_high_near_balance_with_two_capacity_required_medium": balance["MOVi_E_motion"] == {"low": 5, "medium": 2, "high": 5},
        "all_48_crop_assets_exist": len(list(assets.glob("*.png"))) == 48,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 10 checks failed: {[name for name, passed in checks.items() if not passed]}")

    selected_path = args.output_dir / "selected_failure_pairs.jsonl"
    csv_path = args.output_dir / "failure_review.csv"
    gallery_path = args.output_dir / "failure_gallery.html"
    preview_path = args.output_dir / "contact_sheet.png"
    capacity_path = args.output_dir / "capacity_audit.json"
    write_jsonl(selected_path, selected)
    fields = ["selection_index", "dataset", "pair_id", "video_id", "system", "error_type", "label", "negative_difficulty", "dynamic_group", "motion_stratum", "visibility_stratum", "temporal_gap", "temporal_gap_bin", "diagnosis_category", "diagnosis", "gallery_crop_path_a", "gallery_crop_path_b", "C_score", "C_threshold", "D_score", "D_threshold", "camera_displacement", "camera_rotation_degrees", "normalized_camera_displacement"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "selection_index": row["selection_index"], "dataset": row["dataset"], "pair_id": row["pair_id"], "video_id": row["video_id"], "system": row["system"], "error_type": row["error_type"], "label": row["label"], "negative_difficulty": row["negative_difficulty"], "dynamic_group": row["dynamic_group"], "motion_stratum": row["motion_stratum"], "visibility_stratum": row["visibility_stratum"], "temporal_gap": row["temporal_gap"], "temporal_gap_bin": row["temporal_gap_bin"], "diagnosis_category": row["diagnosis_category"], "diagnosis": row["diagnosis"], "gallery_crop_path_a": row["gallery_crop_path_a"], "gallery_crop_path_b": row["gallery_crop_path_b"], "C_score": row["scores"][SYSTEMS[0]], "C_threshold": row["thresholds"][SYSTEMS[0]], "D_score": row["scores"][SYSTEMS[1]], "D_threshold": row["thresholds"][SYSTEMS[1]], "camera_displacement": row["controls"]["camera_displacement_scene_units"], "camera_rotation_degrees": row["controls"]["relative_camera_rotation_degrees"], "normalized_camera_displacement": row["controls"]["normalized_camera_displacement"],
            })
    gallery_path.write_text(gallery_html(selected, balance), encoding="utf-8")
    contact_sheet(selected, args.output_dir, preview_path)
    write_json(capacity_path, {
        "status": "pass", "required_slots": [{"slot_number": index + 1, **asdict(slot)} for index, slot in enumerate(slots())],
        "initial_candidate_capacity_per_slot": slot_capacity, "dataset_error_capacity": dataset_audits,
        "capacity_amendment": "Two MOVi-E medium-motion false-positive slots are required to retain 24 unique pairs plus exact method, error, dynamics, and hard/easy balance; MOVi-E motion selection is therefore 5 low, 2 medium, and 5 high.",
    })
    asset_hashes = {str(path.relative_to(args.output_dir)): sha256(path) for path in sorted(assets.glob("*.png"))}
    manifest_path = args.output_dir / "selection_manifest.json"
    write_json(manifest_path, {
        "pipeline": "MOVi-D/E Phase 10 balanced failure analysis", "version": VERSION, "seed": 20260825, "status": "pass",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": Image.__version__, "platform": platform.platform()},
        "protocol": {"threshold": "each system's Phase 8 development-selected 90%-recall threshold applied unchanged to locked test scores", "selection_timing": "post-confirmatory; cannot alter metrics or decisions", "diagnoses": "deterministic one-line review hypotheses, not adjudicated causes", "uniqueness": "one pair ID appears once even if both systems err"},
        "inputs": {name: {"path": str(path.relative_to(root)), "sha256": sha256(path)} for name, path in sorted(input_paths.items())},
        "balance": balance, "checks": checks,
        "outputs": {selected_path.name: sha256(selected_path), csv_path.name: sha256(csv_path), gallery_path.name: sha256(gallery_path), preview_path.name: sha256(preview_path), capacity_path.name: sha256(capacity_path), "assets": asset_hashes},
    })
    print(f"Complete: selected {len(selected)} unique Phase 10 failures in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
