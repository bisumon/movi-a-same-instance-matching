#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests/movi_de/final_release_manifest.json"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def lines(path: Path) -> int:
    with path.open("rb") as f: return sum(1 for line in f if line.strip())

files=set()
for pattern in [
    ".gitignore","README.md","FINAL_DELIVERABLES_MOVI_DE.md","requirements*.txt","LICENSE.md",
    "run_movi_de_experiment.py","run_movi_de_experiments.sh","configs/movi_de*",
    "docs/MOVI_DE*","docs/SOURCES_AND_LICENSES.md","manifests/movi_de/*",
    "manifests/pairs/movi_de/*","predictions/movi_de/*","results/movi_de*/*",
    "failure_gallery/movi_de_phase10/**/*","src/*movi_de*","tests/test_movi*de*",
    "third_party_licenses/**/*",
]:
    for p in ROOT.glob(pattern):
        if p.is_file() and p != OUT and not p.name.startswith("~$") and not p.name.endswith(".inspect.ndjson"): files.add(p)

video_counts={}
for dataset in ("movi_d","movi_e"):
    video_counts[dataset]={
        "selected":lines(ROOT/f"manifests/movi_de/confirmatory_{dataset}_150.jsonl"),
        **{split:lines(ROOT/f"manifests/movi_de/confirmatory_{dataset}_{split}_{count}.jsonl") for split,count in (("train",90),("dev",30),("test",30))},
    }
pair_counts={}
for dataset in ("movi_d","movi_e"):
    pair_counts[dataset]={split:lines(ROOT/f"manifests/pairs/movi_de/{dataset}_{split}_pairs.jsonl") for split in ("train","dev","test")}
    pair_counts[dataset]["all"]=lines(ROOT/f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl")
prediction_counts={p.name:lines(p) for p in sorted((ROOT/"predictions/movi_de").glob("*.jsonl"))}
gallery_count=lines(ROOT/"failure_gallery/movi_de_phase10/selected_failure_pairs.jsonl")
checks={
    "video_counts_are_150_and_90_30_30":all(v=={"selected":150,"train":90,"dev":30,"test":30} for v in video_counts.values()),
    "pair_counts_are_10000_and_6000_2000_2000":all(v=={"train":6000,"dev":2000,"test":2000,"all":10000} for v in pair_counts.values()),
    "in_domain_and_noise_predictions_have_10000_rows":all(n==10000 for k,n in prediction_counts.items() if "regime3" not in k),
    "transfer_predictions_have_2000_rows":prediction_counts.get("movi_d_to_e_phase8_regime3_predictions.jsonl")==2000,
    "failure_gallery_has_at_least_20_items":gallery_count>=20,
    "one_command_runner_present_and_executable":(ROOT/"run_movi_de_experiments.sh").exists() and bool((ROOT/"run_movi_de_experiments.sh").stat().st_mode & 0o111),
    "results_workbook_present":(ROOT/"docs/MOVI_DE_RESULTS_TABLES.xlsx").exists(),
    "conclusions_docx_present":(ROOT/"docs/MOVI_DE_CONCLUSIONS_AND_NEXT_STEPS.docx").exists(),
    "sources_and_licenses_present":(ROOT/"docs/SOURCES_AND_LICENSES.md").exists() and (ROOT/"third_party_licenses").is_dir(),
}
payload={
    "release":"movi_de_camera_pose_extension_v1.0.0","created":"2026-08-25","seed":20260825,
    "scope":"Final publishable MOVi-D/E code, locks, predictions, results, failure analysis, conclusions, and license documentation",
    "counts":{"videos":video_counts,"pairs":pair_counts,"predictions":prediction_counts,"failure_gallery_items":gallery_count,"hashed_files":len(files)},
    "checks":checks,
    "files":{str(p.relative_to(ROOT)): {"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(files)},
    "authoritative_stage_locks":[
        "manifests/movi_de/phase5_pair_manifest_freeze.json","manifests/movi_de/phase6_system_configuration_freeze.json",
        "manifests/movi_de/phase7_pose_noise_study_freeze.json","manifests/movi_de/phase8_regime1_in_domain_movi_e_freeze.json",
        "manifests/movi_de/phase8_regime2_in_domain_movi_d_freeze.json","manifests/movi_de/phase8_regime3_d_to_e_transfer_freeze.json",
        "manifests/movi_de/phase9_criteria_evaluation_manifest.json","manifests/movi_de/phase10_failure_analysis_freeze.json",
    ],
}
if not all(checks.values()): raise RuntimeError(json.dumps(checks,indent=2))
OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"output":str(OUT),"checks":checks,"hashed_files":len(files)},indent=2))
