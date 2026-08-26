#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/movi_de_final"; OUT.mkdir(parents=True,exist_ok=True)
SYSTEMS=("A_rgb","B_rgb_2d","C_camera_geometry","D_pose_aligned_geometry","G_camera_geometry_only","G_pose_aligned_geometry_only","P_pose_only","S_shuffled_pose")
C="C_camera_geometry"; D="D_pose_aligned_geometry"; REPS=10000

def read_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cuts(rows, getter): return np.quantile(np.asarray([getter(r) for r in rows],float),[1/3,2/3])
def tertile(v,c): return "low" if v<=c[0] else ("medium" if v<=c[1] else "high")
def weighted_auc(labels,scores,weights):
    order=np.argsort(scores,kind="stable"); y=labels[order]; s=scores[order]; w=weights[:,order]
    starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]; pg=np.add.reduceat(w*(y==1),starts,axis=1); ng=np.add.reduceat(w*(y==0),starts,axis=1)
    before=np.cumsum(ng,axis=1)-ng; den=pg.sum(1)*ng.sum(1)
    return np.divide(np.sum(pg*(before+0.5*ng),axis=1),den,out=np.full(len(w),np.nan),where=den>0)
def ci_delta(labels,cs,ds,clusters,selection,seed):
    y=labels[selection]; c=cs[selection]; d=ds[selection]; cl=clusters[selection]
    rng=np.random.default_rng(seed); n=int(clusters.max())+1; counts=rng.multinomial(n,np.full(n,1/n),size=REPS); out=[]
    if len(np.unique(y))==2:
        point=roc_auc_score(y,d)-roc_auc_score(y,c); metric="auroc"
        for start in range(0,REPS,500):
            w=counts[start:start+500,cl].astype(float); out.append(weighted_auc(y,d,w)-weighted_auc(y,c,w))
    else:
        target=int(y[0]); metric="recall" if target==1 else "false_match_rate"
        cp=(c>=thresholds[C]); dp=(d>=thresholds[D])
        point=float(dp.mean()-cp.mean());
        for start in range(0,REPS,500):
            w=counts[start:start+500,cl].astype(float); den=w.sum(1); out.append((w@(dp.astype(float)-cp.astype(float)))/den)
    samples=np.concatenate(out); samples=samples[np.isfinite(samples)]
    return metric,float(point),float(np.quantile(samples,.025)),float(np.quantile(samples,.975)),len(samples)

aggregate=[]; intervals=[]; inputs={}
for dataset,regime in (("movi_d","regime2"),("movi_e","regime1")):
    pair_path=ROOT/f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl"; pred_path=ROOT/f"predictions/movi_de/{dataset}_phase8_{regime}_predictions.jsonl"; lock_path=ROOT/f"results/movi_de_phase8_{regime}/{dataset}_in_domain_locked_config.json"
    pairs=read_jsonl(pair_path); preds=read_jsonl(pred_path); lock=json.loads(lock_path.read_text()); global thresholds; thresholds={k:float(v["recall_90_threshold"]) for k,v in lock["systems"].items()}
    assert [r["pair_id"] for r in pairs]==[r["pair_id"] for r in preds]
    train=[r for r in pairs if r["split"]=="train"]; test=[r for r in pairs if r["split"]=="test"]; tp=[r for r in preds if r["split"]=="test"]
    boundaries={"temporal_gap":cuts(train,lambda r:r["temporal_gap"]),"minimum_visibility":cuts(train,lambda r:r["controls"]["minimum_visibility"])}
    for field in ("camera_displacement_scene_units","relative_camera_rotation_degrees","normalized_camera_displacement"):
        if dataset=="movi_e": boundaries[field]=cuts(train,lambda r,f=field:r["controls"][f])
    def assignments(r):
        out={
            "temporal_gap":tertile(float(r["temporal_gap"]),boundaries["temporal_gap"]),
            "minimum_visibility":tertile(float(r["controls"]["minimum_visibility"]),boundaries["minimum_visibility"]),
            "dynamic_status":str(r["controls"]["object_dynamic_static_status"]),
            "label":"positive" if int(r["label"])==1 else "negative",
            "negative_difficulty":"positive_not_applicable" if int(r["label"])==1 else str(r["negative_difficulty"]),
        }
        for f in ("camera_displacement_scene_units","relative_camera_rotation_degrees","normalized_camera_displacement"):
            out[f]="zero" if dataset=="movi_d" else tertile(float(r["controls"][f]),boundaries[f])
        return out
    strata=[assignments(r) for r in test]; labels=np.asarray([r["label"] for r in test],int); videos=[str(r["video_id"]) for r in test]; vm={v:i for i,v in enumerate(sorted(set(videos),key=int))}; clusters=np.asarray([vm[v] for v in videos],int)
    scores={s:np.asarray([r["scores"][s] for r in tp],float) for s in SYSTEMS}
    for variable in strata[0]:
        for level in sorted({x[variable] for x in strata}):
            sel=np.asarray([x[variable]==level for x in strata],bool); y=labels[sel]; pos=int((y==1).sum()); neg=int((y==0).sum())
            for system in SYSTEMS:
                sc=scores[system][sel]; pred=sc>=thresholds[system]
                aggregate.append({"dataset":dataset,"stratum_variable":variable,"stratum_level":level,"system":system,"pairs":int(sel.sum()),"positives":pos,"negatives":neg,"auroc":float(roc_auc_score(y,sc)) if pos and neg else None,"pr_auc":float(average_precision_score(y,sc)) if pos and neg else None,"false_match_rate":float(pred[y==0].mean()) if neg else None,"recall":float(pred[y==1].mean()) if pos else None,"threshold_source":"development_locked_90_recall"})
            metric,point,lo,hi,valid=ci_delta(labels,scores[C],scores[D],clusters,sel,20260825+len(intervals)*17)
            intervals.append({"dataset":dataset,"stratum_variable":variable,"stratum_level":level,"metric":metric,"D_minus_C":point,"paired_video_cluster_ci_low":lo,"paired_video_cluster_ci_high":hi,"pairs":int(sel.sum()),"positives":pos,"negatives":neg,"bootstrap_replicates":REPS,"valid_replicates":valid})
    inputs[dataset]={"pairs_sha256":sha(pair_path),"predictions_sha256":sha(pred_path),"locked_config_sha256":sha(lock_path),"train_derived_boundaries":{k:[float(x) for x in v] for k,v in boundaries.items()}}

def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
write_csv(OUT/"pair_strata_results.csv",aggregate); write_csv(OUT/"pair_strata_d_minus_c_intervals.csv",intervals)
manifest={"pipeline":"Descriptive release reporting for all Phase 0 predeclared pair strata","version":"1.0.0","seed":20260825,"note":"Does not change locked Phase 9 hypothesis decisions; continuous boundaries use training pairs only.","counts":{"aggregate_rows":len(aggregate),"paired_interval_rows":len(intervals)},"inputs":inputs,"outputs":{"pair_strata_results.csv":sha(OUT/"pair_strata_results.csv"),"pair_strata_d_minus_c_intervals.csv":sha(OUT/"pair_strata_d_minus_c_intervals.csv")},"checks":{"all_eight_systems_reported":len({r["system"] for r in aggregate})==8,"test_only_metrics":True,"train_only_boundaries":True,"paired_by_identical_test_pairs":True,"video_cluster_bootstrap":True}}
(OUT/"pair_strata_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps(manifest["counts"],indent=2))
