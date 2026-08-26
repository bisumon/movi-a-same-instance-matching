# Phase 10: MOVi-D/E failure analysis

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Operating point:** each system’s development-selected 90%-recall threshold  
**Selection:** 24 unique locked-test pairs, post-confirmatory  
**Status:** Complete and locked

## Design and balance

The gallery contains three errors in every dataset × system × error-type cell:

| Dataset | C false positives | C false negatives | D false positives | D false negatives | Total |
|---|---:|---:|---:|---:|---:|
| MOVi-D | 3 | 3 | 3 | 3 | 12 |
| MOVi-E | 3 | 3 | 3 | 3 | 12 |
| Total | 6 | 6 | 6 | 6 | **24** |

The selected pairs are also exactly balanced as follows:

- 12 static-static pairs and 12 pairs involving at least one dynamic object;
- 6 hard-negative and 6 easy-negative false positives;
- 8 short, 8 medium, and 8 long temporal gaps;
- 12 C and 12 D errors;
- 12 false positives and 12 false negatives.

MOVi-D contributes only the structurally correct zero-motion stratum. MOVi-E contributes 5 low-, 2 medium-, and 5 high-motion items. The two medium-motion items are necessary because MOVi-E contains only nine C and seven D false-positive method-events at the locked thresholds; exact pair uniqueness plus system, error, dynamics, and hard/easy balance cannot be achieved using low/high false positives alone. The capacity audit records every requested slot and its candidate count.

No same-asset test negatives exist in either locked test manifest, so the predefined `matched-asset distractor` tag has zero capacity. This is a property of the locked sample, not a selection omission.

## Qualitative patterns

Diagnosis counts describe the deliberately balanced gallery and must not be interpreted as population prevalence.

| Rule-based diagnosis | Selected items | Interpretation |
|---|---:|---|
| Matched-category distractor | 6 | Similar-category, similar-scale negatives remain difficult even with geometry. |
| Biased depth/centroid estimate | 4 | Visible-surface depth or centroid changes unusually strongly between observations. |
| Occlusion-truncated mask | 4 | Low support or image-edge truncation removes reliable appearance and geometry cues. |
| Long-gap motion | 4 | Dynamic motion or a long interval changes the visible object evidence. |
| Ambiguous appearance | 3 | Easy negatives can still look unexpectedly similar, or positives can change substantially. |
| Camera-motion misalignment | 2 | High-motion MOVi-E pairs missed by C are correctly retained by pose-aligned D. |
| Possible pose-transform error | 1 | One high-motion pair is retained by C but missed by D, suggesting transform or visible-surface instability. |

### False positives

The hard-negative errors illustrate the expected matched-category failure mode: semantic and projected-scale matching can leave two different objects compatible in both appearance and coarse geometry. Easy-negative failures show that category mismatch does not guarantee visual dissimilarity. Several false positives also have weak or truncated mask support, which can suppress the features that would otherwise separate the objects.

### False negatives

False negatives are dominated by changing visible evidence rather than a single universal cause. Dynamic objects and long gaps change appearance, depth summaries, and visible-surface centroids. Low-visibility or edge-truncated masks remove stable object support. On two high-camera-motion MOVi-E examples, C fails while D is correct, providing qualitative counterparts to the quantitative motion-stratum result from Phase 9. The selected D-specific high-motion miss is a useful counterexample: pose alignment helps on average but does not make every transformed observation more reliable.

## Interpretation boundary

The diagnosis labels are deterministic review hypotheses based on pair controls, model disagreement, mask support, depth/centroid changes, and camera motion. They are not causal ground truth and have not been manually adjudicated. Selection occurred only after predictions, thresholds, and confirmatory conclusions were locked; it cannot modify any metric or decision.

The analysis remains limited to simulator-provided crops, instance masks, depth, intrinsics, and camera poses. It does not estimate failure rates under learned detection, monocular depth, estimated pose, real camera artifacts, or cross-video re-identification.

## Outputs

- Interactive gallery: `failure_gallery/movi_de_phase10/failure_gallery.html`
- Visual overview: `failure_gallery/movi_de_phase10/contact_sheet.png`
- Machine-readable selected cases: `failure_gallery/movi_de_phase10/selected_failure_pairs.jsonl`
- Review table: `failure_gallery/movi_de_phase10/failure_review.csv`
- Capacity audit: `failure_gallery/movi_de_phase10/capacity_audit.json`
- Selection and checksum manifest: `failure_gallery/movi_de_phase10/selection_manifest.json`
