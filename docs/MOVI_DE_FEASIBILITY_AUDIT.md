# MOVi-D/E pilot feasibility audit

**Protocol:** MOVI-DE-POSE-001  
**Audit date:** 2026-08-25  
**Verdict:** Feasible for the 150-video-per-dataset confirmatory design

## Gate summary

| Check | MOVi-D | MOVi-E | Result |
|---|---:|---:|---|
| Eligible pilot videos | 20/20 | 20/20 | Pass |
| Included observations | 4,452 | 5,275 | Pass |
| Observation exclusion rate | 19.0% | 18.6% | Acceptable and reason-coded |
| Positive capacity / scaled target | 71.6x | 86.4x | Pass |
| Hard-negative capacity / scaled target | 62.5x | 86.1x | Pass |
| Easy-negative capacity / scaled target | 1152.8x | 1584.6x | Pass |

The pilot-equivalent quota is 1,335 pairs per 20 videos: 667 positive, 334 hard-negative, and 334 easy-negative pairs. This preserves the protocol's per-video density for a 6,000-pair, 90-video training pool; development and test use the same density.

## Pair capacity

| Dataset | Positive | Hard negative | Easy negative | Videos with hard negatives | Same-asset negatives |
|---|---:|---:|---:|---:|---:|
| MOVi-D | 47,783 | 20,859 | 385,031 | 18/20 | 0 |
| MOVi-E | 57,647 | 28,744 | 529,261 | 20/20 | 0 |

The pilot-only scale cutoffs retaining at least 25% of same-category candidates were 0.501 for MOVi-D and 0.523 for MOVi-E. These values are not locked and must be recomputed independently from each final training pool.

Positive, hard-negative, and easy-negative candidates exist in short (1–5), medium (6–11), and long (12–23) temporal-gap ranges in both datasets. Static and dynamic positive candidates are present. Dynamic–dynamic hard negatives are less common but have no predeclared quota.

## Camera-motion and strata coverage

MOVi-E covers camera path lengths from 0.042 to 3.419 scene units and start-to-end rotations from 0.090° to 31.260°. This is sufficient to derive train-only motion tertiles and audit the motion interaction.

MOVi-D translation is exactly zero and its numerical rotation is effectively zero. Camera-motion tertiles are therefore degenerate on D by design; D should be reported as a zero-motion falsification control, while low/medium/high motion strata should be interpreted on E only.

## Scale, storage, and stopping-rule assessment

After reserving the 20 pilot videos, 230 of the official 250 validation videos remain per dataset. The minimum confirmatory requirement is 150, leaving an 80-video margin per dataset. The current raw pilot download occupies 0.49 GiB; linear extrapolation to all validation shards is approximately 3.95 GiB. Phase 1 outputs occupy 0.13 GiB for 40 videos, implying approximately 0.94 GiB for the minimum confirmatory sample.

The observed Phase 1 write spans were 40.6 seconds for 20 MOVi-D videos and 45.6 seconds for 20 MOVi-E videos. Linear extrapolation is approximately 5.7 minutes when the two dataset jobs run concurrently, or 10.8 minutes sequentially, excluding download time and allowing for ordinary video-to-video variation.

No protocol stopping rule is triggered: data are obtainable, coordinate validation passed, aggregate hard-negative capacity is ample, and storage/compute requirements are modest for the current machine.

## Constraints to preserve in the confirmatory pipeline

1. Enforce pair quotas at the pool level. Do not require every video to supply hard negatives; 2 of 20 MOVi-D pilot videos had none after the provisional scale filter.
2. Keep the very-hard same-asset subgroup diagnostic-only. No such candidate occurred in this pilot, consistent with the protocol's no-quota rule.
3. Recompute the hard-negative scale cutoff and all continuous stratum tertiles using the locked training pool only, then freeze them before development or test reporting.
4. Keep all 20 pilot videos out of the confirmatory test pool. If any pilot video is proposed for training or development, record that choice in the split manifest before pair generation.
5. Run a capacity audit after each 90/30/30 split is locked; the pilot establishes feasibility but does not guarantee identical category composition in every random pool.

Machine-readable details are in `results/movi_de_feasibility_audit.json`.
