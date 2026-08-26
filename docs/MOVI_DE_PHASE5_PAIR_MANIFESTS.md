# MOVi-D/E Phase 5 locked pair manifests

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Seed:** 20260825  
**Status:** Locked before feature extraction, model fitting, development-set tuning, or test evaluation

The confirmatory benchmark contains 10,000 independently sampled within-video pairs for each dataset. Each dataset has 6,000 training, 2,000 development, and 2,000 test pairs. Every pool is exactly 50% positive, 25% hard negative, and 25% easy negative. No pair crosses a dataset, video, or locked video pool.

Temporal gaps are balanced within every dataset, pool, and pair kind: the training allocations are exactly equal across 1–5, 6–11, and 12–23 frames; development and test differ by at most one pair because their quotas are not divisible by three. Video contribution is deterministically balanced and capped at 250 pairs. The realized maximum is 67 pairs per video in every pool.

Hard negatives use the frozen full-precision dataset thresholds without relaxation: 0.4837969513780713 absolute log area-ratio for MOVi-D and 0.461345566502621 for MOVi-E. Same-asset/different-instance pairs remain a diagnostic subgroup with no quota. The realized manifests contain 11 such pairs in MOVi-D and 18 in MOVi-E.

Each JSONL row contains canonical observation IDs, label, negative difficulty, video/frame/instance join keys, temporal-gap bin, eligibility controls, source mask areas, category and asset relations, object dynamic/static status, and camera translation/rotation controls. Category, asset, instance, and dynamic/static metadata are restricted to sampling, labels, and diagnostic strata; they are not model features.

Normalized camera displacement is the frame-to-frame camera translation divided by the mean camera-to-world-origin distance at the two frames, matching the frozen protocol's operational camera-to-scene-distance convention. Fixed-camera numerical roundoff below the declared tolerance is recorded as exact zero. Consequently all MOVi-D camera-motion controls are exactly zero, while MOVi-E retains nonzero motion.

The authoritative checksum envelope is `manifests/movi_de/phase5_pair_manifest_freeze.json`. Pair generation fails on any locked quota shortfall and has no code path that relaxes cutoffs or quotas. Any later change requires a prospective amendment and replacement freeze before model fitting.
