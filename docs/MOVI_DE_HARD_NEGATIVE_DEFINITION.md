# Locked MOVi-D/E hard-negative definition

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Definition version:** 1.0.0  
**Lock date:** 2026-08-25  
**Status:** Locked after capacity audit

## Definition

A pair is a hard negative only when all of the following are true:

1. Both observations pass the locked Phase 1 visibility, mask-area, and valid-depth gates.
2. Both observations come from the same dataset, locked pool, and video.
3. They come from different frames and different simulator object instances.
4. The instances have the same semantic category.
5. Their projected scale distance passes the dataset-specific threshold below.

The projected scale distance is

`abs(ln(mask_area_a / mask_area_b))`,

where mask area is the source-resolution, unpadded instance segmentation area before cropping or resizing. Comparison uses the full-precision value and is inclusive (`<=`). Pairs are deduplicated as unordered observation-ID pairs.

## Locked dataset thresholds

Each threshold is the 25th percentile of all eligible same-category negative candidates in that dataset's locked training pool, using NumPy's `higher` quantile rule. This is the smallest observed threshold that retains at least 25% of training candidates. The training threshold is then applied unchanged to development and test.

| Dataset | Absolute log-area cutoff | Maximum larger/smaller area ratio | Training candidates | Retained | Retained fraction |
|---|---:|---:|---:|---:|---:|
| MOVi-D | 0.4837969514 | 1.622222 | 365,402 | 91,354 | 25.001% |
| MOVi-E | 0.4613455665 | 1.586207 | 498,534 | 124,643 | 25.002% |

The thresholds are intentionally dataset-specific. They cannot be recalculated from development or test data and cannot be relaxed during pair sampling.

## Pre-lock capacity audit

| Dataset | Pool | Hard-negative target | Available | Capacity/target | Videos represented |
|---|---|---:|---:|---:|---:|
| MOVi-D | Train | 1,500 | 91,354 | 60.9× | 83/90 |
| MOVi-D | Dev | 500 | 30,325 | 60.7× | 29/30 |
| MOVi-D | Test | 500 | 23,664 | 47.3× | 28/30 |
| MOVi-E | Train | 1,500 | 124,643 | 83.1× | 88/90 |
| MOVi-E | Dev | 500 | 31,393 | 62.8× | 30/30 |
| MOVi-E | Test | 500 | 35,065 | 70.1× | 30/30 |

Every pool passes with substantial margin. Short-, medium-, and long-gap candidates and static/dynamic combinations are present. Pool-level quotas are therefore feasible without requiring every video to contribute hard negatives.

## Same-asset diagnostic subgroup

A qualifying hard negative whose two distinct simulator instances share the same `asset_id` is flagged as `very_hard_same_asset`. This is a diagnostic subgroup inside the hard-negative pool, not a separate sampling quota. The audit found such candidates in MOVi-D train/dev and MOVi-E train; their absence from another pool is not a shortfall.

## Leakage and change control

Semantic category, simulator instance identity, and asset identity may be used only for pair construction, labels, and diagnostic strata. They are forbidden as model features.

Pair generation must fail rather than silently relaxing the cutoff or quota. Any change requires a prospective protocol amendment and a replacement definition freeze before pair manifests are created.
