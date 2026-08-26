# MOVi-D/E confirmatory video selection and pool lock

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Seed:** `20260825`  
**Lock date:** 2026-08-25  
**Status:** Locked after post-split capacity audit

## Outcome

Exactly 150 technically eligible, non-pilot videos were selected independently from each dataset and assigned to 90 train, 30 development, and 30 test videos. All 40 pilot videos were excluded. No confirmatory model results were accessed during selection.

| Dataset | Full inventory | Pilot excluded | Other ineligible | Eligible non-pilot candidates | Selected | Train/dev/test |
|---|---:|---:|---:|---:|---:|---:|
| MOVi-D | 250 | 20 | 2 | 228 | 150 | 90/30/30 |
| MOVi-E | 250 | 20 | 0 | 230 | 150 | 90/30/30 |

MOVi-D videos `1894` and `5098` were excluded because fewer than two instances had at least two eligible observations. The complete inventories preserve these reason codes.

## Balance method and result

The deterministic selector used the frozen criteria:

- number of instances with at least two eligible observations;
- number of eligible dynamic instances;
- mean visibility across eligible observations;
- camera translation magnitude; and
- camera rotation magnitude.

MOVi-D motion values were treated as structurally zero, consistent with its fixed-camera role. MOVi-E translation and rotation were balanced directly. The 150-video subsample was matched to the eligible non-pilot inventory, then videos were assigned to pools by deterministic swap optimization. The balance gate was maximum absolute standardized mean difference (SMD) ≤ `0.20`.

| Dataset | Max selection SMD vs. eligible inventory | Max pool SMD vs. selected set | Result |
|---|---:|---:|---|
| MOVi-D | 0.0040 | 0.0084 | Pass |
| MOVi-E | 0.0093 | 0.0164 | Pass |

## Mandatory post-split capacity gate

The hard-negative scale cutoff was derived independently from each locked training pool and then applied unchanged to that dataset's development and test pools:

- MOVi-D cutoff: `0.4837969514` absolute log area-ratio.
- MOVi-E cutoff: `0.4613455665` absolute log area-ratio.

Every positive, hard-negative, and easy-negative quota passed. The tightest capacity was MOVi-D test hard negatives: 23,664 unique candidates for a target of 500 (`47.3×`). The tightest MOVi-E capacity was development hard negatives: 31,393 for a target of 500 (`62.8×`). Short-, medium-, and long-gap candidates and static/dynamic candidates are available in every pool.

Same-asset, different-instance candidates occurred in the full selection. They remain diagnostic-only with no quota, as required by the protocol.

## Integrity checks

- All 38 downloaded source/schema files passed size and SHA-256 verification.
- All 12 primary selection outputs passed independent checksum verification.
- Train, development, and test pools are disjoint within each dataset.
- No pilot video appears in a confirmatory pool.
- A complete repeated run produced byte-identical combined and per-pool manifests (`8/8`).
- Matching numeric IDs across MOVi-D and MOVi-E do not indicate paired scenes; the datasets remain independent experimental samples.

The authoritative checksum envelope is `manifests/movi_de/confirmatory_video_pool_freeze.json`. Any later change to a selected ID or split assignment requires a prospective amendment before pair generation.
