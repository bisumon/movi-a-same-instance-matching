# MOVi-D/E camera-coordinate validation

**Protocol:** MOVI-DE-POSE-001  
**Validation date:** 2026-08-25  
**Pilot scope:** 20 locked MOVi-D videos and 20 locked MOVi-E videos  
**Verdict:** Pass

The camera-coordinate transformation is numerically correct and produces the behavior required by the approved protocol. The adapter processed 4,452 eligible MOVi-D observations and 5,275 eligible MOVi-E observations without a numerical gate failure.

## Protocol gates

| Gate | Locked tolerance or expectation | MOVi-D | MOVi-E | Result |
|---|---:|---:|---:|---|
| Camera → world → camera round trip | ≤ 1e-8 scene units | 8.76e-15 | 1.74e-14 | Pass |
| Pixel reprojection | ≤ 1e-6 pixels | 3.55e-14 | 4.02e-14 | Pass |
| Rotation orthonormality | ≤ 1e-10 | 6.66e-16 | 1.33e-15 | Pass |
| Static-object stability | World coordinates should reduce motion-induced drift | Fixed-camera control is approximately unchanged | World/camera pooled deviation ratio = 0.199 | Pass |
| Convention assertions | wxyz, camera-to-world, explicit axes and scene units | Recorded | Recorded | Pass |
| Reason-coded exclusions | Invalid observations have reason codes | 1,044 excluded | 1,205 excluded | Pass for pilot |

## Convention and behavioral validation

The implementation back-projects radial depth into CV camera axes (`x` right, `y` down, `z` forward), applies the explicit multiplier `[1, -1, -1]` to obtain Kubric camera axes (`x` right, `y` up, `z` backward), then applies the normalized `wxyz` camera-to-world quaternion and camera translation. The inverse uses the transposed rotation and the same axis multiplier.

Self-consistent round trips alone cannot detect a mutually wrong axis convention, so the moving-camera pilot supplies the important external check. Across 200 eligible static MOVi-E tracks, the pooled median centroid deviation fell from 0.2070 scene units in camera coordinates to 0.04123 in world coordinates—a ratio of 0.199. Every one of the 20 MOVi-E videos improved. Restricting each track to observations whose visibility was at least 75% of that track's maximum reduced the ratio further to 0.174, so the conclusion is not driven by strongly truncated masks.

MOVi-D is the falsification control: its camera translation path was exactly zero and its maximum apparent rotation was only 2.09e-6 degrees. As expected, the median per-track world/camera deviation ratio was approximately one (0.980 using all observations; 1.031 in the high-visibility sensitivity analysis). This pattern—large improvement only when the camera moves—is strong evidence that the transform direction, quaternion order, and axis conversion are correct.

## Exclusions and caveats

All pilot exclusions were machine-readable and attributable to the locked visibility and mask-area gates. No included model record contained a non-finite numerical value. Invalid masked depth samples are filtered; if fewer than 32 valid pixels remain, the adapter emits `insufficient_valid_depth_pixels`. An invalid camera quaternion correctly hard-fails the video because all of its observations would be compromised.

Before confirmatory extraction, two small documentation hardening changes are advisable:

1. Add explicit `world_handedness`, `world_up_axis`, `cv_to_kubric_axis_multiplier`, and raw-scene-unit fields to the extraction manifest instead of relying on the paired axis descriptions and adapter documentation.
2. Add explicit `non_finite_reconstruction` and `implausible_reconstruction` reason codes, even though neither condition occurred in this pilot.

The complete machine-readable evidence is in `results/movi_de_camera_coordinate_validation.json`; per-video diagnostics remain in each pilot run's `pose_validation.json`.
