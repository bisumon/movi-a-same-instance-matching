# MOVi-D/E final deliverables

This index is the release entry point for the camera-pose extension. Scientific
decisions remain governed by the frozen stage manifests under `manifests/movi_de/`.

1. **Independent, reproducible code.** Use `./run_movi_de_experiments.sh`.
   Environment pins are in `requirements.txt` and `requirements-lock.txt`;
   operational details are in `docs/MOVI_DE_REPRODUCIBILITY.md`.
2. **Seeded manifests, splits, configurations, and predictions.** Video locks are
   in `manifests/movi_de/`; pair manifests are in `manifests/pairs/movi_de/`;
   fixed systems are in `configs/movi_de_phase6_systems.json`; publishable
   per-pair Phase 7, in-domain, and transfer predictions are under
   `predictions/movi_de/`.
3. **Results table.** `docs/MOVI_DE_RESULTS_TABLES.xlsx` contains all clean
   baselines and controls, development/test metrics, paired confidence intervals,
   pose-noise results, all predeclared pair strata, paired stratum intervals,
   transfer results, and measured latency
   components. Machine-readable source results remain under `results/`.
4. **Failure analysis and conclusions.** Open
   `failure_gallery/movi_de_phase10/failure_gallery.html` for the 24-item gallery.
   Conclusions are in `docs/MOVI_DE_CONCLUSIONS_AND_NEXT_STEPS.md` and `.docx`.
5. **Sources and licenses.** See `docs/SOURCES_AND_LICENSES.md`, `LICENSE.md`, and
   captured notices under `third_party_licenses/`.

The machine-readable checksum index is
`manifests/movi_de/final_release_manifest.json`.
Release validation evidence is in `results/movi_de_final/release_validation.json`.
