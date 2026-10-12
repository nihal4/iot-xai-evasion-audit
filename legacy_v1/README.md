# Legacy experiment results archive

This archive contains outputs from the completed old Kaggle experiment. No model, attack, explanation, retraining, or rerun was performed to create this archive.

## Included

- Baseline and training-history CSVs.
- FGSM/PGD attack results.
- SHAP/LIME explanation metrics.
- Original seed-level paired statistics.
- Four vector PDF figures.
- Original configuration, environment, and both dataset metadata/split summaries.
- SHA-256 hashes for archived metadata/configuration artifacts.
- Reconstructed Edge-IIoTset class-level breakdown from archived aggregate metrics.
- Derived clean-versus-adversarial-training effect sizes.
- Derived clustered-bootstrap sensitivity analysis using training seeds as clusters.
- Figure captions and post-hoc reanalysis audit.

## Known archival limits

The old archive does not contain the frozen feature matrices, validation probability arrays, or per-record SHAP/LIME attribution arrays. Therefore an empirical validation-threshold sweep, partition-level duplicate-row check, feature-correlation analysis, and attribution-mass flatness analysis cannot be recovered without restoring those files or rerunning. The paper treats these as permanent limitations of the completed archive.

Peak process/GPU resource usage was not captured by the old run and is not fabricated here. Raw datasets are intentionally excluded.
