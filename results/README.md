# results/

Outputs of real runs only (git-ignored). Each evaluation writes one JSON file with the
metrics, confusion matrix, label distributions, warnings and the full configuration
(dataset, experiment, model, split, seed). The Evaluation tab lists these files.

* `fever_*` from `training/evaluate_verifier.py` (Experiments A and B) or the FEVER end-to-end tab
* `ragtruth_*` from the RAGTruth tab or `python -m src.cli eval-ragtruth` (Experiment C)
* `controlled_mutations_*` from the batch mutation experiment (Experiment D)
* `controlled_tests__*` from the reviewed-example controlled test run with both verifiers on the same
  generated answers (Experiment D). This file is not a metrics report, so the Evaluation tab does not list it.
* `thresholds_*.json` from `evaluate_verifier.py --tune` (validation split only)
* `dataset_report.json` from `python -m src.cli dataset-report` (dataset counts; no model involved)
