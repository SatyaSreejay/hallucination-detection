# models/

Model weights are git-ignored and never downloaded without an explicit command.

* `hf-cache/`: Hugging Face cache for pretrained checkpoints, filled by
  `python -m src.cli download-nli` or `--allow-download` on the training/evaluation scripts.
* `fever-verifier/` (default output of `training/train_verifier.py`): fine-tuned checkpoint,
  tokenizer and `training_record.json`. A checkpoint counts as "fine-tuned by this project"
  only if `training_record.json` exists with `"status": "completed"`. A failed run writes
  `training_record.failed.json` instead, and the app labels the checkpoint as untrustworthy.
