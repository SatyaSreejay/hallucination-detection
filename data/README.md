# data/

Nothing here is downloaded automatically. Datasets are git-ignored.

| Path | What | Source |
| --- | --- | --- |
| `fever/train.jsonl` | FEVER training split (145,449 claims) | https://fever.ai/dataset/fever.html |
| `fever/shared_task_dev.jsonl` or `paper_dev.jsonl` / `paper_test.jsonl` | Labelled held-out splits used for final evaluation | same |
| `fever/wiki-pages/wiki-*.jsonl` | FEVER's 2017 Wikipedia dump (`wiki-pages.zip`, about 1.7 GB unzipped); needed to turn evidence ids into evidence text | same |
| `fever/prepared/` | Output of `python -m training.prepare_fever` | generated |
| `ragtruth/response.jsonl`, `ragtruth/source_info.jsonl` | RAGTruth (MIT licence), from the repo's `dataset/` folder | https://github.com/ParticleMedia/RAGTruth |
| `controlled_examples.json` | Reviewed true/false claim pairs for the Controlled Hallucination Test (tracked in git) | written for this project |
| `examples/` | Small example inputs for the batch mutation experiment (tracked in git) | written for this project |
| `cache/` | Wikipedia API response cache | generated |
