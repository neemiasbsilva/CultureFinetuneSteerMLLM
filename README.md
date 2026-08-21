# Culture Steering via Finetuning

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-locked-DE5FE9?logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.15-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-FFD21E?logo=huggingface&logoColor=black)
![TRL](https://img.shields.io/badge/TRL-SFT-FFD21E?logo=huggingface&logoColor=black)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-1C3C3C?logo=langchain&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.15-0194E2?logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikitlearn&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-passing-D7FF64?logo=ruff&logoColor=black)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)
![tests](https://img.shields.io/badge/tests-373%20passing-4c1)

This repository extends [CultureLLM](https://arxiv.org/pdf/2402.10946)
(Li et al., NeurIPS 2024), which fine-tunes language models on World Values
Survey data across nine cultures, into the multimodal domain, now spanning a
growing set of MLLM architectures. Its annotation pipeline is built to test
whether the steering effects found in two prior prompting-only studies —
[persona prompting on urban imagery](https://arxiv.org/pdf/2605.29064),
which reported descriptive convergence alongside interpretive variation, and
[persona validity in urban sentiment perception](https://minds-lab-utfpr.github.io/MLLMs-persona-evaluation/),
which found persona-driven variation to be largely superficial — still hold
when culture is instantiated as trained adapter weights instead of a prompt
instruction.

Do culture-specific adapters change how a multimodal model perceives an urban
scene? Each architecture is fine-tuned into nine culture-specific LoRA adapters
from World Values Survey question-answer text, then annotates the same σ₃P₅
image set under a neutral prompt, so any shift in sentiment, caption or
perception tags is attributable to the adapter weights rather than to persona
instructions in the prompt.

```text
├── configs/                     one YAML per architecture: model id, LoRA, quantization
│   ├── gemma4_e2b.yaml  gemma4_e4b.yaml  gemma4_31b.yaml
│   ├── qwen3_5_2b.yaml  qwen3_vl_2b.yaml  qwen3_vl_8b.yaml  qwen3_27b.yaml
│   ├── muse_glimmer_30b.yaml    QLoRA, transformers 5.15 architecture
│   ├── llama3_2_3b.yaml         text-only base, pinned chat template
│   ├── llama_guard4_12b.yaml    QLoRA, gated safety-classifier base
│   ├── phi4.yaml
│   └── templates/               chat templates for bases that ship none
│
├── scripts/                     numbered pipeline stages, each with --help
│   ├── _common.sh               sourced helpers: die, log, require_uv
│   ├── 01_prepare_data.sh       folds + WVS training text
│   ├── 02_train_culture_models.sh   LoRA fine-tuning, resumable
│   ├── 03_run_annotation.sh     matched base/WVS annotation passes
│   ├── 04_evaluate.sh           metrics, paired bootstrap, fold tests
│   ├── 05_tests.sh              regression tests
│   └── 06_lint.sh               ruff format, ruff check, mypy
│
├── src/
│   ├── annotation/              LangGraph annotation pipeline
│   │   ├── conditions.py        the condition registry and checkpoint layout
│   │   ├── config.py            prompts, model id maps, matched-pass seeds
│   │   ├── graph.py  state.py   graph wiring and the shared state contract
│   │   └── nodes/               image_loader, assembler, annotator
│   ├── data/                    folds, WVS training text, JSONL dataset loaders
│   ├── training/                train_hf.py (CUDA/MPS), train_mlx.py (Apple), early stopping
│   ├── evaluation/              matched evaluation, metrics, Holm tests, reports
│   ├── analysis/                sentiment, similarity, topics, convergence
│   ├── inference/  utils/       generation helpers, device and model loading
│
├── notebooks/                   01..05 exploratory analysis over annotation outputs
├── tests/                       conftest.py + unit/
├── data/                        gitignored   symlink to the dataset root
├── checkpoints/                 gitignored   <culture>/<model>/cultural/ adapters
└── outputs/                     gitignored   annotations/ and evaluation/
```

## Contents

- [Install](#install)
- **Pipeline**
  - [1 — Data](#1--data)
  - [2 — Training](#2--training)
  - [3 — Annotation](#3--annotation)
  - [4 — Evaluation](#4--evaluation)
- **Reference**
  - [Conditions](#conditions)
  - [Tests and lint](#tests-and-lint)
  - [Tracking](#tracking)
  - [Citation](#citation)

Four stages, run in this order:

| # | Stage | Command | Needs |
| --- | --- | --- | --- |
| 1 | **Data** — folds and WVS fine-tuning text | `./scripts/01_prepare_data.sh` | the σ₃P₅ agreement CSV and WVS data |
| 2 | **Training** — one LoRA adapter per culture | `./scripts/02_train_culture_models.sh` | a CUDA GPU (or Apple Silicon) |
| 3 | **Annotation** — matched base and WVS passes | `./scripts/03_run_annotation.sh` | stage 2 checkpoints, the image set |
| 4 | **Evaluation** — metrics and significance | `./scripts/04_evaluate.sh` | stage 3 annotations |

---

## Install

```bash
uv sync                 # runtime dependencies
uv sync --group dev     # adds pytest, ruff, mypy, notebook tooling
cp .env.example .env    # then fill in the paths below
```

Python 3.11 is pinned in `.python-version`. Training the 27B–31B models needs a
CUDA GPU; `training.quantization: 4bit` puts them in roughly 32 GB via QLoRA.
Apple Silicon runs the MLX backend through `uv sync --group apple`.

`.env` points at sibling repositories rather than absolute paths, so a clone
placed next to them works unchanged:

| Variable | Points at |
| --- | --- |
| `PERCEPTSENT_IMAGES_DIR` | the PerceptSent image directory |
| `AGREEMENT_CSV` | the σ₃P₅ agreement labels |
| `CULTURE_CONTEXT_JSONL`, `CULTURELLM_DATA_DIR` | the CultureLLM WVS data |
| `MLFLOW_TRACKING_URI` | the tracking server, default `http://127.0.0.1:5000` |
| `HF_TOKEN` | only needed for gated base models |

---

## 1 — Data

Stage 1 builds the stratified evaluation folds and the WVS fine-tuning text.
Fine-tuning is text-only: images enter the experiment at annotation time, so no
visual supervision can leak into the adapters.

```bash
./scripts/01_prepare_data.sh          # 5 folds, seed 42
FOLDS=10 SEED=7 ./scripts/01_prepare_data.sh
```

| Artifact | Contents |
| --- | --- |
| `data/folds/fold_{k}_{train,val}.csv` | stratified image folds for stage 4 |
| `data/processed/<culture>/*.jsonl` | WVS question-answer chat records |

---

## 2 — Training

One LoRA adapter per (architecture, culture), trained on WVS text under a
culture-specific system prompt. The run is resumable: a finished combination is
skipped via its `TRAINING_DONE` sentinel, an interrupted one resumes from the
last checkpoint, and the base model must beat uniform guessing on a real batch
before the first optimizer step.

```bash
./scripts/02_train_culture_models.sh                          # every model × culture
MODELS="gemma4_e2b" CULTURES="arabic german" ./scripts/02_train_culture_models.sh
```

`MODELS` is the run list; `configs/` is the registry. Architectures added for one
culture — `muse_glimmer_30b`, `qwen3_vl_2b`, `llama3_2_3b`, `llama_guard4_12b`, all
German — are named explicitly rather than added to the default sweep. See
[EXPERIMENTS.md](EXPERIMENTS.md) for what each one is there to isolate.

Three `model` keys decide how a base is loaded, and all three are validated
against the checkpoint's own metadata before any weights are read:

| Key | Values | What it selects |
| --- | --- | --- |
| `modality` | `vision_text` (default), `text` | `AutoModelForImageTextToText` + `AutoProcessor`, or `AutoModelForCausalLM` + `AutoTokenizer` |
| `quantization` | omitted, `4bit` | immediate load at `dtype`, or QLoRA via bitsandbytes NF4 |
| `chat_template` | omitted, a `.jinja` path | the checkpoint's own template, or one from `configs/templates/` |

A text-only base still trains on the WVS track — its supervision is text — but it
has no vision path, so it never enters stage 3.

| Artifact | Contents |
| --- | --- |
| `checkpoints/<culture>/<model>/cultural/` | the adapter weights |
| `checkpoints/<culture>/<model>/cultural/TRAINING_DONE` | completion sentinel |

---

## 3 — Annotation

Every image is annotated under each condition with the same prompt and the same
per-pass seed, so the conditions differ only in which weights are loaded.
Repeated passes make answer variance measurable.

```bash
./scripts/03_run_annotation.sh                       # all models, all cultures, 5 passes
LIMIT=10 N_RUNS=2 ./scripts/03_run_annotation.sh     # smoke test
CONDITIONS="inference_only" ./scripts/03_run_annotation.sh
```

| Artifact | Contents |
| --- | --- |
| `outputs/annotations/<model>/<culture>/<condition>/annotations.jsonl` | one record per image and pass |
| `.../annotation_failures.jsonl` | generations that did not parse |

A record carries the parsed sentiment, caption, justification and perception
tags, plus the seed and the checkpoint identity that produced it.

---

## 4 — Evaluation

Repeated passes are reduced to one prediction per image by a deterministic mode
rule, every condition is restricted to the same successfully parsed images, and
uncertainty comes from resampling paired images rather than raw generation rows.

```bash
./scripts/04_evaluate.sh                     # 1,000 bootstrap samples, seed 42
N_BOOTSTRAP=10000 ./scripts/04_evaluate.sh
METRIC=accuracy ./scripts/04_evaluate.sh     # fold tests on another metric
```

| Artifact | Contents |
| --- | --- |
| `annotation_metrics_bootstrap.csv` | per-condition metrics with bootstrap intervals |
| `annotation_paired_bootstrap_deltas.csv` | paired WVS-versus-base deltas |
| `annotation_metrics_per_fold.csv` | per-fold scores for the corrected resampled *t* test |
| `holm_bonferroni_<metric>.csv` | Holm-adjusted p-values within each planned family |
| `annotation_parse_coverage.csv` | how much of the panel each condition parsed |
| `evaluation_report.md` | human-readable index over the CSVs above |

The notebooks in `notebooks/` read the same annotation records for sentiment,
similarity, topic and convergence analysis.

---

## Conditions

| Condition | Weights | Checkpoint directory |
| --- | --- | --- |
| `inference_only` | raw base model, no adapter | — |
| `wvs_cultural` | culture-specific WVS adapter | `<culture>/<model>/cultural/` |

`cultural` is accepted as a legacy alias for `wvs_cultural` so existing runs
resume without producing a second logical condition.

---

## Tests and lint

```bash
./scripts/05_tests.sh              # pytest over tests/
./scripts/05_tests.sh -k holm -vv  # pytest arguments pass straight through
./scripts/06_lint.sh               # ruff format --check, ruff check, mypy
```

The tests pin the invariants that fail silently: the condition registry and its
checkpoint layout, the refusal to fall back to a raw model when an adapter is
missing, the matched-seed gate, the mode tie-break, Holm monotonicity, and the
corrected resampled *t* statistic. Lint runs over tracked files only.

<details>
<summary><b>Troubleshooting</b></summary>

**CUDA out of memory during stage 2.** Set `training.quantization: 4bit` in the
model's config for QLoRA, or lower `training.batch_size` and raise
`gradient_accumulation` to keep the effective batch size.

**A gated base model fails to download.** Put a token with access to that
repository in `HF_TOKEN`, then re-run; stage 2 resumes from the last checkpoint.

**Annotation refuses to start with `Missing adapter`.** The condition asked for a
trained adapter that is not on disk. Run stage 2 for that culture and model, or
annotate with `CONDITIONS="inference_only"`. This is deliberate: a trained
condition never silently falls back to the raw base model.

**MLflow shows nothing.** Start the server before stages 2 and 3:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

</details>

---

## Tracking

Training and annotation log to MLflow: `train_loss` and `eval_loss` per epoch,
`eval_entropy`, the base-model health check, and the resolved data statistics.
Model selection is on validation loss.

---

## Citation

```
TODO
```

## Acknowledgments

```
TODO
```
