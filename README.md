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
![tests](https://img.shields.io/badge/tests-592%20passing-4c1)

This repository complements *Population Fidelity: Evaluating Population
Representativeness in LLMs*, which is under review. It trains the
culture-finetuned adapters that the main repository evaluates in its cultural
experiment, and extends [CultureLLM](https://arxiv.org/pdf/2402.10946) (Li et
al., NeurIPS 2024) to multimodal models along the way.

Each architecture is LoRA fine-tuned into ten cultures on World Values Survey
text. It then annotates the σ₃P₅ urban image set under a neutral prompt, which
tests whether the persona effects seen with
[urban imagery](https://arxiv.org/pdf/2605.29064) and
[urban sentiment](https://minds-lab-utfpr.github.io/MLLMs-persona-evaluation/)
still hold when the culture is in the weights rather than in the prompt. A
second track trains one adapter per architecture on the country-level response
distributions of [Cao et al. (NAACL 2025)](https://arxiv.org/abs/2502.07068), and
a third on the US subpopulation distributions of
[SubPOP (Suh et al., 2025)](https://arxiv.org/abs/2502.16761). The main
repository reads these as its `global` and `subpop` arms.

This README is how to run the code. The method, the metrics and the findings are
in the paper.

## Install

```bash
uv sync --group dev
cp .env.example .env
```

`uv sync` is exact, so name every group in one command: `dev` for tests, lint
and notebooks, and `apple` for the MLX backend on Apple Silicon. `.env` points
at the sibling repositories by relative path: PerceptSent's images, the σ₃P₅
labels and CultureLLM's WVS data. A clone placed next to them works unchanged.
`HF_TOKEN` is needed only for gated bases and for publishing.

## Stages

| # | Stage | Command | Needs |
| --- | --- | --- | --- |
| 1 | **Data**: folds and WVS training text | `./scripts/01_prepare_data.sh` | σ₃P₅ labels, CultureLLM data |
| 2 | **Training**: ten cultures per architecture | `./scripts/02_train_culture_models.sh` | CUDA GPU or Apple Silicon |
| 3 | **Annotation**: matched base and fine-tuned passes | `./scripts/03_run_annotation.sh` | stage 2, the image set |
| 4 | **Evaluation**: metrics and significance | `./scripts/04_evaluate.sh` | stage 3 |
| 9 | **Publish**: adapters to the Hub | `./scripts/09_push_hub.sh push` | `HF_TOKEN` with write access |
| 10 | **Distributional**: one Cao et al. adapter per architecture | `./scripts/10_train_distributional.sh` | SimLLMCultureDist clone, CUDA GPU |
| 11 | **SubPOP**: one Suh et al. adapter per architecture | `./scripts/11_train_subpop.sh` | access to `jjssuh/subpop`, subpop clone, CUDA GPU |

```bash
./scripts/01_prepare_data.sh
FOLDS=10 SEED=7 ./scripts/01_prepare_data.sh

./scripts/02_train_culture_models.sh
MODELS="gemma4_e2b" CULTURES="arabic german" ./scripts/02_train_culture_models.sh

./scripts/03_run_annotation.sh
LIMIT=10 N_RUNS=2 ./scripts/03_run_annotation.sh
CONDITIONS="inference_only" ./scripts/03_run_annotation.sh

./scripts/04_evaluate.sh
N_BOOTSTRAP=10000 METRIC=accuracy ./scripts/04_evaluate.sh

./scripts/09_push_hub.sh status
DRY_RUN=1 ./scripts/09_push_hub.sh push

./scripts/10_train_distributional.sh
MODELS="qwen3_vl_2b" STEPS="train" ./scripts/10_train_distributional.sh
CULTURES=global MODELS=qwen3_vl_2b PROBLEM=distributional ./scripts/09_push_hub.sh push

./scripts/11_train_subpop.sh

./scripts/05_tests.sh && ./scripts/06_lint.sh
```

Every script takes `--help` and reads its options from the environment. Training
skips a finished pair through its `TRAINING_DONE` sentinel and resumes an
interrupted one from its last checkpoint. It also stops before the first step if
the base cannot beat uniform guessing on a real batch. `MODELS` is the run list
and `configs/` the registry. `muse_glimmer_30b`, `qwen3_vl_2b` and `llama3_2_3b`
are trained for German alone and must be named explicitly. `llama3_2_3b` is
text-only, so it never enters stage 3. The 27B–31B bases need
`training.quantization: 4bit`, which fits them in about 32 GB. Stages 2, 3, 10
and 11 log to MLflow, so start the server first with
`uv run mlflow server --host 127.0.0.1 --port 5000`.

The ten cultures are CultureLLM's nine plus `spanish-mx`, which is the Mexican
half of the Argentine and Mexican respondents that `spanish` pools. Stage 3 runs
two conditions: `inference_only` is the raw base, and `wvs_cultural` (legacy
alias `cultural`) is the culture's adapter. Both use the same prompt and
per-pass seeds. A missing adapter stops the run and never falls back to the base.

Stage 10 copies Cao et al.'s splits from a clone pinned at `402ee0b` and trains
one adapter per architecture under the pseudo-culture `global`, following their
recipe. It then scores the held-out test split with the adapter on and off.
`data.exclude_evaluation_items: true` removes the four evaluation items that the
reference protocol keeps. If stage 10 runs out of memory, `batch_size: 4` with
`gradient_accumulation: 2` keeps the recipe's effective batch of 8.

Stage 11 fetches the gated SubPOP tables at a pinned revision and trains one
adapter per architecture under the pseudo-culture `subpop`, following Suh et
al.'s QA steering prompt, forward-KL loss and recipe, then scores SubPOP-Eval
with the adapter on and off. Both stages also take `MODELS="muse_glimmer_30b"`,
which trains in 4-bit and needs the card to itself.

## Artifacts

Folds and WVS text land in `data/`, and adapters in
`checkpoints/<culture>/<model>/<problem>/`: `cultural` for stage 2 and
`global/<model>/distributional` for stage 10 and `subpop/<model>/subpop` for
stage 11. Annotations go to
`outputs/annotations/<model>/<culture>/<condition>/`. Evaluation tables go to
`outputs/evaluation/`, indexed by `evaluation_report.md`, and the held-out
distributional fits to `outputs/evaluation/{distributional,subpop}/<model>/heldout.json`.
The notebooks in `notebooks/` read the annotation records.

## Adapters

The adapters will be published on Hugging Face with the paper, and their link
will go here. They are laid out as one repository with every adapter under
`<culture>/<backbone>/<problem>`, the same tree as `checkpoints/`. Until then,
stage 2, 10 or 11 rebuilds any of them locally.

The survey data are the World Values Survey's. Cao et al.'s tables are their
per-country percentages from Wave 7 (Haerpfer et al., 2022). Their repository
ships no license file, so the tables are never redistributed here, and
`prepare` records their provenance beside the local copy. The SubPOP tables come
from Pew Research Center's American Trends Panel and NORC's General Social
Survey, which bear no responsibility for the analyses here. They are gated under
CC-BY-NC-SA-4.0 and are never redistributed either.

## Citation

```
TODO
```

## Acknowledgments

```
TODO
```
