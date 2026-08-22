"""Pin the culture roster against every list that has to agree with it.

The roster used to be copy-pasted into six modules, four shell scripts and seven
configs.  Nothing failed when a copy went stale — a sweep simply skipped a culture
and the missing runs looked like runs nobody had got to yet.  These tests make the
copies agree by assertion instead of by grep.

The ``spanish-mx`` partition also has to be checked against what it claims to be.
CultureLLM's ``spanish`` concatenates Argentine and Mexican respondents, so a
Mexico-only partition must reproduce that file's second half exactly; a prompt off
by one character would silently answer a different question than the one asked.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

from src.analysis import data_loading
from src.annotation import pipeline
from src.data import visual_training_data
from src.data.culture_training_data import build_derived_culture_examples, load_wvs_culture_data
from src.data.cultures import (
    CULTURE_DIR_MAP,
    CULTURES,
    DERIVED_CULTURES,
    EXTRA_CULTURE_CONTEXTS,
    INFERENCE_ONLY_CULTURE,
)
from src.data.yfcc_schema import COUNTRY_TO_CULTURE
from src.data.yfcc_schema import STUDY_CULTURES as YFCC_CULTURES
from src.evaluation import annotation_eval, statistical_tests

CONFIG_DIR = Path("configs")
SWEEP_CONFIGS = (
    "gemma4_e2b",
    "gemma4_e4b",
    "gemma4_31b",
    "qwen3_5_2b",
    "qwen3_27b",
    "qwen3_vl_8b",
    "phi4",
)
GERMAN_ONLY_CONFIGS = ("qwen3_vl_2b", "llama3_2_3b", "muse_glimmer_30b")
SWEEP_SCRIPTS = (
    "02_train_culture_models.sh",
    "03_run_annotation.sh",
    "04_evaluate.sh",
)
CULTURELLM_DATA_DIR = Path(os.getenv("CULTURELLM_DATA_DIR", "../CultureLLM/data"))
needs_culturellm = pytest.mark.skipif(
    not (CULTURELLM_DATA_DIR / "Spanish" / "Mexico.csv").is_file(),
    reason="CultureLLM data root is not checked out beside the repository",
)


def _config(model_key: str) -> dict[str, object]:
    with open(CONFIG_DIR / f"{model_key}.yaml") as handle:
        return dict(yaml.safe_load(handle))


def _script_default(script: str) -> list[str]:
    text = Path("scripts") / script
    match = re.search(r'^CULTURES="\$\{CULTURES:-(.*?)\}"$', text.read_text(), re.MULTILINE)
    assert match, f"{script} declares no CULTURES default"
    return match.group(1).split()


def test_the_roster_is_ten_unique_names_in_sorted_order() -> None:
    assert len(CULTURES) == 10
    assert len(set(CULTURES)) == 10
    assert list(CULTURES) == sorted(CULTURES)
    assert "spanish-mx" in CULTURES
    assert INFERENCE_ONLY_CULTURE not in CULTURES


def test_every_culture_names_a_directory_in_the_culturellm_tree() -> None:
    assert set(CULTURE_DIR_MAP) == set(CULTURES)
    assert CULTURE_DIR_MAP["spanish-mx"] == CULTURE_DIR_MAP["spanish"]


@pytest.mark.parametrize(
    "module_cultures",
    [
        visual_training_data.CULTURES,
        statistical_tests.CULTURES,
        annotation_eval.STUDY_CULTURES,
        pipeline.CULTURES,
    ],
)
def test_each_consuming_module_shares_the_one_roster(module_cultures: object) -> None:
    assert module_cultures is CULTURES


def test_the_analysis_roster_appends_the_untrained_pseudo_culture() -> None:
    assert list(data_loading.CULTURES) == [*CULTURES, INFERENCE_ONLY_CULTURE]
    assert pipeline.INFERENCE_ONLY_CULTURE == INFERENCE_ONLY_CULTURE


@pytest.mark.parametrize("model_key", SWEEP_CONFIGS)
def test_every_sweep_config_lists_the_full_roster(model_key: str) -> None:
    assert _config(model_key)["cultures"] == list(CULTURES)


@pytest.mark.parametrize("model_key", GERMAN_ONLY_CONFIGS)
def test_the_scoped_configs_stay_out_of_the_sweep(model_key: str) -> None:
    assert _config(model_key)["cultures"] == ["german"]


@pytest.mark.parametrize("script", SWEEP_SCRIPTS)
def test_each_runner_defaults_to_the_full_roster(script: str) -> None:
    assert _script_default(script) == list(CULTURES)


def test_the_photo_track_keeps_its_own_nine_way_split() -> None:
    assert set(YFCC_CULTURES) == set(COUNTRY_TO_CULTURE.values())
    assert "spanish-mx" not in YFCC_CULTURES
    assert COUNTRY_TO_CULTURE["MX"] == "spanish"
    assert _script_default("08_train_yfcc.sh") == sorted(YFCC_CULTURES)


def test_the_derived_culture_declares_where_its_answers_come_from() -> None:
    spec = DERIVED_CULTURES["spanish-mx"]
    assert spec.country_csv == "Spanish/Mexico.csv"
    assert spec.aggregate_row == "Avg"
    assert spec.system_prompt_token == "Mexico"
    assert spec.question_files == ("WVQ.jsonl", "new_WVQ_1000.jsonl")
    assert set(DERIVED_CULTURES) <= set(CULTURES)


def test_every_derived_culture_carries_its_own_context_blurb() -> None:
    assert set(DERIVED_CULTURES) <= set(EXTRA_CULTURE_CONTEXTS)
    assert len(EXTRA_CULTURE_CONTEXTS["spanish-mx"]) > 500


@needs_culturellm
def test_spanish_mx_reproduces_the_mexican_half_of_spanish() -> None:
    derived = build_derived_culture_examples(DERIVED_CULTURES["spanish-mx"])
    pooled = load_wvs_culture_data("spanish")

    assert len(pooled) == 2 * len(derived)
    mexican_half = pooled[len(derived) :]
    for built, shipped in zip(derived, mexican_half, strict=True):
        assert built["messages"][1] == shipped["messages"][1]
        assert built["messages"][2] == shipped["messages"][2]


@needs_culturellm
def test_spanish_mx_differs_from_spanish_only_in_the_country_it_names() -> None:
    derived = build_derived_culture_examples(DERIVED_CULTURES["spanish-mx"])
    pooled = load_wvs_culture_data("spanish")

    assert {ex["messages"][0]["content"] for ex in derived} == {
        "You are an Mexico chatbot that know Mexico very well."
    }
    assert {ex["messages"][0]["content"] for ex in pooled} == {
        "You are an Spanish chatbot that know Spanish very well."
    }


@needs_culturellm
def test_pooling_argentina_with_mexico_contradicts_a_third_of_the_survey() -> None:
    """The premise of the comparison: the two halves of ``spanish`` disagree.

    If they agreed, ``spanish-mx`` would be a smaller copy of ``spanish`` and the
    experiment would have nothing to measure.
    """
    pooled = load_wvs_culture_data("spanish")
    half = len(pooled) // 2
    seed_questions = 50
    contradictions = sum(
        pooled[i]["messages"][2] != pooled[half + i]["messages"][2] for i in range(seed_questions)
    )
    assert contradictions == 18


@needs_culturellm
def test_the_derived_culture_loads_through_the_ordinary_entry_point() -> None:
    assert len(load_wvs_culture_data("spanish-mx")) == 1050


@needs_culturellm
def test_the_context_blurbs_cover_every_culture_that_trains_on_images() -> None:
    from src.data.culture_training_data import load_culture_contexts

    contexts = load_culture_contexts()
    assert set(CULTURES) <= set(contexts)
    assert "Mexico" in contexts["spanish-mx"]


@needs_culturellm
def test_the_derived_partition_is_half_the_size_of_the_pooled_one() -> None:
    """Guard the size confound the comparison has to report.

    ``spanish-mx`` draws one country's answers where ``spanish`` draws two, so it
    holds half the examples and takes half the optimizer steps at equal epochs.
    EXPERIMENTS.md states that ratio; should a future recipe change it, the
    write-up has to change with it.
    """
    derived = load_wvs_culture_data("spanish-mx")
    pooled = load_wvs_culture_data("spanish")
    assert len(derived) * 2 == len(pooled)
