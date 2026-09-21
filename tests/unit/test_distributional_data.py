"""Pin the reference records against the tensors the distributional trainer sees."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from src.data.distributional_data import (
    ANSWER_PREFIX,
    EVALUATION_ITEMS,
    MAX_OPTIONS,
    OPTION_LETTERS,
    SOURCE_COMMIT,
    SPLIT_FILES,
    TARGET_PAD,
    DistributionalCollator,
    OptionTokenError,
    PromptEncoder,
    RecordError,
    build_prompt,
    copy_source_data,
    count_items,
    country_of,
    dedupe_records,
    filter_evaluation_items,
    load_split,
    pad_token_id_for,
    parse_record,
    resolve_option_token_ids,
    split_summary,
    write_provenance,
)

CULTURELLM_DATA_DIR = Path(os.getenv("CULTURELLM_DATA_DIR", "../CultureLLM/data"))
needs_culturellm = pytest.mark.skipif(
    not (CULTURELLM_DATA_DIR / "Spanish" / "Mexico.csv").is_file(),
    reason="CultureLLM data root is not checked out beside the repository",
)

RecordFactory = Callable[..., dict[str, Any]]
TokenizerFactory = Callable[..., Any]


def test_a_record_becomes_the_reference_prompt_ending_inside_the_parenthesis(
    raw_survey_record: RecordFactory,
) -> None:
    record = parse_record(raw_survey_record())

    prompt = build_prompt(record)

    assert prompt == (
        "How would someone from Andorra answer the following question:\n\n"
        "Taking all things together, would you say you are? \nHere are the options: \n"
        "(A)Very happy, (B)Quite happy, (C)Not very happy" + ANSWER_PREFIX
    )
    assert prompt.endswith("(")


def test_the_target_follows_option_order_not_distribution_key_order(
    raw_survey_record: RecordFactory,
) -> None:
    record = parse_record(raw_survey_record())

    assert record.letters == ("A", "B", "C")
    assert record.distribution == pytest.approx((0.30, 0.60, 0.10))


def test_x_first_and_descending_options_keep_their_file_order(
    raw_survey_record: RecordFactory,
) -> None:
    record = parse_record(
        raw_survey_record(
            record_id="119",
            options=("(X)Hard to say", "(B)Somewhat", "(A)Very"),
            dist={"A": 25.0, "B": 75.0, "X": 0.0},
        )
    )

    assert record.letters == ("X", "B", "A")
    assert record.distribution == pytest.approx((0.0, 0.75, 0.25))


def test_the_target_sums_to_one_and_keeps_genuine_zeros(raw_survey_record: RecordFactory) -> None:
    record = parse_record(raw_survey_record(dist={"A": 0.0, "B": 100.0, "C": 0.0}))

    assert sum(record.distribution) == pytest.approx(1.0)
    assert record.distribution[0] == 0.0
    assert record.distribution[2] == 0.0


def test_distribution_keys_that_are_not_the_option_letters_are_rejected(
    raw_survey_record: RecordFactory,
) -> None:
    with pytest.raises(RecordError, match="do not match the option letters"):
        parse_record(raw_survey_record(dist={"A": 50.0, "B": 50.0}))


def test_percentages_that_do_not_sum_to_one_hundred_are_rejected(
    raw_survey_record: RecordFactory,
) -> None:
    with pytest.raises(RecordError, match=r"sum to 99\.0000, not 100"):
        parse_record(raw_survey_record(dist={"A": 50.0, "B": 40.0, "C": 9.0}))


def test_a_negative_percentage_is_rejected(raw_survey_record: RecordFactory) -> None:
    with pytest.raises(RecordError, match="negative percentage"):
        parse_record(raw_survey_record(dist={"A": 105.0, "B": -5.0, "C": 0.0}))


def test_an_option_without_a_letter_prefix_is_rejected(raw_survey_record: RecordFactory) -> None:
    with pytest.raises(RecordError, match=r"lacks a \(letter\) prefix"):
        parse_record(
            raw_survey_record(options=("(A)Very happy", "Quite happy", "(C)Not very happy"))
        )


def test_a_letter_outside_the_reference_alphabet_is_rejected(
    raw_survey_record: RecordFactory,
) -> None:
    with pytest.raises(RecordError, match="is not one of"):
        parse_record(
            raw_survey_record(
                options=("(A)Very happy", "(B)Quite happy", "(Z)Other"),
                dist={"A": 50.0, "B": 50.0, "Z": 0.0},
            )
        )


def test_a_repeated_letter_is_rejected(raw_survey_record: RecordFactory) -> None:
    with pytest.raises(RecordError, match="repeats option letter"):
        parse_record(
            raw_survey_record(options=("(A)Very happy", "(A)Quite happy", "(C)Not very happy"))
        )


def test_country_of_reads_the_reference_instruction_and_rejects_others() -> None:
    instruction = "How would someone from Hong Kong SAR answer the following question:\n\n"

    assert country_of(instruction) == "Hong Kong SAR"
    with pytest.raises(RecordError, match="does not name a country"):
        country_of("Answer the following question:")


def test_option_ids_are_resolved_in_context_for_all_eleven_letters(
    survey_tokenizer: TokenizerFactory,
) -> None:
    tokenizer = survey_tokenizer(bos=True)

    ids = resolve_option_token_ids(tokenizer)

    assert set(ids) == set(OPTION_LETTERS)
    assert len(set(ids.values())) == len(OPTION_LETTERS)
    assert ids["A"] == tokenizer(ANSWER_PREFIX + "A", add_special_tokens=False)["input_ids"][-1]


def test_a_tokenizer_that_merges_the_parenthesis_with_the_letter_fails_loudly(
    survey_tokenizer: TokenizerFactory,
) -> None:
    with pytest.raises(OptionTokenError, match="prefix tokens change"):
        resolve_option_token_ids(survey_tokenizer(merge_paren=True))


def test_a_letter_that_needs_two_tokens_fails_loudly(survey_tokenizer: TokenizerFactory) -> None:
    with pytest.raises(OptionTokenError, match="'X' is multi-token"):
        resolve_option_token_ids(survey_tokenizer(split_letter="X"))


def test_two_letters_sharing_one_token_id_fail_loudly(survey_tokenizer: TokenizerFactory) -> None:
    with pytest.raises(OptionTokenError, match="collide"):
        resolve_option_token_ids(survey_tokenizer(collide_letters=True))


def test_encode_returns_aligned_columns_and_keeps_the_bos_token(
    survey_tokenizer: TokenizerFactory, raw_survey_record: RecordFactory
) -> None:
    encoder = PromptEncoder.build(survey_tokenizer(bos=True), max_length=256)
    record = parse_record(raw_survey_record())

    encoded = encoder.encode(record)

    assert encoded["input_ids"][0] == 2
    assert encoded["input_ids"][-1] == encoder.answer_token_id
    assert encoded["option_token_ids"] == [encoder.letter_ids[letter] for letter in "ABC"]
    assert encoded["target_dist"] == pytest.approx([0.30, 0.60, 0.10])


def test_encode_refuses_prompts_above_max_length_instead_of_truncating(
    survey_tokenizer: TokenizerFactory, raw_survey_record: RecordFactory
) -> None:
    encoder = PromptEncoder.build(survey_tokenizer(), max_length=8)

    with pytest.raises(RecordError, match="above max_length 8"):
        encoder.encode(parse_record(raw_survey_record()))


def test_encode_refuses_a_prompt_that_does_not_end_at_the_answer_slot(
    survey_tokenizer: TokenizerFactory, raw_survey_record: RecordFactory
) -> None:
    encoder = PromptEncoder.build(survey_tokenizer(), max_length=256)
    broken = PromptEncoder(
        encoder.tokenizer, encoder.letter_ids, encoder.answer_token_id + 1, encoder.max_length
    )

    with pytest.raises(RecordError, match="does not end with the answer prefix token"):
        broken.encode(parse_record(raw_survey_record()))


def test_the_dataset_has_one_row_per_record_with_ragged_columns(
    survey_tokenizer: TokenizerFactory, raw_survey_record: RecordFactory
) -> None:
    encoder = PromptEncoder.build(survey_tokenizer(), max_length=256)
    records = [
        parse_record(raw_survey_record()),
        parse_record(
            raw_survey_record(
                record_id="57", options=("(A)Yes", "(B)No"), dist={"A": 40.0, "B": 60.0}
            )
        ),
    ]

    dataset = encoder.dataset(records)

    assert dataset.num_rows == 2
    assert dataset.column_names == ["input_ids", "option_token_ids", "target_dist"]
    assert len(dataset[0]["option_token_ids"]) == 3
    assert len(dataset[1]["option_token_ids"]) == 2


def test_the_collator_right_pads_and_marks_the_answer_index() -> None:
    collator = DistributionalCollator(pad_token_id=99)
    features = [
        {"input_ids": [5, 6, 7], "option_token_ids": [11, 12], "target_dist": [0.25, 0.75]},
        {"input_ids": [8, 9], "option_token_ids": [11, 12, 13], "target_dist": [0.5, 0.5, 0.0]},
    ]

    batch = collator(features)

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 9, 99]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["answer_index"].tolist() == [2, 1]
    assert batch["option_token_ids"].shape == (2, MAX_OPTIONS)
    assert batch["option_token_ids"][0].tolist() == [11, 12] + [0] * (MAX_OPTIONS - 2)
    assert batch["target_dist"].shape == (2, MAX_OPTIONS)
    assert batch["target_dist"][0].tolist() == pytest.approx(
        [0.25, 0.75] + [TARGET_PAD] * (MAX_OPTIONS - 2)
    )
    assert batch["target_dist"][1, 2].item() == 0.0
    assert batch["target_dist"][1, 3].item() == TARGET_PAD


def test_the_collator_emits_float32_targets_even_for_integral_values() -> None:
    batch = DistributionalCollator(pad_token_id=0)(
        [{"input_ids": [1], "option_token_ids": [3, 4], "target_dist": [1, 0]}]
    )

    assert batch["target_dist"].dtype is torch.float32
    assert batch["input_ids"].dtype is torch.long


def test_the_collator_refuses_more_options_than_it_is_sized_for() -> None:
    with pytest.raises(ValueError, match="more than the 2"):
        DistributionalCollator(pad_token_id=0, max_options=2)(
            [{"input_ids": [1], "option_token_ids": [3, 4, 5], "target_dist": [0.5, 0.5, 0.0]}]
        )


def test_the_pad_id_falls_back_to_eos_and_fails_without_either(
    survey_tokenizer: TokenizerFactory,
) -> None:
    assert pad_token_id_for(survey_tokenizer(pad_token_id=7)) == 7
    assert pad_token_id_for(survey_tokenizer(pad_token_id=None, eos_token_id=3)) == 3
    with pytest.raises(ValueError, match="neither a pad nor an eos"):
        pad_token_id_for(survey_tokenizer(pad_token_id=None, eos_token_id=None))


def _mixed_records(raw_survey_record: RecordFactory) -> list[Any]:
    return [
        parse_record(raw_survey_record(record_id=record_id, country=country))
        for record_id, country in (
            ("46", "Germany"),
            ("57", "Germany"),
            ("171", "Germany"),
            ("240", "Germany"),
            ("172", "Germany"),
            ("1", "Germany"),
            ("46", "Mexico"),
        )
    ]


def test_the_filter_drops_exactly_the_four_evaluation_items_and_counts_them(
    raw_survey_record: RecordFactory,
) -> None:
    kept, dropped = filter_evaluation_items(_mixed_records(raw_survey_record))

    assert [record.id for record in kept] == ["172", "1"]
    assert dropped == {"46": 2, "57": 1, "171": 1, "240": 1}


def test_count_items_reports_zero_for_absent_items(raw_survey_record: RecordFactory) -> None:
    counts = count_items([parse_record(raw_survey_record(record_id="1"))])

    assert counts == {"46": 0, "57": 0, "171": 0, "240": 0}


def test_dedupe_keeps_the_first_row_per_country_and_question(
    raw_survey_record: RecordFactory,
) -> None:
    records = [
        parse_record(raw_survey_record(record_id="46", country="Nigeria", data_type="test_2")),
        parse_record(raw_survey_record(record_id="46", country="Nigeria", data_type="test_5")),
        parse_record(raw_survey_record(record_id="46", country="Morocco", data_type="test_2")),
    ]

    unique = dedupe_records(records)

    assert [(record.country, record.data_type) for record in unique] == [
        ("Nigeria", "test_2"),
        ("Morocco", "test_2"),
    ]
    assert split_summary(records) == {
        "rows": 3,
        "countries": 2,
        "questions": 1,
        "duplicate_pairs": 1,
    }


def test_the_evaluation_items_are_the_four_items_the_sibling_paper_scores() -> None:
    assert EVALUATION_ITEMS == {
        "46": "happiness",
        "57": "social trust",
        "171": "religious attendance",
        "240": "left-right self-placement",
    }


@needs_culturellm
def test_the_culturellm_seed_set_contains_none_of_the_evaluation_items() -> None:
    header = (CULTURELLM_DATA_DIR / "Spanish" / "Mexico.csv").read_text().splitlines()[0]
    seed_items = {column[1:] for column in header.split(",") if column.startswith("Q")}

    assert seed_items.isdisjoint(EVALUATION_ITEMS)


def _fake_clone(root: Path, raw_survey_record: RecordFactory) -> Path:
    clone = root / "SimLLMCultureDist"
    (clone / "dataset").mkdir(parents=True)
    rows = {
        "train": [
            raw_survey_record(record_id="46", country="Germany"),
            raw_survey_record(record_id="1", country="Germany"),
        ],
        "valid": [raw_survey_record(record_id="171", country="Germany", data_type="valid")],
        "test": [
            raw_survey_record(record_id="46", country="Mexico", data_type="test_5"),
            raw_survey_record(record_id="46", country="Mexico", data_type="test_2"),
        ],
    }
    for split, name in SPLIT_FILES.items():
        (clone / "dataset" / name).write_text(json.dumps(rows[split]))
    return clone


def test_copy_source_data_copies_once_and_describes_every_split(
    tmp_path: Path, raw_survey_record: RecordFactory
) -> None:
    clone = _fake_clone(tmp_path, raw_survey_record)
    raw_dir = tmp_path / "raw"

    first = copy_source_data(clone, raw_dir, revision=lambda _: SOURCE_COMMIT)
    second = copy_source_data(clone, raw_dir, revision=lambda _: SOURCE_COMMIT)

    assert first["copied_files"] == 3
    assert second["copied_files"] == 0
    assert first["commit"] == SOURCE_COMMIT
    assert first["files"]["sft_wvs_train.json"]["rows"] == 2
    assert first["files"]["sft_wvs_train.json"]["evaluation_item_rows"]["46"] == 1
    assert first["files"]["sft_wvs_test.json"]["duplicate_pairs"] == 1
    assert len(load_split("valid", raw_dir)) == 1


def test_copy_source_data_fails_on_a_clone_at_another_commit(
    tmp_path: Path, raw_survey_record: RecordFactory
) -> None:
    clone = _fake_clone(tmp_path, raw_survey_record)

    with pytest.raises(ValueError, match="this track pins"):
        copy_source_data(clone, tmp_path / "raw", revision=lambda _: "deadbeef")


def test_copy_source_data_fails_when_a_split_is_missing(
    tmp_path: Path, raw_survey_record: RecordFactory
) -> None:
    clone = _fake_clone(tmp_path, raw_survey_record)
    (clone / "dataset" / "sft_wvs_valid.json").unlink()

    with pytest.raises(FileNotFoundError, match="missing from the reference clone"):
        copy_source_data(clone, tmp_path / "raw", revision=lambda _: SOURCE_COMMIT)


def test_write_provenance_records_commit_checksums_counts_and_both_attributions(
    tmp_path: Path, raw_survey_record: RecordFactory
) -> None:
    clone = _fake_clone(tmp_path, raw_survey_record)
    raw_dir = tmp_path / "raw"
    provenance = copy_source_data(clone, raw_dir, revision=lambda _: SOURCE_COMMIT)

    path = write_provenance(raw_dir, provenance)
    written = json.loads(path.read_text())

    assert path == raw_dir / "provenance.json"
    assert written["commit"] == SOURCE_COMMIT
    assert len(written["files"]["sft_wvs_train.json"]["sha256"]) == 64
    assert written["evaluation_items"] == EVALUATION_ITEMS
    assert "World Values Survey" in written["attribution"]["data"]
    assert "2502.07068" in written["attribution"]["method"]
    assert not path.with_name("provenance.json.part").exists()


def test_load_split_rejects_unknown_splits_and_points_at_prepare(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown split"):
        load_split("dev", tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare"):
        load_split("train", tmp_path)
