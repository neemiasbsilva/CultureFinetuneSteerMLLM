"""Pin the SubPOP record contract, the QA steering prompt, the split and the pinned fetch."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.data.distributional_data import (
    DistributionalCollator,
    OptionTokenError,
    RecordError,
)
from src.data.subpop_data import (
    ANSWER_STUB,
    DATASET_FILES,
    DATASET_LICENSE,
    DATASET_REPO_ID,
    DATASET_REVISION,
    MAX_OPTIONS,
    OPTION_LETTERS,
    SOURCE_COMMIT,
    STEERING_FILENAME,
    STEERING_SOURCE,
    SteeringPrompt,
    SubpopEncoder,
    build_prompt,
    fetch_source_data,
    filter_related_items,
    format_mcq,
    load_records,
    load_steering,
    parse_record,
    related_qkeys,
    split_by_question,
    split_summary,
)

RecordFactory = Callable[..., dict[str, Any]]
TokenizerFactory = Callable[..., Any]

UPSTREAM_QA_PROMPT = (
    "Question: In general, would you describe your political views as\n"
    "A. Very conservative\n"
    "B. Conservative\n"
    "C. Moderate\n"
    "D. Liberal\n"
    "E. Very liberal\n"
    "\n"
    "Answer: D. Liberal\n"
    "\n"
    "Answer the following question keeping in mind your previous answers.\n"
    "Question: Please indicate whether the following is a major reason, a minor reason, "
    "or not a reason why you own a gun. As part of a gun collection\n"
    "A. Major reason\n"
    "B. Minor reason\n"
    "C. Not a reason\n"
    "D. Refused\n"
    "Answer as a choice between A.,B.,C.,D.\n"
    "\n"
    "Answer:"
)


def _steering(entries: list[dict[str, str]], raw_dir: Path) -> dict[str, SteeringPrompt]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / STEERING_FILENAME).write_text(json.dumps(entries), encoding="utf-8")
    return load_steering(raw_dir)


def test_the_pinned_constants_name_ten_spaced_letters_and_full_revisions() -> None:
    assert OPTION_LETTERS == (" A", " B", " C", " D", " E", " F", " G", " H", " I", " J")
    assert MAX_OPTIONS == 10
    assert ANSWER_STUB == "Answer:"
    assert re.fullmatch(r"[0-9a-f]{40}", DATASET_REVISION)
    assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT)
    assert set(DATASET_FILES) == {"train", "eval"}


def test_a_record_keeps_the_refusal_rate_as_the_last_slot_of_its_target(
    raw_subpop_record: RecordFactory,
) -> None:
    record = parse_record(raw_subpop_record())

    assert record.qkey == "REASONGUND_W26"
    assert (record.attribute, record.group) == ("POLIDEOLOGY", "Liberal")
    assert record.responses == (0.1, 0.3, 0.6)
    assert record.target == (0.1, 0.3, 0.6, 0.02)
    assert record.ordinal == (1.0, 1.0, 1.0)


def test_a_single_substantive_option_is_a_valid_record(raw_subpop_record: RecordFactory) -> None:
    record = parse_record(
        raw_subpop_record(options=("Did not refuse", "Refused"), responses=(1.0,))
    )

    assert record.target == (1.0, 0.02)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"options": ("Yes", "No", "Maybe", "Other", "Refused")},
            "last option must be the refusal",
        ),
        ({"responses": (0.5, 0.5)}, "last option must be the refusal"),
        ({"responses": (-0.1, 0.5, 0.6)}, "negative response share"),
        ({"responses": (0.1, 0.3, 0.5)}, "not 1"),
        ({"refusal_rate": 1.5}, r"outside \[0, 1\]"),
        ({"ordinal": (1.0, 2.0)}, "ordinal values"),
        (
            {
                "options": tuple(f"option {index}" for index in range(11)),
                "responses": tuple([0.1] * 10),
            },
            "more than the 10 letters",
        ),
    ],
)
def test_a_malformed_record_is_refused(
    raw_subpop_record: RecordFactory, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(RecordError, match=message):
        parse_record(raw_subpop_record(**overrides))


def test_format_mcq_letters_the_options_and_ends_on_the_answer_stub() -> None:
    assert format_mcq(" Do you agree? ", [" Yes ", "No"]) == (
        "Question: Do you agree?\nA. Yes\nB. No\n\nAnswer:"
    )
    assert format_mcq("Do you agree?", ["Yes", "No", "Refused"], answer_forcing=True) == (
        "Question: Do you agree?\nA. Yes\nB. No\nC. Refused\n"
        "Answer as a choice between A.,B.,C.\n\nAnswer:"
    )


def test_the_prompt_is_the_upstream_qa_steering_string(
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    tmp_path: Path,
) -> None:
    steering = _steering(subpop_steering_entries, tmp_path)

    prompt = build_prompt(parse_record(raw_subpop_record()), steering)

    assert prompt == UPSTREAM_QA_PROMPT


def test_a_group_the_steering_prompt_does_not_offer_is_refused(
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    tmp_path: Path,
) -> None:
    steering = _steering(subpop_steering_entries, tmp_path)

    with pytest.raises(RecordError, match="not a steering option"):
        build_prompt(parse_record(raw_subpop_record(group="Libertarian")), steering)
    with pytest.raises(RecordError, match="no steering prompt"):
        build_prompt(parse_record(raw_subpop_record(attribute="AGE", group="18-29")), steering)


def test_load_steering_reads_the_upstream_python_literal_options(
    subpop_steering_entries: list[dict[str, str]], tmp_path: Path
) -> None:
    steering = _steering(subpop_steering_entries, tmp_path)

    assert set(steering) == {"POLIDEOLOGY", "SEX"}
    assert steering["SEX"].options == ("Male", "Female")
    assert steering["SEX"].question == "What is the sex that you were assigned at birth?"


def test_missing_tables_point_at_the_prepare_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"subpop_data\.py prepare"):
        load_steering(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"subpop_data\.py prepare"):
        load_records("train", tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        load_records("test", tmp_path)


def test_load_records_reads_one_record_per_jsonl_line(
    raw_subpop_record: RecordFactory, tmp_path: Path
) -> None:
    rows = [raw_subpop_record(qkey="Q1_W1"), raw_subpop_record(qkey="Q2_W1", group="Moderate")]
    (tmp_path / DATASET_FILES["train"]).write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8"
    )

    records = load_records("train", tmp_path)

    assert [record.qkey for record in records] == ["Q1_W1", "Q2_W1"]
    assert split_summary(records) == {"rows": 2, "questions": 2, "groups": 2, "attributes": 1}


def test_the_encoder_scores_all_ten_letters_and_zero_pads_the_target(
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    survey_tokenizer: TokenizerFactory,
    tmp_path: Path,
) -> None:
    tokenizer = survey_tokenizer()
    encoder = SubpopEncoder.build(
        tokenizer, _steering(subpop_steering_entries, tmp_path), max_length=256
    )

    encoded = encoder.encode(parse_record(raw_subpop_record()))

    assert list(encoder.letter_ids) == list(OPTION_LETTERS)
    assert len(set(encoder.letter_ids.values())) == MAX_OPTIONS
    assert encoded["option_token_ids"] == [encoder.letter_ids[letter] for letter in OPTION_LETTERS]
    assert encoded["target_dist"] == [0.1, 0.3, 0.6, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert encoded["input_ids"][-1] == encoder.answer_token_id
    assert encoder.answer_token_id == tokenizer(":", add_special_tokens=False)["input_ids"][0]


def test_the_encoder_keeps_a_leading_special_token(
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    survey_tokenizer: TokenizerFactory,
    tmp_path: Path,
) -> None:
    encoder = SubpopEncoder.build(
        survey_tokenizer(bos=True), _steering(subpop_steering_entries, tmp_path), max_length=256
    )

    encoded = encoder.encode(parse_record(raw_subpop_record()))

    assert encoded["input_ids"][0] == 2
    assert encoded["input_ids"][-1] == encoder.answer_token_id


def test_the_encoder_refuses_to_truncate(
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    survey_tokenizer: TokenizerFactory,
    tmp_path: Path,
) -> None:
    encoder = SubpopEncoder.build(
        survey_tokenizer(), _steering(subpop_steering_entries, tmp_path), max_length=16
    )

    with pytest.raises(RecordError, match=r"raise training\.max_seq_len instead of truncating"):
        encoder.encode(parse_record(raw_subpop_record()))


def test_the_encoder_refuses_letters_that_are_not_one_distinct_token_each(
    subpop_steering_entries: list[dict[str, str]],
    survey_tokenizer: TokenizerFactory,
    tmp_path: Path,
) -> None:
    steering = _steering(subpop_steering_entries, tmp_path)

    with pytest.raises(OptionTokenError, match="collide"):
        SubpopEncoder.build(survey_tokenizer(collide_letters=True), steering, max_length=256)
    with pytest.raises(OptionTokenError, match="multi-token"):
        SubpopEncoder.build(survey_tokenizer(split_letter="C"), steering, max_length=256)


def test_a_collated_batch_treats_every_letter_slot_as_valid(
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    survey_tokenizer: TokenizerFactory,
    tmp_path: Path,
) -> None:
    encoder = SubpopEncoder.build(
        survey_tokenizer(), _steering(subpop_steering_entries, tmp_path), max_length=256
    )
    records = [
        parse_record(raw_subpop_record()),
        parse_record(
            raw_subpop_record(
                attribute="SEX", group="Female", options=("Yes", "Refused"), responses=(1.0,)
            )
        ),
    ]

    dataset = encoder.dataset(records)
    batch = DistributionalCollator(pad_token_id=0, max_options=MAX_OPTIONS)(list(dataset))

    assert dataset.column_names == ["input_ids", "option_token_ids", "target_dist"]
    assert batch["option_token_ids"].shape == (2, MAX_OPTIONS)
    assert batch["target_dist"].shape == (2, MAX_OPTIONS)
    assert bool((batch["target_dist"] >= 0).all())
    assert batch["target_dist"][1].tolist() == pytest.approx([1.0, 0.02] + [0.0] * 8)
    rows = range(2)
    assert [int(batch["input_ids"][row, batch["answer_index"][row]]) for row in rows] == [
        encoder.answer_token_id
    ] * 2


def test_the_split_holds_out_whole_questions_and_is_deterministic(
    raw_subpop_record: RecordFactory,
) -> None:
    records = [
        parse_record(raw_subpop_record(qkey=f"Q{index}_W1", group=group))
        for index in range(20)
        for group in ("Liberal", "Moderate", "Conservative")
    ]

    train, valid = split_by_question(records, val_ratio=0.1, seed=42)
    again_train, again_valid = split_by_question(list(reversed(records)), val_ratio=0.1, seed=42)
    other_train, _ = split_by_question(records, val_ratio=0.1, seed=7)

    train_keys = {record.qkey for record in train}
    valid_keys = {record.qkey for record in valid}
    assert len(valid_keys) == 2
    assert len(train_keys) == 18
    assert train_keys.isdisjoint(valid_keys)
    assert len(train) == 54
    assert len(valid) == 6
    assert {record.qkey for record in again_valid} == valid_keys
    assert {record.qkey for record in again_train} == train_keys
    assert {record.qkey for record in other_train} != train_keys


def test_the_split_keeps_everything_for_training_at_ratio_zero_and_rejects_bad_ratios(
    raw_subpop_record: RecordFactory,
) -> None:
    records = [parse_record(raw_subpop_record(qkey=f"Q{index}_W1")) for index in range(5)]

    train, valid = split_by_question(records, val_ratio=0.0)

    assert len(train) == 5
    assert valid == []
    with pytest.raises(ValueError, match="val_ratio"):
        split_by_question(records, val_ratio=1.0)


def test_the_related_item_audit_reads_the_options_as_well_as_the_question(
    raw_subpop_record: RecordFactory,
) -> None:
    trust = raw_subpop_record(
        qkey="SOCTRUST_W91",
        question="Which statement comes closer to your own view, even if neither is exactly right?",
        options=(
            "Most people can be trusted",
            "You can’t be too careful in dealing with people",
            "Refused",
        ),
        responses=(0.4, 0.6),
    )
    attend = raw_subpop_record(
        qkey="ATTENDPERSON2_W117",
        question="In general, how often do you attend religious services in person?",
    )
    records = [
        parse_record(trust),
        parse_record({**trust, "group": "Moderate"}),
        parse_record(attend),
        parse_record(raw_subpop_record()),
    ]

    related = related_qkeys(records)
    kept, dropped = filter_related_items(records)

    assert related["social trust"] == ["SOCTRUST_W91"]
    assert related["religious attendance"] == ["ATTENDPERSON2_W117"]
    assert related["happiness"] == []
    assert related["left-right self-placement"] == []
    assert dropped == 3
    assert [record.qkey for record in kept] == ["REASONGUND_W26"]


def _reference_clone(tmp_path: Path, entries: list[dict[str, str]]) -> Path:
    clone = tmp_path / "subpop"
    (clone / STEERING_SOURCE).parent.mkdir(parents=True)
    (clone / STEERING_SOURCE).write_text(json.dumps(entries), encoding="utf-8")
    return clone


class _FakeHub:
    def __init__(self, cache: Path, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.calls: list[dict[str, Any]] = []
        cache.mkdir()
        self.cache = cache
        for name, records in rows.items():
            (cache / name).write_text(
                "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
            )

    def __call__(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return str(self.cache / kwargs["filename"])


def test_fetch_pins_the_revision_copies_once_and_records_provenance(
    raw_subpop_record: RecordFactory,
    subpop_steering_entries: list[dict[str, str]],
    tmp_path: Path,
) -> None:
    clone = _reference_clone(tmp_path, subpop_steering_entries)
    hub = _FakeHub(
        tmp_path / "cache",
        {
            DATASET_FILES["train"]: [raw_subpop_record(), raw_subpop_record(group="Moderate")],
            DATASET_FILES["eval"]: [raw_subpop_record(qkey="happy", ordinal=(1.0, 2.0, 3.0))],
        },
    )
    raw_dir = tmp_path / "raw"

    first = fetch_source_data(clone, raw_dir, revision=lambda _: SOURCE_COMMIT, download=hub)
    second = fetch_source_data(clone, raw_dir, revision=lambda _: SOURCE_COMMIT, download=hub)

    assert first["copied_files"] == 3
    assert second["copied_files"] == 0
    assert all(
        call
        == {
            "repo_id": DATASET_REPO_ID,
            "filename": call["filename"],
            "repo_type": "dataset",
            "revision": DATASET_REVISION,
        }
        for call in hub.calls
    )
    assert first["dataset"] == {
        "repo": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "license": DATASET_LICENSE,
    }
    assert first["code"]["commit"] == SOURCE_COMMIT
    train_file = first["files"][DATASET_FILES["train"]]
    assert train_file["rows"] == 2
    assert train_file["questions"] == 1
    assert train_file["groups"] == 2
    assert len(train_file["sha256"]) == 64
    assert set(train_file["related_questions"]) == {
        "happiness",
        "social trust",
        "religious attendance",
        "left-right self-placement",
    }
    assert first["split"] == {"by": "question", "val_ratio": 0.1, "seed": 42}
    assert "Pew Research Center" in first["attribution"]["data"]
    assert "2502.16761" in first["attribution"]["method"]
    assert set(load_steering(raw_dir)) == {"POLIDEOLOGY", "SEX"}


def test_fetch_refuses_a_clone_at_another_commit_or_without_the_steering_file(
    subpop_steering_entries: list[dict[str, str]], tmp_path: Path
) -> None:
    clone = _reference_clone(tmp_path, subpop_steering_entries)
    hub = _FakeHub(tmp_path / "cache", {})

    with pytest.raises(ValueError, match="this track pins"):
        fetch_source_data(clone, tmp_path / "raw", revision=lambda _: "0" * 40, download=hub)

    (clone / STEERING_SOURCE).unlink()
    with pytest.raises(FileNotFoundError, match="missing from the reference clone"):
        fetch_source_data(clone, tmp_path / "raw", revision=lambda _: SOURCE_COMMIT, download=hub)
    assert hub.calls == []
