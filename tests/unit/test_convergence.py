"""Pin the per-image convergence contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.convergence import (
    DEFAULT_CONDITION_PAIRS,
    compute_condition_pair_agreement,
    compute_convergence,
    compute_label_jaccard,
    compute_sentiment_agreement,
    jaccard,
)

BOOTSTRAP_COLUMNS = [
    "caption_cosine_lo",
    "caption_cosine_hi",
    "tag_jaccard_lo",
    "tag_jaccard_hi",
    "sentiment_agreement_lo",
    "sentiment_agreement_hi",
]


def _annotation(
    image_id: str,
    condition: str,
    culture: str,
    sentiment: int,
    perceptions: list[str] | None = None,
    model_name: str = "qwen3-vl-8b",
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "culture": culture,
        "condition": condition,
        "image_id": image_id,
        "predicted_sentiment": sentiment,
        "predicted_perceptions": [] if perceptions is None else perceptions,
        "caption": f"caption for {image_id} under {condition}",
    }


def _row_for(frame: pd.DataFrame, condition: str, image_id: str) -> pd.Series:
    selected = frame[(frame["condition"] == condition) & (frame["image_id"] == image_id)]
    assert len(selected) == 1
    return selected.iloc[0]


def _paired_condition_frame(n_images: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    embeddings = []
    for index in range(n_images):
        image_id = str(index)
        rows.append(_annotation(image_id, "wvs_cultural", "arabic", index % 5, ["street"]))
        embeddings.append([1.0, float(index) / n_images])
        rows.append(_annotation(image_id, "inference_only", "arabic", (index + 1) % 5, ["avenue"]))
        embeddings.append([float(index) / n_images, 1.0])
    return pd.DataFrame(rows), np.asarray(embeddings, dtype=float)


def test_jaccard_of_two_empty_sets_is_one_by_convention() -> None:
    assert jaccard(set(), set()) == 1.0


@pytest.mark.parametrize(
    ("set_a", "set_b", "expected"),
    [
        ({"a", "b"}, {"c", "d"}, 0.0),
        ({"a"}, set(), 0.0),
        ({"a", "b"}, {"a", "b"}, 1.0),
        ({"a", "b"}, {"b", "c"}, 1 / 3),
        ({"a", "b", "c", "d"}, {"a", "b"}, 0.5),
        ({"a", "b", "c"}, {"b", "c", "d", "e"}, 0.4),
    ],
)
def test_jaccard_equals_intersection_over_union_for_nonempty_unions(
    set_a: set[str],
    set_b: set[str],
    expected: float,
) -> None:
    assert jaccard(set_a, set_b) == pytest.approx(expected)
    assert jaccard(set_b, set_a) == pytest.approx(expected)


def test_sentiment_agreement_is_one_for_an_image_annotated_exactly_once() -> None:
    frame = pd.DataFrame([_annotation("img-1", "wvs_cultural", "english", 3)])
    result = compute_sentiment_agreement(frame)
    row = _row_for(result, "wvs_cultural", "img-1")
    assert row["sentiment_agreement"] == 1.0
    assert row["majority_sentiment"] == 3
    assert row["n_models"] == 1


def test_sentiment_agreement_never_pairs_the_same_image_across_two_conditions() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 4),
            _annotation("img-1", "inference_only", "inference_only", 0),
        ]
    )
    result = compute_sentiment_agreement(frame)
    assert len(result) == 2
    for condition in ("wvs_cultural", "inference_only"):
        row = _row_for(result, condition, "img-1")
        assert row["sentiment_agreement"] == 1.0
        assert row["n_models"] == 1


def test_sentiment_agreement_is_the_fraction_of_agreeing_model_pairs_in_one_condition() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 2, model_name="a"),
            _annotation("img-1", "wvs_cultural", "arabic", 2, model_name="b"),
            _annotation("img-1", "wvs_cultural", "german", 3, model_name="c"),
            _annotation("img-2", "wvs_cultural", "english", 1, model_name="a"),
            _annotation("img-2", "wvs_cultural", "arabic", 4, model_name="b"),
        ]
    )
    result = compute_sentiment_agreement(frame)
    agreeing = _row_for(result, "wvs_cultural", "img-1")
    assert agreeing["sentiment_agreement"] == pytest.approx(1 / 3)
    assert agreeing["majority_sentiment"] == 2
    assert agreeing["n_models"] == 3
    disagreeing = _row_for(result, "wvs_cultural", "img-2")
    assert disagreeing["sentiment_agreement"] == 0.0
    assert disagreeing["n_models"] == 2


def test_sentiment_agreement_groups_on_image_alone_when_no_condition_column_exists() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 4),
            _annotation("img-1", "inference_only", "inference_only", 0),
        ]
    ).drop(columns=["condition"])
    result = compute_sentiment_agreement(frame)
    assert list(result.columns) == [
        "image_id",
        "sentiment_agreement",
        "majority_sentiment",
        "n_models",
    ]
    assert len(result) == 1
    assert result.iloc[0]["sentiment_agreement"] == 0.0
    assert result.iloc[0]["n_models"] == 2


def test_label_jaccard_lowercases_tags_and_treats_non_lists_as_empty_sets() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 2, ["Street", "Park"]),
            _annotation("img-1", "wvs_cultural", "arabic", 2, ["street"]),
            _annotation("img-2", "wvs_cultural", "english", 2, ["street"]),
            _annotation("img-2", "wvs_cultural", "arabic", 2, None),
        ]
    )
    frame.loc[3, "predicted_perceptions"] = None
    result = compute_label_jaccard(frame)
    assert _row_for(result, "wvs_cultural", "img-1")["label_jaccard"] == pytest.approx(0.5)
    assert _row_for(result, "wvs_cultural", "img-2")["label_jaccard"] == 0.0


def test_default_condition_pairs_declares_only_the_wvs_versus_base_contrast() -> None:
    assert DEFAULT_CONDITION_PAIRS == {"wvs_vs_base": ("wvs_cultural", "inference_only")}
    assert list(DEFAULT_CONDITION_PAIRS) == ["wvs_vs_base"]


def test_condition_pair_agreement_uses_the_default_pair_when_none_is_supplied() -> None:
    frame, embeddings = _paired_condition_frame(3)
    result = compute_condition_pair_agreement(frame, caption_embeddings=embeddings)
    assert list(result["comparison_family"]) == ["wvs_vs_base"]
    assert list(result["condition_a"]) == ["wvs_cultural"]
    assert list(result["condition_b"]) == ["inference_only"]
    assert result.iloc[0]["n_images"] == 3


def test_condition_pair_agreement_rejects_embeddings_that_are_not_row_aligned() -> None:
    frame, embeddings = _paired_condition_frame(3)
    with pytest.raises(ValueError, match="exactly one row per annotation row"):
        compute_condition_pair_agreement(frame, caption_embeddings=embeddings[:-1])


def test_condition_pair_agreement_skips_groups_missing_one_side_of_the_pair() -> None:
    frame, embeddings = _paired_condition_frame(2)
    keep = frame["condition"] == "wvs_cultural"
    result = compute_condition_pair_agreement(
        frame[keep].reset_index(drop=True),
        caption_embeddings=embeddings[keep.to_numpy()],
    )
    assert result.empty


def test_condition_pair_agreement_bootstrap_bounds_are_identical_across_two_calls() -> None:
    frame, embeddings = _paired_condition_frame(6)
    first = compute_condition_pair_agreement(
        frame, caption_embeddings=embeddings, n_bootstrap=20, seed=1234
    )
    second = compute_condition_pair_agreement(
        frame, caption_embeddings=embeddings, n_bootstrap=20, seed=1234
    )
    pd.testing.assert_frame_equal(first, second)
    for column in BOOTSTRAP_COLUMNS:
        assert first[column].tolist() == second[column].tolist()
    row = first.iloc[0]
    for metric in ("caption_cosine", "tag_jaccard", "sentiment_agreement"):
        assert row[f"{metric}_lo"] <= row[f"{metric}_hi"]


def test_condition_pair_agreement_bootstrap_bounds_respond_to_the_seed_argument() -> None:
    frame, embeddings = _paired_condition_frame(6)
    bounds = {
        seed: tuple(
            compute_condition_pair_agreement(
                frame, caption_embeddings=embeddings, n_bootstrap=20, seed=seed
            ).iloc[0][["caption_cosine_lo", "caption_cosine_hi"]]
        )
        for seed in (7, 99, 1234)
    }
    assert len(set(bounds.values())) == 3


def test_condition_pair_agreement_resamples_each_group_independently_of_its_twin() -> None:
    frame, embeddings = _paired_condition_frame(6)
    combined = pd.concat([frame, frame.assign(culture="german")], ignore_index=True)
    result = compute_condition_pair_agreement(
        combined,
        caption_embeddings=np.concatenate([embeddings, embeddings]),
        n_bootstrap=20,
        seed=1234,
    )
    arabic = result[result["culture"] == "arabic"].iloc[0]
    german = result[result["culture"] == "german"].iloc[0]
    assert arabic["caption_cosine"] == pytest.approx(german["caption_cosine"])
    assert arabic["caption_cosine_hi"] != german["caption_cosine_hi"]


def test_condition_pair_agreement_omits_interval_columns_without_bootstrap_replicates() -> None:
    frame, embeddings = _paired_condition_frame(4)
    result = compute_condition_pair_agreement(frame, caption_embeddings=embeddings)
    assert not set(BOOTSTRAP_COLUMNS) & set(result.columns)


def test_condition_pair_agreement_scores_each_model_culture_group_separately() -> None:
    frame, embeddings = _paired_condition_frame(2)
    other = frame.copy()
    other["culture"] = "german"
    combined = pd.concat([frame, other], ignore_index=True)
    stacked = np.concatenate([embeddings, embeddings])
    result = compute_condition_pair_agreement(combined, caption_embeddings=stacked)
    assert set(result["culture"]) == {"arabic", "german"}
    assert set(result["n_images"]) == {2}


def test_condition_pair_agreement_reports_exact_cosine_jaccard_and_sentiment_means() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 3, ["street", "park"]),
            _annotation("img-1", "inference_only", "english", 3, ["street"]),
            _annotation("img-2", "wvs_cultural", "english", 4, ["market"]),
            _annotation("img-2", "inference_only", "english", 1, ["market"]),
        ]
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.6, 0.8],
            [0.0, 2.0],
            [0.0, 5.0],
        ]
    )
    result = compute_condition_pair_agreement(frame, caption_embeddings=embeddings)
    row = result.iloc[0]
    assert row["caption_cosine"] == pytest.approx((0.6 + 1.0) / 2)
    assert row["tag_jaccard"] == pytest.approx((0.5 + 1.0) / 2)
    assert row["sentiment_agreement"] == pytest.approx(0.5)


def test_convergence_merges_all_dimensions_per_condition_and_image() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 3, ["Street"]),
            _annotation("img-1", "wvs_cultural", "arabic", 3, ["street", "park"]),
            _annotation("img-1", "inference_only", "english", 1, ["a"]),
            _annotation("img-1", "inference_only", "arabic", 4, ["b"]),
        ]
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.6, 0.8],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    result = compute_convergence(frame, embeddings)
    assert "justification_sim" not in result.columns
    wvs = _row_for(result, "wvs_cultural", "img-1")
    assert wvs["caption_sim"] == pytest.approx(0.6)
    assert wvs["sentiment_agreement"] == 1.0
    assert wvs["label_jaccard"] == pytest.approx(0.5)
    assert wvs["n_models"] == 2
    base = _row_for(result, "inference_only", "img-1")
    assert base["caption_sim"] == pytest.approx(1.0)
    assert base["sentiment_agreement"] == 0.0
    assert base["label_jaccard"] == 0.0


def test_convergence_keeps_singly_annotated_images_with_missing_caption_similarity() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 3, ["street"]),
            _annotation("img-1", "wvs_cultural", "arabic", 3, ["street"]),
            _annotation("img-2", "wvs_cultural", "english", 2, ["market"]),
        ]
    )
    embeddings = np.asarray([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    result = compute_convergence(frame, embeddings)
    assert set(result["image_id"]) == {"img-1", "img-2"}
    lonely = _row_for(result, "wvs_cultural", "img-2")
    assert np.isnan(lonely["caption_sim"])
    assert lonely["sentiment_agreement"] == 1.0
    assert lonely["n_models"] == 1


def test_convergence_raises_key_error_when_every_image_is_annotated_only_once() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 3, ["street"]),
            _annotation("img-2", "wvs_cultural", "english", 2, ["market"]),
        ]
    )
    with pytest.raises(KeyError, match="condition"):
        compute_convergence(frame, np.asarray([[1.0, 0.0], [0.0, 1.0]]))


def test_convergence_adds_justification_similarity_only_when_embeddings_are_given() -> None:
    frame = pd.DataFrame(
        [
            _annotation("img-1", "wvs_cultural", "english", 3, ["street"]),
            _annotation("img-1", "wvs_cultural", "arabic", 3, ["street"]),
        ]
    )
    caption_embeddings = np.asarray([[1.0, 0.0], [0.6, 0.8]])
    justification_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = compute_convergence(frame, caption_embeddings, justification_embeddings)
    row = _row_for(result, "wvs_cultural", "img-1")
    assert row["caption_sim"] == pytest.approx(0.6)
    assert row["justification_sim"] == pytest.approx(0.0)
