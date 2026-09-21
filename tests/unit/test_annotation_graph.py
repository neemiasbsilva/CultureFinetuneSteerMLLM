"""Pin the Stage-3 annotation graph contract."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import src.annotation.graph as graph_module
from src.annotation.config import (
    ANNOTATION_SYSTEM_PROMPT,
    ANNOTATION_USER_PROMPT,
    AnnotatorSettings,
)
from src.annotation.graph import build_annotation_graph
from src.annotation.nodes.assembler import assembler_node, build_messages
from src.annotation.nodes.image_loader import ImageCache, image_loader_node

EXPECTED_NODES = {"__start__", "image_loader", "assembler", "annotator"}
EXPECTED_EDGES = {
    ("__start__", "image_loader"),
    ("image_loader", "assembler"),
    ("assembler", "annotator"),
    ("annotator", "__end__"),
}


class _RecordingAnnotatorFactory:
    def __init__(self) -> None:
        self.settings_seen: list[AnnotatorSettings] = []
        self.states_seen: list[dict[str, Any]] = []

    def __call__(self, settings: AnnotatorSettings) -> Callable[[dict[str, Any]], dict[str, Any]]:
        self.settings_seen.append(settings)

        def _annotator_node(state: dict[str, Any]) -> dict[str, Any]:
            self.states_seen.append(dict(state))
            return {**state, "raw_output": "recorded", "parse_strategy": "direct_json"}

        return _annotator_node


class _OpenRecorder:
    def __init__(self, real_open: Callable[..., Any], suffix: str) -> None:
        self._real_open = real_open
        self._suffix = suffix
        self.paths: list[str] = []

    def __call__(self, file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, Path)) and str(file).endswith(self._suffix):
            self.paths.append(str(file))
        return self._real_open(file, *args, **kwargs)


def _write_jpeg(path: Path, color: tuple[int, int, int] = (12, 200, 90)) -> bytes:
    image = Image.new("RGB", (8, 6), color)
    image.save(path, format="JPEG")
    return path.read_bytes()


def _write_png(path: Path) -> bytes:
    image = Image.new("RGB", (4, 4), (255, 0, 0))
    image.save(path, format="PNG")
    return path.read_bytes()


def _base_state(image_path: Path) -> dict[str, Any]:
    return {
        "image_id": "img-1",
        "image_path": str(image_path),
        "image_b64": None,
        "culture": "arabic",
        "condition": "wvs_cultural",
        "model_name": "qwen3_5_2b",
        "run_id": "r01",
        "run_index": 1,
        "n_runs": 5,
        "seed": 4242,
        "generation_seed": 4242,
        "system_prompt": "",
        "checkpoint_identity": None,
        "checkpoint_path": None,
        "adapter_backend": None,
        "raw_output": None,
        "parsed": None,
        "parse_strategy": "not_attempted",
        "parse_retries": 0,
        "error": None,
        "ground_truth_sentiment": 3,
    }


def test_assembler_installs_the_neutral_system_prompt_verbatim(tmp_path: Path) -> None:
    state = _base_state(tmp_path / "img-1.jpg")
    result = assembler_node(state)
    assert result["system_prompt"] == ANNOTATION_SYSTEM_PROMPT


@pytest.mark.parametrize(
    "incoming_prompt",
    ["", "You are a 34-year-old man from Cairo.", ANNOTATION_SYSTEM_PROMPT],
)
def test_assembler_overwrites_any_persona_left_in_the_incoming_prompt(
    tmp_path: Path, incoming_prompt: str
) -> None:
    state = _base_state(tmp_path / "img-1.jpg")
    state["system_prompt"] = incoming_prompt
    assert assembler_node(state)["system_prompt"] == ANNOTATION_SYSTEM_PROMPT


def test_assembler_preserves_every_other_state_key_untouched(tmp_path: Path) -> None:
    state = _base_state(tmp_path / "img-1.jpg")
    result = assembler_node(state)
    assert set(result) == set(state)
    for key, value in state.items():
        if key != "system_prompt":
            assert result[key] == value


def test_assembler_does_not_mutate_the_state_it_was_given(tmp_path: Path) -> None:
    state = _base_state(tmp_path / "img-1.jpg")
    original = dict(state)
    assembler_node(state)
    assert state == original


def test_built_messages_carry_system_then_user_roles(tmp_path: Path) -> None:
    state = assembler_node(_base_state(tmp_path / "img-1.jpg"))
    state["image_b64"] = "QUJD"
    messages = build_messages(state)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == ANNOTATION_SYSTEM_PROMPT


def test_built_messages_embed_the_image_as_a_base64_data_url_beside_the_user_prompt(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "img-1.jpg"
    raw_bytes = _write_jpeg(image_path)
    state = assembler_node(image_loader_node(_base_state(image_path)))
    messages = build_messages(state)
    image_part, text_part = messages[1]["content"]
    assert image_part["type"] == "image_url"
    assert text_part == {"type": "text", "text": ANNOTATION_USER_PROMPT}
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw_bytes


def test_built_messages_fall_back_to_an_empty_payload_when_the_image_key_is_absent(
    tmp_path: Path,
) -> None:
    state = assembler_node(_base_state(tmp_path / "img-1.jpg"))
    del state["image_b64"]
    image_part = build_messages(state)[1]["content"][0]
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64,"


def test_built_messages_stringify_a_none_payload_into_the_data_url(tmp_path: Path) -> None:
    state = assembler_node(_base_state(tmp_path / "img-1.jpg"))
    assert state["image_b64"] is None
    image_part = build_messages(state)[1]["content"][0]
    assert image_part["image_url"]["url"] == "data:image/jpeg;base64,None"


@pytest.mark.parametrize("writer", [_write_jpeg, _write_png])
def test_image_loader_base64_encodes_the_exact_file_bytes(
    tmp_path: Path, writer: Callable[[Path], bytes]
) -> None:
    image_path = tmp_path / "img-1.jpg"
    raw_bytes = writer(image_path)
    result = image_loader_node(_base_state(image_path))
    assert base64.b64decode(result["image_b64"]) == raw_bytes
    assert result["error"] is None


def test_image_loader_reports_a_missing_file_instead_of_raising(tmp_path: Path) -> None:
    missing = tmp_path / "absent.jpg"
    result = image_loader_node(_base_state(missing))
    assert result["error"] == f"Image not found: {missing}"
    assert result["image_b64"] is None


def test_image_loader_re_reads_the_file_even_when_a_payload_is_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "img-1.jpg"
    _write_jpeg(image_path)
    state = _base_state(image_path)
    state["image_b64"] = "already-encoded"
    recorder = _OpenRecorder(open, ".jpg")
    monkeypatch.setattr("builtins.open", recorder)
    result = image_loader_node(state)
    monkeypatch.undo()
    assert recorder.paths == [str(image_path)]
    assert result["image_b64"] != "already-encoded"


def test_image_cache_reads_each_image_once_and_serves_repeats_from_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        image_id: _write_jpeg(tmp_path / f"{image_id}.jpg", color)
        for image_id, color in [("a", (10, 20, 30)), ("b", (200, 100, 50))]
    }
    recorder = _OpenRecorder(open, ".jpg")
    monkeypatch.setattr("builtins.open", recorder)
    cache = ImageCache(["a", "b"], str(tmp_path))
    reads_after_preload = list(recorder.paths)
    first = cache.get("a")
    second = cache.get("a")
    monkeypatch.undo()
    assert len(reads_after_preload) == 2
    assert recorder.paths == reads_after_preload
    assert first == second
    assert base64.b64decode(first) == payloads["a"]


def test_image_cache_skips_missing_ids_without_raising(tmp_path: Path) -> None:
    _write_jpeg(tmp_path / "present.jpg")
    cache = ImageCache(["present", "absent"], str(tmp_path))
    assert cache.get("absent") is None
    assert cache.get("present") is not None
    assert len(cache) == 1


def test_image_cache_only_recognises_the_jpg_suffix(tmp_path: Path) -> None:
    _write_png(tmp_path / "png-only.png")
    (tmp_path / "jpeg-only.jpeg").write_bytes(_write_jpeg(tmp_path / "scratch.jpg"))
    cache = ImageCache(["png-only", "jpeg-only"], str(tmp_path))
    assert cache.get("png-only") is None
    assert cache.get("jpeg-only") is None
    assert len(cache) == 0


def test_image_cache_and_image_loader_agree_on_the_encoding(tmp_path: Path) -> None:
    image_path = tmp_path / "img-1.jpg"
    _write_jpeg(image_path)
    cache = ImageCache(["img-1"], str(tmp_path))
    assert cache.get("img-1") == image_loader_node(_base_state(image_path))["image_b64"]


def test_build_annotation_graph_compiles_exactly_the_expected_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _RecordingAnnotatorFactory()
    monkeypatch.setattr(graph_module, "make_annotator_node", factory)
    settings = AnnotatorSettings()
    compiled = build_annotation_graph(settings)
    assert set(compiled.nodes) == EXPECTED_NODES
    assert factory.settings_seen == [settings]


def test_build_annotation_graph_wires_loader_then_assembler_then_annotator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_module, "make_annotator_node", _RecordingAnnotatorFactory())
    drawable = build_annotation_graph(AnnotatorSettings()).get_graph()
    assert {(edge.source, edge.target) for edge in drawable.edges} == EXPECTED_EDGES


def test_compiled_graph_hands_the_annotator_an_encoded_image_and_the_neutral_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "img-1.jpg"
    raw_bytes = _write_jpeg(image_path)
    factory = _RecordingAnnotatorFactory()
    monkeypatch.setattr(graph_module, "make_annotator_node", factory)
    compiled = build_annotation_graph(AnnotatorSettings())
    final_state = compiled.invoke(_base_state(image_path))
    assert len(factory.states_seen) == 1
    seen = factory.states_seen[0]
    assert seen["system_prompt"] == ANNOTATION_SYSTEM_PROMPT
    assert base64.b64decode(seen["image_b64"]) == raw_bytes
    assert seen["culture"] == "arabic"
    assert final_state["raw_output"] == "recorded"
    assert final_state["parse_strategy"] == "direct_json"


def test_compiled_graph_still_reaches_the_annotator_when_the_image_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _RecordingAnnotatorFactory()
    monkeypatch.setattr(graph_module, "make_annotator_node", factory)
    compiled = build_annotation_graph(AnnotatorSettings())
    missing = tmp_path / "absent.jpg"
    final_state = compiled.invoke(_base_state(missing))
    assert factory.states_seen[0]["error"] == f"Image not found: {missing}"
    assert final_state["system_prompt"] == ANNOTATION_SYSTEM_PROMPT


def test_annotation_prompts_inject_no_culture_or_persona() -> None:
    combined = f"{ANNOTATION_SYSTEM_PROMPT} {ANNOTATION_USER_PROMPT}".lower()
    for token in ["arabic", "brazilian", "english", "turkish", "persona", "years old"]:
        assert token not in combined
    assert "do not assume a viewer identity" in ANNOTATION_SYSTEM_PROMPT.lower()
