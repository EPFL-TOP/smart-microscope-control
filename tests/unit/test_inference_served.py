"""The model server's client: its codecs, its check, its detector, and ``load_model`` with a server set (inference.md §8.3 to §8.5).

Everything here runs on ``smc.testing.FakeModelServer`` around a ``StubDetector``: no
socket, no torch, no weights, no kit (FM-40). The wire on a real socket is
``test_inference_served_http.py``'s.
"""

from __future__ import annotations

import base64
import copy
import json
import pickle
from typing import Any

import numpy as np
import pytest

from smc.inference.detector import Detection
from smc.inference.served import (
    SCHEMA,
    ServedError,
    ServedModel,
    decode_answer,
    decode_frame,
    decode_listing,
    encode_answer,
    encode_frame,
    encode_listing,
    error_from_status,
    result_text,
)

# --- the codecs -------------------------------------------------------------------


def test_a_detection_crosses_the_wire_exactly() -> None:
    detections = [
        Detection(
            0.1 + 0.2, 1 / 3, 1e-7, 123456.789, "single", float(np.float32(0.93))
        ),
        Detection(0.0, 0.0, 5.0, 5.0, "débris", 1.0),
    ]

    answer = decode_answer(encode_answer("m", "i", "cuda", detections))

    assert list(answer.detections) == detections
    assert (answer.model, answer.identity, answer.device) == ("m", "i", "cuda")


def test_a_listing_crosses_the_wire_exactly() -> None:
    models = [
        ServedModel("a", "sha+pre", ("single", "doublet"), "cuda"),
        ServedModel("b", "sha2+pre2", ("x",), "cpu"),
    ]

    assert decode_listing(encode_listing(models)) == tuple(models)
    assert decode_listing(encode_listing([])) == ()


@pytest.mark.parametrize("schema", [2, 0, "1", True, 1.0, None], ids=repr)
def test_a_listing_or_an_answer_of_another_schema_is_malformed(schema: object) -> None:
    listing = json.dumps({"schema": schema, "models": []})
    answer = json.dumps(
        {
            "schema": schema,
            "model": "m",
            "identity": "i",
            "device": "cpu",
            "detections": [],
        }
    )

    for decode, text in ((decode_listing, listing), (decode_answer, answer)):
        with pytest.raises(ServedError) as caught:
            decode(text)
        assert caught.value.kind == "malformed"
        assert f"schema {SCHEMA}" in str(caught.value)


@pytest.mark.parametrize("text", ["", "not json", "[]", '"x"', "null", "{"])
def test_a_document_that_is_not_a_json_object_is_malformed(text: str) -> None:
    for decode in (decode_listing, decode_answer):
        with pytest.raises(ServedError) as caught:
            decode(text)
        assert caught.value.kind == "malformed"


@pytest.mark.parametrize(
    "value", ["NaN", "Infinity", "-Infinity", "1e999", "true", '"1.0"', "null"]
)
def test_a_number_that_is_not_finite_and_a_bool_are_malformed(value: str) -> None:
    for slot in range(4):
        box = ["1.0"] * 4
        box[slot] = value
        text = (
            f'{{"schema": 1, "model": "m", "identity": "i", "device": "d", '
            f'"detections": [[{", ".join(box)}, "single", 0.5]]}}'
        )
        with pytest.raises(ServedError) as caught:
            decode_answer(text)
        assert caught.value.kind == "malformed"
    score = (
        f'{{"schema": 1, "model": "m", "identity": "i", "device": "d", '
        f'"detections": [[1, 2, 3, 4, "single", {value}]]}}'
    )
    with pytest.raises(ServedError):
        decode_answer(score)


@pytest.mark.parametrize(
    "detection",
    [
        [1, 2, 3, 4, "single"],
        [1, 2, 3, 4, "single", 0.5, 9],
        {"row0": 1},
        "single",
        [1, 2, 3, 4, 5, 0.5],
        [1, 2, 3, 4, None, 0.5],
    ],
    ids=["five", "seven", "object", "string", "label-is-an-int", "label-is-null"],
)
def test_a_detection_that_is_not_six_values_is_malformed(detection: object) -> None:
    text = json.dumps(
        {
            "schema": 1,
            "model": "m",
            "identity": "i",
            "device": "d",
            "detections": [detection],
        }
    )

    with pytest.raises(ServedError) as caught:
        decode_answer(text)

    assert caught.value.kind == "malformed"


@pytest.mark.parametrize(
    "field", ["model", "identity", "device"], ids=lambda f: f"answer-{f}"
)
def test_an_answer_whose_field_is_not_a_string_is_malformed(field: str) -> None:
    document: dict[str, Any] = {
        "schema": 1,
        "model": "m",
        "identity": "i",
        "device": "d",
        "detections": [],
    }
    document[field] = 7

    with pytest.raises(ServedError) as caught:
        decode_answer(json.dumps(document))

    assert caught.value.kind == "malformed"


@pytest.mark.parametrize(
    "models",
    [
        "x",
        [1],
        [{"name": "a", "identity": "i", "classes": "single", "device": "d"}],
        [{"name": "a", "identity": "i", "classes": [1], "device": "d"}],
        [{"name": 1, "identity": "i", "classes": [], "device": "d"}],
        [{"identity": "i", "classes": [], "device": "d"}],
    ],
)
def test_a_listing_that_is_not_a_list_of_models_is_malformed(models: object) -> None:
    with pytest.raises(ServedError) as caught:
        decode_listing(json.dumps({"schema": 1, "models": models}))

    assert caught.value.kind == "malformed"


def test_a_nan_is_never_written() -> None:
    with pytest.raises(ValueError, match="Out of range"):
        encode_answer(
            "m", "i", "cpu", [Detection(0.0, 0.0, float("nan"), 1.0, "a", 1.0)]
        )


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.uint8])
def test_a_frame_crosses_the_wire_as_it_is(dtype: Any) -> None:
    frame = (np.arange(30 * 40).reshape(30, 40) % 251).astype(dtype)

    text = encode_frame(frame)

    assert text.isascii()
    got = decode_frame(text)
    assert got.dtype == frame.dtype
    assert np.array_equal(got, frame)


def test_a_frame_that_is_a_view_is_sent_as_its_own_pixels() -> None:
    frame = np.arange(30 * 40, dtype=np.uint16).reshape(30, 40)
    view = frame[::2, ::3]

    assert not view.flags.c_contiguous
    assert np.array_equal(decode_frame(encode_frame(view)), view)


@pytest.mark.parametrize(
    "text",
    ["not base64!", base64.b64encode(b"this is not a TIFF").decode("ascii"), ""],
)
def test_text_that_is_not_a_tiff_is_a_value_error(text: str) -> None:
    with pytest.raises(ValueError, match="the image is not"):
        decode_frame(text)


def test_the_text_result_of_a_kit_answer_is_found_by_its_type() -> None:
    result = [
        {"type": "image", "data": "x", "data_params": {}},
        {"type": "text", "data": "the document", "data_params": {"name": "models"}},
    ]

    assert result_text(result, "listing") == "the document"


@pytest.mark.parametrize(
    "result",
    [
        [],
        {},
        "text",
        [{"type": "text", "data": 5, "data_params": {}}],
        [{"type": "image", "data": "x"}],
        [None],
    ],
)
def test_a_kit_answer_with_no_text_result_is_malformed(result: object) -> None:
    with pytest.raises(ServedError) as caught:
        result_text(result, "listing")

    assert caught.value.kind == "malformed"


def test_a_status_error_names_the_status_and_the_servers_first_line() -> None:
    error = error_from_status(500, {"message": "boom\nsecond"})

    assert error.kind == "error"
    assert str(error) == "HTTP 500 from the model server; the server said: boom"
    assert str(error_from_status(502, None)) == "HTTP 502 from the model server"
    assert str(error_from_status(500, ["not", "a", "dict"])) == (
        "HTTP 500 from the model server"
    )


def test_an_error_survives_a_copy_and_a_pickle_with_its_kind() -> None:
    error = ServedError("timeout", "the model server did not answer in 2 s")

    for again in (copy.copy(error), pickle.loads(pickle.dumps(error))):
        assert isinstance(again, ServedError)
        assert again.kind == "timeout"
        assert str(again) == str(error)
