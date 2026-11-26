"""The model server's client: its codecs, its check, its detector, and ``load_model`` with a server set (inference.md §8.3 to §8.5).

Everything here runs on ``smc.testing.FakeModelServer`` around a ``StubDetector``: no
socket, no torch, no weights, no kit (FM-40). The wire on a real socket is
``test_inference_served_http.py``'s.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smc.inference import models as models_mod
from smc.inference import served as served_mod
from smc.inference.cards import ModelCard
from smc.inference.detector import Detection
from smc.inference.errors import ModelError
from smc.inference.models import ResolvedModel, load_model, model_identity
from smc.inference.preprocessing import ModelPreprocessing
from smc.inference.served import (
    CHECK_TIMEOUT_S,
    DETECT_TIMEOUT_S,
    SCHEMA,
    SERVER_ENV,
    TIMEOUT_ENV,
    ServedDetector,
    ServedError,
    ServedModel,
    check_served,
    decode_answer,
    decode_frame,
    decode_listing,
    encode_answer,
    encode_frame,
    encode_listing,
    error_from_status,
    result_text,
)
from smc.testing import FakeModelServer, StubDetector, stub_model

NATIVE = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
SHA = "a" * 64
CARD = ModelCard(
    name="stubmodel",
    weights="w.pth",
    weights_sha256=SHA,
    classes=("single", "doublet", "debris"),
    trained_um_per_px=None,
    preprocessing=NATIVE,
)
IDENTITY = model_identity(SHA, NATIVE)
SHAPE = (6, 9)
HOST_AND_PORT = ("models.test", "8000")

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


# --- the client, on a fake model server --------------------------------------------


def _stub(**kwargs: Any) -> StubDetector:
    """Frame 7 holds one confident detection; the device is cuda, so that it is seen to travel."""
    return StubDetector(
        {7: [Detection(1.0, 2.0, 3.0, 4.0, "single", 0.9)]}, device="cuda", **kwargs
    )


def _fake(stub: StubDetector | None = None, **kwargs: Any) -> FakeModelServer:
    return FakeModelServer(
        {
            CARD.name: stub_model(
                stub or _stub(),
                name=CARD.name,
                weights_sha256=SHA,
                preprocessing=NATIVE,
            )
        },
        **kwargs,
    )


def _load(
    fake: FakeModelServer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    threshold: float = 0.5,
    **overrides: Any,
) -> ResolvedModel:
    """``load_model`` with the fake installed, the way ``cells`` asks for a model."""
    fake.install(monkeypatch)
    arguments: dict[str, Any] = {
        "preprocessing": NATIVE,
        "frame_shape_px": SHAPE,
        "frame_um_per_px": 0.275,
        "device": "auto",
        "score_threshold": threshold,
        "weights": None,
    }
    arguments.update(overrides)
    return load_model(CARD, **arguments)


def _detector(
    fake: FakeModelServer, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> ServedDetector:
    detector = _load(fake, monkeypatch, **kwargs).detector
    assert isinstance(detector, ServedDetector)
    return detector


def _frame(key: int = 7) -> Any:
    return np.full(SHAPE, key, dtype=np.uint16)


def _detect_calls(fake: FakeModelServer) -> list[Any]:
    return [c for c in fake.calls if c.path == "/detect/process"]


def test_it_sends_the_frame_and_the_pixel_size_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    detector = _detector(fake, monkeypatch)
    frame = _frame(7).astype(np.float32)

    found = detector.detect(frame)

    (call,) = _detect_calls(fake)
    assert set(call.body) == {"model", "image", "pixel_size_um"}
    assert call.body["model"] == CARD.name
    assert call.body["pixel_size_um"] == 0.275
    assert call.frame.dtype == np.float32
    assert np.array_equal(call.frame, frame)
    assert call.read_timeout_s == DETECT_TIMEOUT_S
    assert found == [Detection(1.0, 2.0, 3.0, 4.0, "single", 0.9)]


def test_the_score_threshold_stays_on_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StubDetector(
        {
            7: [
                Detection(1.0, 2.0, 3.0, 4.0, "single", 0.3),
                Detection(5.0, 6.0, 7.0, 8.0, "single", 0.6),
                Detection(5.0, 6.0, 7.0, 8.0, "doublet", 0.5),
            ]
        }
    )
    fake = _fake(stub)

    strict = _detector(fake, monkeypatch, threshold=0.5).detect(_frame())
    loose = _detector(fake, monkeypatch, threshold=0.2).detect(_frame())

    # The predicate is `score >= threshold`: a score equal to it stays.
    assert [d.score for d in strict] == [0.6, 0.5]
    assert [d.score for d in loose] == [0.3, 0.6, 0.5]
    assert all("score_threshold" not in c.body for c in fake.calls)


@pytest.mark.parametrize(
    ("override", "says"),
    [
        (f"{'b' * 64}+{IDENTITY.split('+')[1]}", "the weights differ"),
        (f"{SHA}+{'c' * 64}", "the pre-processing differs"),
        (f"{'b' * 64}+{'c' * 64}", "the weights and the pre-processing differ"),
    ],
    ids=["weights", "pre-processing", "both"],
)
def test_an_answer_with_another_identity_is_refused_naming_the_half_that_differs(
    monkeypatch: pytest.MonkeyPatch, override: str, says: str
) -> None:
    fake = _fake()
    detector = _detector(fake, monkeypatch)
    fake.identity_override = override

    with pytest.raises(ServedError) as caught:
        detector.detect(_frame())

    assert caught.value.kind == "identity"
    assert says in str(caught.value)
    # The other half is not named when it did not differ.
    if says == "the weights differ":
        assert "pre-processing differs" not in str(caught.value)
    if says == "the pre-processing differs":
        assert "weights differ" not in str(caught.value)


class _Doctored:
    """A wire that answers every call with the same text: for the answers a real server does not give."""

    def __init__(self, text: str) -> None:
        self._text = text

    def post(
        self, path: str, body: Mapping[str, object], *, read_timeout_s: float
    ) -> object:
        return [{"type": "text", "data": self._text, "data_params": {"name": "x"}}]


def test_an_answer_for_another_model_is_refused() -> None:
    wire = _Doctored(encode_answer("another-model", IDENTITY, "cpu", []))
    detector = ServedDetector(
        wire,
        model=CARD.name,
        identity=IDENTITY,
        classes=CARD.classes,
        pixel_size_um=0.0,
        score_threshold=0.5,
        timeout_s=1.0,
        check=check_served(wire, CARD),
    )

    with pytest.raises(ServedError) as caught:
        detector.detect(_frame())

    assert caught.value.kind == "identity"
    assert "'another-model'" in str(caught.value)
    assert repr(CARD.name) in str(caught.value)


def test_an_absent_server_fails_the_call_with_its_diagnosis_and_never_an_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    detector = _detector(fake, monkeypatch)
    fake.absent = True

    with pytest.raises(ServedError) as caught:
        detector.detect(_frame())

    assert caught.value.kind == "unreachable"
    assert "does not answer" in str(caught.value)
    # The server never saw it: only the check's call is in the record.
    assert [c.path for c in fake.calls] == ["/models/process"]


def test_a_slow_server_is_a_timeout_at_the_calls_own_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    fake.latency_s = 5.0

    monkeypatch.setenv(TIMEOUT_ENV, "1")
    with pytest.raises(ServedError) as caught:
        _detector(fake, monkeypatch).detect(_frame())
    assert caught.value.kind == "timeout"
    assert "1 s" in str(caught.value)
    assert TIMEOUT_ENV in str(caught.value)
    assert _detect_calls(fake)[-1].read_timeout_s == 1.0

    # Against the default of 60 s the same server answers.
    monkeypatch.delenv(TIMEOUT_ENV)
    assert _detector(fake, monkeypatch).detect(_frame())


def test_a_failing_server_is_an_error_and_the_next_call_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    detector = _detector(fake, monkeypatch)
    fake.fail_next = 1

    with pytest.raises(ServedError) as caught:
        detector.detect(_frame())
    assert caught.value.kind == "error"
    assert "HTTP 500" in str(caught.value)

    assert detector.detect(_frame())


def test_it_retries_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake()
    detector = _detector(fake, monkeypatch)
    fake.fail_next = 1

    with pytest.raises(ServedError):
        detector.detect(_frame())

    assert len(_detect_calls(fake)) == 1


def test_the_device_is_unknown_before_an_answer_and_the_servers_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    fake.absent = True
    detector = _detector(fake, monkeypatch)
    assert detector.device == "unknown"

    fake.absent = False
    assert detector.detect(_frame())
    assert detector.device == "cuda"
    # With a check that found the model, the device is known before any frame.
    assert _detector(fake, monkeypatch).device == "cuda"


def test_a_frame_that_is_not_two_dimensional_is_refused_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    detector = _detector(fake, monkeypatch)
    sent = len(fake.calls)

    for shape in ((4, 5, 3), (4,), (), (0, 5)):
        with pytest.raises(ValueError, match="2-D"):
            detector.detect(np.zeros(shape, np.uint16))

    assert len(fake.calls) == sent


def test_the_detector_declares_the_cards_classes_and_the_check_it_was_made_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector(_fake(), monkeypatch)

    assert detector.classes == CARD.classes
    assert detector.check.ok
    assert detector.check.kind is None


# --- check_served ------------------------------------------------------------------


def test_the_check_of_a_good_server_names_the_model_its_device_and_the_identity() -> (
    None
):
    found = check_served(_fake(), CARD)

    assert found.ok
    assert found.kind is None
    assert found.device == "cuda"
    weights, rule = IDENTITY.split("+")
    assert found.message == (
        f"model server: stubmodel is served on cuda, identity {weights[:12]}+{rule[:12]}"
    )


def test_the_check_of_an_absent_server_is_a_finding_and_never_raises() -> None:
    fake = _fake()
    fake.absent = True

    found = check_served(fake, CARD)

    assert not found.ok
    assert found.kind == "unreachable"
    assert "does not answer" in found.message
    assert found.device is None


def test_the_check_of_a_server_that_does_not_serve_the_model_names_what_it_serves() -> (
    None
):
    other = StubDetector()
    fake = FakeModelServer(
        {
            "a": stub_model(other, name="a"),
            "b": stub_model(other, name="b"),
            CARD.name: stub_model(
                other, name=CARD.name, weights_sha256=SHA, preprocessing=NATIVE
            ),
        }
    )
    fake.hidden = {CARD.name}

    found = check_served(fake, CARD)

    assert (found.ok, found.kind, found.device) == (False, "not_served", None)
    assert "does not serve stubmodel (it serves: a, b)" in found.message

    fake.hidden = {"a", "b", CARD.name}
    nothing = check_served(fake, CARD)
    assert "does not serve stubmodel (it serves no model)" in nothing.message


@pytest.mark.parametrize(
    ("override", "says"),
    [
        (f"{'b' * 64}+{IDENTITY.split('+')[1]}", "the weights differ"),
        (f"{SHA}+{'c' * 64}", "the pre-processing differs"),
        (f"{'b' * 64}+{'c' * 64}", "the weights and the pre-processing differ"),
    ],
    ids=["weights", "pre-processing", "both"],
)
def test_the_check_of_another_identity_names_the_half(override: str, says: str) -> None:
    fake = _fake()
    fake.identity_override = override

    found = check_served(fake, CARD)

    assert (found.ok, found.kind) == (False, "identity")
    assert "another identity" in found.message
    assert says in found.message
    assert found.device == "cuda"


def test_the_check_of_a_slow_server() -> None:
    fake = _fake()
    fake.latency_s = CHECK_TIMEOUT_S + 1

    found = check_served(fake, CARD)

    assert (found.ok, found.kind) == (False, "timeout")
    assert "did not answer in 10 s" in found.message
    assert fake.calls[0].read_timeout_s == CHECK_TIMEOUT_S


def test_the_check_of_an_error_and_of_nonsense_are_findings_too() -> None:
    fake = _fake()
    fake.fail_next = 1
    assert check_served(fake, CARD).kind == "error"

    class Nonsense:
        def post(
            self, path: str, body: Mapping[str, object], *, read_timeout_s: float
        ) -> object:
            return {"not": "a kit answer"}

    found = check_served(Nonsense(), CARD)
    assert (found.ok, found.kind) == (False, "malformed")


# --- load_model with a server set --------------------------------------------------


def _not_here(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the local loaders fail the test if they are called; the list says if one was."""
    called: list[str] = []

    def refuse(*args: object, **kwargs: object) -> ResolvedModel:
        called.append("local")
        raise AssertionError("the model was loaded on this PC")

    monkeypatch.setattr(models_mod, "load_local_model", refuse)
    monkeypatch.setattr(models_mod, "load_local", refuse)
    return called


def _ask(device: str = "auto") -> ResolvedModel:
    return load_model(
        CARD,
        preprocessing=NATIVE,
        frame_shape_px=SHAPE,
        frame_um_per_px=0.275,
        device=device,
        score_threshold=0.5,
    )


def test_with_a_server_set_the_name_resolves_to_the_served_model_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = _not_here(monkeypatch)

    resolved = _load(_fake(), monkeypatch)

    assert called == []
    assert resolved.runs_on == "served"
    assert resolved.name == CARD.name
    assert resolved.identity == model_identity(CARD.weights_sha256, CARD.preprocessing)
    assert resolved.weights_sha256 == CARD.weights_sha256
    assert isinstance(resolved.detector, ServedDetector)


def test_without_a_server_it_resolves_locally_and_no_call_is_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = stub_model(StubDetector(), name=CARD.name)
    asked: list[object] = []

    def load_here(*args: object, **kwargs: object) -> ResolvedModel:
        asked.append(kwargs)
        return local

    def no_wire(address: str) -> object:
        raise AssertionError("a wire was opened with no server set")

    monkeypatch.setattr(models_mod, "load_local_model", load_here)
    monkeypatch.setattr(served_mod, "open_wire", no_wire)

    assert _ask("cpu") is local
    assert len(asked) == 1


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_server_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    # FM-78: a variable that is set and blank must not turn the model into a call.
    local = stub_model(StubDetector(), name=CARD.name)

    def no_wire(address: str) -> object:
        raise AssertionError("a wire was opened")

    monkeypatch.setattr(models_mod, "load_local_model", lambda *a, **k: local)
    monkeypatch.setattr(served_mod, "open_wire", no_wire)
    monkeypatch.setenv(SERVER_ENV, blank)

    assert _ask("cpu") is local


def test_a_variable_that_is_not_an_address_is_refused_and_never_means_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = _not_here(monkeypatch)
    monkeypatch.setenv(SERVER_ENV, "localhost:8000")

    with pytest.raises(ModelError) as caught:
        _ask()

    assert str(caught.value) == f"{SERVER_ENV} is not an address like http://host:8000"
    assert called == []


def test_a_dead_server_never_loads_the_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = _not_here(monkeypatch)
    fake = _fake()
    fake.absent = True

    resolved = _load(fake, monkeypatch)

    assert resolved.runs_on == "served"
    detector = resolved.detector
    assert isinstance(detector, ServedDetector)
    assert not detector.check.ok
    assert detector.check.kind == "unreachable"
    with pytest.raises(ServedError) as caught:
        detector.detect(_frame())
    assert caught.value.kind == "unreachable"
    assert called == []


@pytest.mark.parametrize(
    ("overrides", "says"),
    [
        ({"weights": Path("w.pth")}, "model_path names a file on this PC"),
        ({"device": "cuda"}, "device=cuda is for a model that runs on this PC"),
        ({"device": "cpu"}, "the server chooses its own"),
        (
            {
                "preprocessing": ModelPreprocessing(
                    percentiles=(1.0, 99.5), model_um_per_px=None
                )
            },
            "the server runs the card's pre-processing",
        ),
        ({"frame_um_per_px": -1.0}, "pixel size must be a number of at least 0"),
        (
            {"frame_um_per_px": float("nan")},
            "pixel size must be a number of at least 0",
        ),
        (
            {"frame_um_per_px": float("inf")},
            "pixel size must be a number of at least 0",
        ),
    ],
    ids=["weights", "cuda", "cpu", "percentile", "negative", "nan", "inf"],
)
def test_what_a_served_model_cannot_honour_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], says: str
) -> None:
    fake = _fake()

    with pytest.raises(ModelError) as caught:
        _load(fake, monkeypatch, **overrides)

    assert says in str(caught.value)
    assert fake.calls == []


def test_a_pixel_size_of_zero_is_sent_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake()

    _detector(fake, monkeypatch, frame_um_per_px=0.0).detect(_frame())

    (call,) = _detect_calls(fake)
    assert call.body["pixel_size_um"] == 0.0
    assert isinstance(call.body["pixel_size_um"], float)


def test_the_timeout_variable_sets_the_detect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    monkeypatch.setenv(TIMEOUT_ENV, "2.5")

    _detector(fake, monkeypatch).detect(_frame())

    assert _detect_calls(fake)[0].read_timeout_s == 2.5
    # The check keeps its own timeout.
    assert fake.calls[0].read_timeout_s == CHECK_TIMEOUT_S


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_timeout_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    fake = _fake()
    monkeypatch.setenv(TIMEOUT_ENV, blank)

    _detector(fake, monkeypatch).detect(_frame())

    assert _detect_calls(fake)[0].read_timeout_s == DETECT_TIMEOUT_S


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "nan", "inf", "-inf", "1 s"])
def test_a_bad_timeout_value_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    fake = _fake()
    monkeypatch.setenv(TIMEOUT_ENV, bad)

    with pytest.raises(ModelError) as caught:
        _load(fake, monkeypatch)

    assert str(caught.value) == f"{TIMEOUT_ENV} must be a number of seconds above 0"
    assert fake.calls == []


def test_an_address_other_than_the_fakes_is_never_contacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake()
    fake.install(monkeypatch)
    monkeypatch.setenv(SERVER_ENV, "http://elsewhere.test:8000")

    detector = _ask().detector

    assert isinstance(detector, ServedDetector)
    assert detector.check.kind == "unreachable"
    with pytest.raises(ServedError):
        detector.detect(_frame())
    assert fake.calls == []


def test_the_check_is_logged_without_the_address(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="smc.inference.served")
    fake = _fake()

    _load(fake, monkeypatch)
    fake.absent = True
    _load(fake, monkeypatch)

    good, bad = [r for r in caplog.records if r.name == "smc.inference.served"]
    assert (good.levelno, bad.levelno) == (logging.INFO, logging.WARNING)
    assert "stubmodel is served on cuda" in good.getMessage()
    assert "does not answer" in bad.getMessage()
    for part in HOST_AND_PORT:
        assert part not in caplog.text
