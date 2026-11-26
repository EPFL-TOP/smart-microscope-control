"""The fake model server: it answers as the kit does, fails as a test asks, and records what it was asked (inference.md §8.8).

It is the bed of every test of the served detector and of ``cells`` with a server set, so
its own behaviour is pinned here, through ``post`` and with no client in between: a client
that was wrong in the same way as the fake would hide it.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

from smc.inference import served
from smc.inference.detector import Detection
from smc.inference.models import model_identity
from smc.inference.preprocessing import ModelPreprocessing
from smc.inference.served import (
    ServedError,
    decode_answer,
    decode_listing,
    encode_frame,
    result_text,
)
from smc.testing import FakeModelServer, ServedCall, StubDetector, stub_model

NATIVE = ModelPreprocessing(percentiles=(0.5, 99.5), model_um_per_px=None)
OTHER_RULE = ModelPreprocessing(percentiles=(1.0, 99.0), model_um_per_px=None)
SHA = "a" * 64
FOUND = Detection(1.0, 2.0, 3.0, 4.0, "single", 0.9)


def _fake(**kwargs: Any) -> FakeModelServer:
    stub = StubDetector({7: [FOUND]}, device="cuda")
    models = {
        "alpha": stub_model(
            stub, name="alpha", weights_sha256=SHA, preprocessing=NATIVE
        ),
        "beta": stub_model(
            StubDetector(classes=("blob",)),
            name="beta",
            weights_sha256="b" * 64,
            preprocessing=NATIVE,
        ),
    }
    return FakeModelServer(models, **kwargs)


def _body(
    key: int = 7, model: str = "alpha", pixel_size_um: Any = 0.275
) -> dict[str, object]:
    return {
        "model": model,
        "image": encode_frame(np.full((6, 9), key, dtype=np.uint16)),
        "pixel_size_um": pixel_size_um,
    }


def _detect(fake: FakeModelServer, body: dict[str, object] | None = None) -> object:
    return fake.post(
        "/detect/process", _body() if body is None else body, read_timeout_s=5.0
    )


def _listing(fake: FakeModelServer) -> object:
    return fake.post("/models/process", {}, read_timeout_s=5.0)


def _error(fake: FakeModelServer, path: str, body: dict[str, object]) -> ServedError:
    with pytest.raises(ServedError) as caught:
        fake.post(path, body, read_timeout_s=5.0)
    return caught.value


def test_it_answers_as_the_kit_does() -> None:
    fake = _fake()

    # 201 and the result list, one text result per algorithm.
    answered = _detect(fake)
    assert isinstance(answered, list)
    assert answered[0]["type"] == "text"
    assert answered[0]["data_params"] == {"name": "detections"}
    assert decode_answer(result_text(answered, "answer")).detections == (FOUND,)
    listed = _listing(fake)
    assert isinstance(listed, list)
    assert listed[0]["data_params"] == {"name": "models"}
    assert [m.name for m in decode_listing(result_text(listed, "listing"))] == [
        "alpha",
        "beta",
    ]

    # 404: another path, or an algorithm the server does not have.
    not_found = _error(fake, "/nothing/process", {})
    assert not_found.kind == "error"
    assert "HTTP 404" in str(not_found)
    assert "Algorithm nothing not found" in str(not_found)
    assert "Not Found" in str(_error(fake, "/", {}))

    # 422: a fourth key, a missing one, a wrong type, a pixel size below 0.
    extra = _error(fake, "/detect/process", {**_body(), "threshold": 0.5})
    assert "HTTP 422" in str(extra)
    assert "body.threshold: Extra inputs are not permitted" in str(extra)
    assert "body.pixel_size_um: Field required" in str(
        _error(fake, "/detect/process", {"model": "alpha", "image": "x"})
    )
    assert "HTTP 422" in str(_error(fake, "/detect/process", _body(model=7)))  # type: ignore[arg-type]
    assert "greater than or equal to 0" in str(
        _error(fake, "/detect/process", _body(pixel_size_um=-0.1))
    )
    assert "HTTP 422" in str(_error(fake, "/detect/process", _body(pixel_size_um=True)))
    assert "HTTP 422" in str(_error(fake, "/models/process", {"x": 1}))

    # 500: a ValueError of the model's, and an image that is not a TIFF.
    unknown = _error(fake, "/detect/process", _body(model="gamma"))
    assert "HTTP 500" in str(unknown)
    assert "the server said: the server does not serve 'gamma'" in str(unknown)
    flat = _error(
        fake,
        "/detect/process",
        {**_body(), "image": encode_frame(np.zeros((3,), np.uint8))},
    )
    assert "HTTP 500" in str(flat)
    assert "2-D" in str(flat)
    not_a_tiff = _error(fake, "/detect/process", {**_body(), "image": "AAAA"})
    assert "HTTP 500" in str(not_a_tiff)
    assert "the image is not a TIFF" in str(not_a_tiff)


def test_a_model_that_raises_is_a_500_with_its_message() -> None:
    def boom(key: int) -> None:
        raise RuntimeError("CUDA out of memory\nsecond line")

    stub = StubDetector(on_frame=boom)
    fake = FakeModelServer({"m": stub_model(stub, name="m")})

    error = _error(fake, "/detect/process", _body(model="m"))

    assert error.kind == "error"
    assert str(error).endswith("the server said: CUDA out of memory")


def test_absent_is_unreachable_and_the_server_never_saw_the_call() -> None:
    fake = _fake()
    fake.absent = True

    for path, body in (("/models/process", {}), ("/detect/process", _body())):
        error = _error(fake, path, body)
        assert error.kind == "unreachable"
        assert "models.test" not in str(error)

    assert fake.calls == []
    fake.absent = False
    _listing(fake)
    assert len(fake.calls) == 1


def test_latency_above_the_calls_timeout_is_a_timeout_and_nothing_sleeps() -> None:
    fake = _fake()
    fake.latency_s = 5.0

    with pytest.raises(ServedError) as caught:
        fake.post("/detect/process", _body(), read_timeout_s=1.0)
    assert caught.value.kind == "timeout"
    assert "1 s" in str(caught.value)
    # Within the timeout it answers; the model did not run for the one that timed out.
    assert fake.post("/detect/process", _body(), read_timeout_s=60.0)
    assert len(fake.calls) == 2


def test_fail_next_fails_n_calls_then_answers() -> None:
    fake = _fake()
    fake.fail_next = 2

    failures = [_error(fake, "/detect/process", _body()) for _ in range(2)]

    assert [f.kind for f in failures] == ["error", "error"]
    assert all("HTTP 500" in str(f) for f in failures)
    assert fake.fail_next == 0
    assert _detect(fake)
    assert len(fake.calls) == 3


def test_identity_override_changes_the_listing_and_the_answers() -> None:
    fake = _fake()
    own = model_identity(SHA, NATIVE)
    assert decode_answer(result_text(_detect(fake), "answer")).identity == own

    fake.identity_override = "another+identity"

    listing = decode_listing(result_text(_listing(fake), "listing"))
    assert {m.identity for m in listing} == {"another+identity"}
    answer = decode_answer(result_text(_detect(fake), "answer"))
    assert answer.identity == "another+identity"
    # Nothing else of the answer changes.
    assert answer.detections == (FOUND,)
    assert answer.device == "cuda"


def test_hidden_names_are_not_served() -> None:
    fake = _fake()
    fake.hidden = {"alpha"}

    listing = decode_listing(result_text(_listing(fake), "listing"))
    refused = _error(fake, "/detect/process", _body(model="alpha"))

    assert [m.name for m in listing] == ["beta"]
    assert "does not serve 'alpha'" in str(refused)
    assert "it serves: beta" in str(refused)
    fake.hidden.clear()
    assert len(decode_listing(result_text(_listing(fake), "listing"))) == 2


def test_on_call_runs_before_the_answer() -> None:
    stub = StubDetector({7: [FOUND]})
    fake = FakeModelServer({"alpha": stub_model(stub, name="alpha")})
    seen: list[ServedCall] = []

    def die(call: ServedCall) -> None:
        seen.append(call)
        fake.absent = True

    fake.on_call = die

    error = _error(fake, "/detect/process", _body())

    # That very call was refused, and it had arrived: it is recorded.
    assert error.kind == "unreachable"
    assert seen == fake.calls
    assert len(fake.calls) == 1
    # The model did not see the frame; the call had it.
    assert stub.seen == []
    assert fake.calls[0].frame is not None


def test_the_calls_are_recorded_in_order_with_the_exact_body_and_the_decoded_frame() -> (
    None
):
    fake = _fake()
    first = _body(key=7)

    _listing(fake)
    fake.post("/detect/process", first, read_timeout_s=12.5)
    with pytest.raises(ServedError):
        fake.post(
            "/detect/process", {**_body(key=9), "threshold": 0.5}, read_timeout_s=3.0
        )

    listing_call, detect_call, rejected = fake.calls
    assert [c.path for c in fake.calls] == [
        "/models/process",
        "/detect/process",
        "/detect/process",
    ]
    assert listing_call.body == {}
    assert listing_call.frame is None
    assert detect_call.body == first
    assert detect_call.read_timeout_s == 12.5
    assert detect_call.frame is not None
    assert detect_call.frame.dtype == np.uint16
    assert int(detect_call.frame[0, 0]) == 7
    assert detect_call.frame.shape == (6, 9)
    # A call the fake refused is recorded as it was sent, with the key it should not have had.
    assert set(rejected.body) == {"model", "image", "pixel_size_um", "threshold"}
    assert int(rejected.frame[0, 0]) == 9  # type: ignore[index]
    assert json.dumps(first["pixel_size_um"]) == "0.275"


def test_the_body_a_test_built_is_not_changed_by_the_record() -> None:
    fake = _fake()
    body = _body()

    _detect(fake, body)
    body["model"] = "changed afterwards"

    assert fake.calls[0].body["model"] == "alpha"


def test_the_identity_follows_the_rule_of_the_model_it_holds() -> None:
    # FM-260: the identity is worked out from the rule, not written in the test.
    stub = StubDetector()
    fake = FakeModelServer(
        {
            "native": stub_model(
                stub, name="native", weights_sha256=SHA, preprocessing=NATIVE
            ),
            "other": stub_model(
                stub, name="other", weights_sha256=SHA, preprocessing=OTHER_RULE
            ),
        }
    )

    identities = {
        m.name: m.identity
        for m in decode_listing(result_text(_listing(fake), "listing"))
    }

    assert identities == {
        "native": model_identity(SHA, NATIVE),
        "other": model_identity(SHA, OTHER_RULE),
    }
    assert identities["native"] != identities["other"]
    # The two halves: the same weights, another rule.
    assert identities["native"].split("+")[0] == identities["other"].split("+")[0]


def test_install_serves_only_its_own_address(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake(address="http://stand-gpu.test:9123")

    fake.install(monkeypatch)

    assert os.environ[served.SERVER_ENV] == "http://stand-gpu.test:9123"
    assert served.open_wire("http://stand-gpu.test:9123") is fake
    elsewhere = served.open_wire("http://another.test:8000")
    assert elsewhere is not fake
    with pytest.raises(ServedError) as caught:
        elsewhere.post("/models/process", {}, read_timeout_s=1.0)
    assert caught.value.kind == "unreachable"
    assert fake.calls == []


def test_its_repr_does_not_show_the_address() -> None:
    assert "stand-gpu" not in repr(_fake(address="http://stand-gpu.test:9123"))
