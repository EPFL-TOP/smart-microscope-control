"""The model cards: what a name stands for, and every way a card can be wrong (inference.md §3).

Cards are read with no torch and no weights, so CI runs all of it. Every key is
required and every refusal names the card's file and the key (FM-50: never an
absolute path), because a card is written by hand, on the analysis track's side
of the repository, by someone who does not read the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.inference import cards as cards_mod
from smc.inference.blobs import BLOB_DETECTOR
from smc.inference.cards import (
    DEFAULT_MODEL_NAME,
    ModelCard,
    model_card,
    model_cards,
    model_names,
    parse_card,
)
from smc.inference.errors import ModelError

SHA = "b1160edae7dd2e8e1897f959fd55b77481bbfa92d236aaa1202cd94101b2cb7e"
SOURCE = "cards/x.toml"


def _card(
    *,
    name: str = "x",
    weights: str = "w.pth",
    sha: str = SHA,
    classes: str = '["a", "b"]',
    trained: str | None = '"unknown"',
    percentiles: str = "[0.5, 99.5]",
    native: str = '"native"',
    extra: str = "",
) -> bytes:
    """A valid card as TOML text, with any value replaced; ``trained=None`` leaves the key out."""
    lines = [
        f'name = "{name}"',
        # A backslash is a TOML escape: the file has to say it, or it is not TOML.
        f'weights = "{weights.replace(chr(92), chr(92) * 2)}"',
        f'weights_sha256 = "{sha}"',
        f"classes = {classes}",
    ]
    if trained is not None:
        lines.append(f"trained_um_per_px = {trained}")
    lines += [extra, "[preprocessing]", f"percentiles = {percentiles}"]
    lines.append(f"model_um_per_px = {native}")
    return "\n".join(lines).encode("utf-8")


def _refusal(data: bytes, source: str = SOURCE) -> str:
    with pytest.raises(ModelError) as caught:
        parse_card(data, source=source)
    return str(caught.value)


# --- the card that ships ------------------------------------------------------


def test_the_cell_classes_v2_card_names_its_weights_classes_and_preprocessing() -> None:
    card = model_card(DEFAULT_MODEL_NAME)

    assert DEFAULT_MODEL_NAME == "cell-classes-v2"
    assert card.name == "cell-classes-v2"
    assert card.weights == "cell_classes_model-cat-v2.pth"
    assert card.weights_sha256 == (
        "b1160edae7dd2e8e1897f959fd55b77481bbfa92d236aaa1202cd94101b2cb7e"
    )
    assert card.classes == ("single", "doublet", "debris")
    # Unknown, never guessed: the checkpoint records no pixel size.
    assert card.trained_um_per_px is None
    assert card.preprocessing.percentiles == (0.5, 99.5)
    assert card.preprocessing.model_um_per_px is None


def test_every_card_in_the_package_is_valid_and_named_as_its_file() -> None:
    names = model_names()
    cards = model_cards()

    assert names
    assert tuple(sorted(cards)) == names
    assert all(card.name == name for name, card in cards.items())
    assert DEFAULT_MODEL_NAME in names


def test_the_name_the_card_reserves_is_the_blob_detectors() -> None:
    """``cards.py`` cannot import ``blobs.py`` (scikit-image): the two spellings are pinned equal."""
    assert BLOB_DETECTOR == "blobs"
    assert "blobs" not in model_names()


def test_only_toml_files_of_the_folder_are_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "cards"
    folder.mkdir()
    (folder / "b.toml").write_bytes(_card(name="b"))
    (folder / "a.toml").write_bytes(_card(name="a"))
    (folder / "README.md").write_text("not a card", encoding="utf-8")
    (folder / "notes.toml.bak").write_text("not a card", encoding="utf-8")
    (folder / "dir.toml").mkdir()
    monkeypatch.setattr(cards_mod, "files", lambda package: tmp_path)

    assert model_names() == ("a", "b")
    assert model_card("a").name == "a"


# --- the words for "none" -----------------------------------------------------


def test_unknown_and_native_are_read_as_none() -> None:
    card = parse_card(_card(trained='"unknown"', native='"native"'), source=SOURCE)

    assert card.trained_um_per_px is None
    assert card.preprocessing.model_um_per_px is None


def test_a_number_is_kept_as_a_pixel_size() -> None:
    card = parse_card(_card(trained="0.27555", native="0.49"), source=SOURCE)

    assert card.trained_um_per_px == 0.27555
    assert card.preprocessing.model_um_per_px == 0.49


def test_a_card_without_its_trained_pixel_size_is_refused() -> None:
    """TOML has no null: a missing key is refused, never defaulted to unknown."""
    reason = _refusal(_card(trained=None))

    assert reason.startswith(SOURCE)
    assert "trained_um_per_px" in reason
    assert "required" in reason


def test_a_card_without_its_preprocessing_table_is_refused() -> None:
    data = b'name = "x"\nweights = "w"\nweights_sha256 = "' + SHA.encode() + b'"\n'
    data += b'classes = ["a"]\ntrained_um_per_px = "unknown"\n'

    assert "preprocessing" in _refusal(data)


@pytest.mark.parametrize("bad", ["true", '"0.5"', '"unkown"', '"native"', "[1.0]"])
def test_a_bool_or_a_string_is_not_a_trained_pixel_size(bad: str) -> None:
    reason = _refusal(_card(trained=bad))

    assert reason.startswith(SOURCE)
    assert "trained_um_per_px" in reason


@pytest.mark.parametrize("bad", ["true", '"0.5"', '"natve"', '"unknown"'])
def test_a_bool_or_a_string_is_not_a_model_pixel_size(bad: str) -> None:
    reason = _refusal(_card(native=bad))

    assert reason.startswith(SOURCE)
    assert "preprocessing.model_um_per_px" in reason


@pytest.mark.parametrize("bad", ["0", "-1", "10.5", "inf", "nan"])
def test_a_pixel_size_outside_what_a_camera_has_is_refused(bad: str) -> None:
    assert "trained_um_per_px" in _refusal(_card(trained=bad))
    assert "model_um_per_px" in _refusal(_card(native=bad))


def test_an_int_in_a_card_is_a_float() -> None:
    card = parse_card(
        _card(trained="1", percentiles="[1, 99]", native="2"), source=SOURCE
    )

    assert card.trained_um_per_px == 1.0
    assert isinstance(card.trained_um_per_px, float)
    assert card.preprocessing.percentiles == (1.0, 99.0)
    assert all(isinstance(p, float) for p in card.preprocessing.percentiles)
    assert isinstance(card.preprocessing.model_um_per_px, float)


@pytest.mark.parametrize(
    "percentiles",
    ["[50, 99.5]", "[0.5, 50]", "[-1, 99]", "[0.5, 101]", "[0.5]", "[1, 2, 3]", "true"],
)
def test_card_percentiles_outside_the_cells_bounds_are_refused(
    percentiles: str,
) -> None:
    """The bounds of ``cells``' parameters, so that a parameter always leaves a valid pair."""
    reason = _refusal(_card(percentiles=percentiles))

    assert "preprocessing.percentiles" in reason


# --- the other keys -----------------------------------------------------------


def test_a_card_with_an_unknown_key_is_refused_with_the_key_and_the_file() -> None:
    reason = _refusal(_card(extra='architecture = "unet"'))

    assert reason.startswith(SOURCE)
    assert "architecture" in reason


def test_a_card_with_an_unknown_key_in_its_preprocessing_is_refused() -> None:
    reason = _refusal(_card(native='"native"\nbit_depth = 12'))

    assert "preprocessing.bit_depth" in reason


def test_a_card_whose_name_is_not_its_file_name_is_refused() -> None:
    reason = _refusal(_card(name="x"), source="cards/other.toml")

    assert reason.startswith("cards/other.toml")
    assert "'x'" in reason
    assert "'other'" in reason


@pytest.mark.parametrize(
    "name", ["Cell", "-a", "a-", "a--b", "a_b", "a b", "", "x" * 65, "blobs"]
)
def test_a_name_that_is_not_a_model_name_is_refused(name: str) -> None:
    reason = _refusal(_card(name=name), source=f"cards/{name}.toml")

    assert "name" in reason


def test_a_name_with_dots_and_hyphens_is_a_model_name() -> None:
    card = parse_card(_card(name="cells-v1.5-a"), source="cards/cells-v1.5-a.toml")

    assert card.name == "cells-v1.5-a"


@pytest.mark.parametrize(
    "weights", ["", ".", "..", "../w.pth", "a/w.pth", "a\\w.pth", "C:w.pth"]
)
def test_a_weights_entry_that_is_a_path_is_refused(weights: str) -> None:
    reason = _refusal(_card(weights=weights))

    assert reason.startswith(SOURCE)
    assert "weights" in reason
    assert "path" in reason


@pytest.mark.parametrize(
    "sha",
    [SHA[:-1], SHA.upper(), SHA[:-1] + "g", SHA + "0", "", " " + SHA[1:]],
    ids=["63 characters", "upper case", "a g", "65 characters", "empty", "a space"],
)
def test_a_weights_sha256_that_is_not_64_lowercase_hex_is_refused(sha: str) -> None:
    reason = _refusal(_card(sha=sha))

    assert "weights_sha256" in reason
    assert "64" in reason


@pytest.mark.parametrize(
    "classes",
    [
        '["a", "a"]',
        '["__background__", "a"]',
        "[]",
        '["a", ""]',
        '["a", " "]',
        "[1, 2]",
    ],
    ids=["duplicate", "background", "none", "empty name", "blank name", "numbers"],
)
def test_classes_must_be_distinct_and_exclude_the_background(classes: str) -> None:
    assert "classes" in _refusal(_card(classes=classes))


# --- the file ----------------------------------------------------------------


def test_a_card_with_a_byte_order_mark_is_read() -> None:
    """Notepad writes one (FM-80)."""
    card = parse_card(b"\xef\xbb\xbf" + _card(), source=SOURCE)

    assert card.name == "x"


def test_a_card_in_utf16_is_refused_with_its_name() -> None:
    data = _card().decode("utf-8").encode("utf-16")

    reason = _refusal(data)

    assert reason.startswith(SOURCE)
    assert "UTF-8" in reason


def test_a_card_that_is_not_toml_is_refused_with_its_name() -> None:
    reason = _refusal(b"name = ")

    assert reason.startswith(SOURCE)
    assert "TOML" in reason


def test_no_card_message_holds_a_path_of_this_pc() -> None:
    """Run documents and issues carry these lines (FM-50)."""
    reasons = [_refusal(b"name = "), _refusal(_card(trained=None))]
    with pytest.raises(ModelError) as unknown:
        model_card("nope")

    for reason in [*reasons, str(unknown.value)]:
        assert str(Path(cards_mod.__file__).parent) not in reason
        assert "\\" not in reason


# --- the look-up --------------------------------------------------------------


def test_an_unknown_model_name_lists_the_known_ones() -> None:
    with pytest.raises(ModelError) as caught:
        model_card("cell-classes-v9")

    message = str(caught.value)
    assert "'cell-classes-v9'" in message
    assert "cell-classes-v2" in message
    assert "src/smc/inference/cards/" in message


@pytest.mark.parametrize(
    "name", ["../cells", "a/b", "a\\b", "Cell", "blobs", "", "x" * 65]
)
def test_a_name_with_a_path_in_it_is_refused_before_any_file_is_opened(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def opened(card_name: str) -> bytes:
        raise AssertionError(f"a file was opened for {card_name!r}")

    monkeypatch.setattr(cards_mod, "_read_card_bytes", opened)

    with pytest.raises(ModelError, match="no model named"):
        model_card(name)


def test_a_bad_card_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "cards"
    folder.mkdir()
    (folder / "good.toml").write_bytes(_card(name="good"))
    (folder / "bad.toml").write_bytes(b"name = ")
    monkeypatch.setattr(cards_mod, "files", lambda package: tmp_path)

    assert model_card("good").name == "good"
    with pytest.raises(ModelError, match=r"cards/bad\.toml"):
        model_card("bad")
    with pytest.raises(ModelError, match=r"cards/bad\.toml"):
        model_cards()


def test_a_card_is_frozen() -> None:
    card = model_card(DEFAULT_MODEL_NAME)

    assert isinstance(card, ModelCard)
    with pytest.raises(ValueError, match="frozen"):
        card.name = "other"  # type: ignore[misc]
