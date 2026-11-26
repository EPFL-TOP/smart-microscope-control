"""Models by name: one TOML card per model, in ``cards/``, read into ``ModelCard`` (inference.md §3).

A card says what a name stands for: the weights' file and SHA-256, the classes,
the pixel size the model was trained at and the pre-processing. It is shipped in
the package, and the folder is the analysis track's in ``.github/CODEOWNERS``
(ADR-0011): a model is added by a pull request that adds its card. Nothing here
opens the weights or imports torch.

Every key is required, so that nothing is defaulted silently, and an *unknown*
or *native* value is a word (TOML has no null): a pixel size that is not known
is said as unknown, never guessed (CLAUDE.md, *Measured beats assumed*).
"""

from __future__ import annotations

import codecs
import re
import sys
from importlib.resources import files

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictStr,
    ValidationError,
    field_validator,
)

from smc.inference.errors import ModelError
from smc.inference.preprocessing import (
    ModelPreprocessing,
    check_pixel_size,
    pixel_size_or_word,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on the 3.10 CI job
    import tomli as tomllib

__all__ = [
    "DEFAULT_MODEL_NAME",
    "ModelCard",
    "model_card",
    "model_cards",
    "model_names",
    "parse_card",
]

#: The model ``cells`` runs when none is asked for: the single / doublet / debris
#: Faster R-CNN that ``nikon-control`` trained (m3 §9.4).
DEFAULT_MODEL_NAME = "cell-classes-v2"

#: What a model's name may be: it is the card's file name and a value in a result.
_NAME = re.compile(r"[a-z0-9]+([.-][a-z0-9]+)*")
_MAX_NAME_LENGTH = 64
#: ``blobs.BLOB_DETECTOR``, the name a result gives the blob detector, which is
#: not a card (a test pins the two equal; ``blobs.py`` imports scikit-image).
_RESERVED_NAME = "blobs"
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: The names Windows keeps for devices, in upper case: a file named ``NUL.pth`` or
#: ``con.toml`` is a device there, and Git for Windows cannot check it out, so a card
#: made on a Mac with one passes every test but the Windows job's checkout (FM-24).
#: The list is its own, as ``smc.plugins.scan`` has: ``smc.inference`` imports nothing
#: of ``smc.plugins``, and ``PurePath.is_reserved`` is deprecated in Python 3.13.
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(10)}
    | {f"LPT{n}" for n in range(10)}
)
_BACKGROUND = "__background__"
_CARDS_FOLDER = "cards"
_SUFFIX = ".toml"
#: Where the cards are, for a message: the repository's path, never an absolute one (FM-50).
_WHERE = "src/smc/inference/cards/"


def _is_device_name(file_name: str) -> bool:
    """Whether Windows reads this file name as a device: the part before the first dot, in any case."""
    return file_name.split(".", 1)[0].upper() in _DEVICE_NAMES


class ModelCard(BaseModel):
    """A model by name: its weights, its classes, its training pixel size, its pre-processing."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str
    weights: str
    weights_sha256: str
    classes: tuple[StrictStr, ...]
    trained_um_per_px: StrictFloat | None
    preprocessing: ModelPreprocessing

    @field_validator("name")
    @classmethod
    def _name_is_a_file_stem(cls, value: str) -> str:
        if (
            not _NAME.fullmatch(value)
            or len(value) > _MAX_NAME_LENGTH
            or value == _RESERVED_NAME
            or _is_device_name(value)
        ):
            raise ValueError(
                f"{value!r} is not a model name: lower case letters and digits "
                f"joined by '-' or '.', at most {_MAX_NAME_LENGTH} characters, "
                f"not {_RESERVED_NAME!r} (the blob detector's) and not a name "
                f"Windows keeps for a device (con, nul, com1, ...)"
            )
        return value

    @field_validator("weights")
    @classmethod
    def _weights_is_a_file_name(cls, value: str) -> str:
        if value in ("", ".", "..") or any(char in value for char in "/\\:"):
            raise ValueError(
                f"{value!r} is a path, not a file name: the weights are looked "
                f"for in the models folder (SMC_MODELS)"
            )
        if _is_device_name(value):
            raise ValueError(
                f"{value!r} is a name Windows keeps for a device (con, nul, "
                f"com1, ...): it cannot be a file there"
            )
        return value

    @field_validator("weights_sha256")
    @classmethod
    def _sha256_is_64_lower_case_hex(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError(
                "must be a SHA-256: 64 lower case hexadecimal characters "
                "(`shasum -a 256` prints it)"
            )
        return value

    @field_validator("classes")
    @classmethod
    def _classes_are_distinct_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a model has at least one class")
        if any(not label.strip() for label in value):
            raise ValueError("a class is not an empty name")
        if _BACKGROUND in value:
            raise ValueError(
                f"{_BACKGROUND} is the checkpoint's own first class: list the "
                f"classes after it"
            )
        if len(set(value)) != len(value):
            raise ValueError(f"{list(value)} are not distinct")
        return value

    @field_validator("trained_um_per_px", mode="before")
    @classmethod
    def _unknown_is_none(cls, value: object) -> object:
        return pixel_size_or_word(value, "unknown")

    @field_validator("trained_um_per_px")
    @classmethod
    def _trained_pixel_size_is_plausible(cls, value: float | None) -> float | None:
        return check_pixel_size(value)


def model_names() -> tuple[str, ...]:
    """The names of the cards in the package, sorted; other files of the folder are not cards.

    The folder is read through ``importlib.resources``, never through a path
    built from ``__file__``, so that it is found in a wheel as in a checkout.
    """
    folder = files("smc.inference").joinpath(_CARDS_FOLDER)
    return tuple(
        sorted(
            entry.name.removesuffix(_SUFFIX)
            for entry in folder.iterdir()
            if entry.name.endswith(_SUFFIX) and entry.is_file()
        )
    )


def _read_card_bytes(name: str) -> bytes:
    """One card's file; an unreadable one is a ``ModelError`` that names the card, not a path."""
    try:
        return (
            files("smc.inference").joinpath(_CARDS_FOLDER, name + _SUFFIX).read_bytes()
        )
    except OSError as exc:
        raise ModelError(
            f"{_CARDS_FOLDER}/{name}{_SUFFIX} cannot be read ({exc.strerror or exc})"
        ) from exc


def model_card(name: str) -> ModelCard:
    """The card of the model called ``name``.

    A name that is not a model's, or has no card, is refused **before any file is
    opened**, so that ``"../cells"`` never reaches the file system, and the
    message lists the names there are. Otherwise only that one card is read: a
    bad card does not stop the others.

    Raises:
        ModelError: No such model, or its card cannot be read or is not valid.
    """
    known = model_names()
    if (
        not _NAME.fullmatch(name)
        or len(name) > _MAX_NAME_LENGTH
        or name == _RESERVED_NAME
        or name not in known
    ):
        raise ModelError(
            f"no model named {name!r}; the models are: {', '.join(known) or 'none'} "
            f"(cards in {_WHERE})"
        )
    return parse_card(_read_card_bytes(name), source=f"{_CARDS_FOLDER}/{name}{_SUFFIX}")


def model_cards() -> dict[str, ModelCard]:
    """Every card by name, raising on the first bad one: the repository's test reads them all.

    Raises:
        ModelError: A card cannot be read or is not valid.
    """
    return {name: model_card(name) for name in model_names()}


def parse_card(data: bytes, *, source: str) -> ModelCard:
    """A card from its bytes: UTF-8 (a byte-order mark accepted), TOML, then validated.

    ``source`` is the file as a message names it (``cards/<name>.toml``, never an
    absolute path, FM-50) and the card's ``name`` must be its stem, so that the
    name in a result and the file that defines it cannot drift apart. A
    byte-order mark is accepted because Notepad writes one (FM-80).

    Raises:
        ModelError: The file is not UTF-8 text, not TOML, or not a valid card.
            The message starts with ``source`` and gives the first problem as
            ``<key>: <message>``.
    """
    # The byte-order mark is taken off first, so that the line a message names is the
    # one an editor shows (`utf-8-sig` strips it too, but counts bytes after it).
    body = data.removeprefix(codecs.BOM_UTF8)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = body.count(b"\n", 0, exc.start) + 1
        raise ModelError(
            f"{source} is not UTF-8 text ({exc.reason} on line {line}); "
            f"save it as UTF-8, not ANSI or UTF-16"
        ) from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ModelError(f"{source} is not valid TOML ({exc})") from exc
    try:
        card = ModelCard.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        key = ".".join(str(part) for part in first["loc"]) or "card"
        message = str(first["msg"]).removeprefix("Value error, ")
        more = len(exc.errors()) - 1
        extra = f" (and {more} more)" if more else ""
        raise ModelError(f"{source}: {key}: {message}{extra}") from exc
    stem = source.rsplit("/", 1)[-1].removesuffix(_SUFFIX)
    if card.name != stem:
        raise ModelError(
            f"{source}: name: {card.name!r} is not the file's name {stem!r}; "
            f"a card is named as its file"
        )
    return card
