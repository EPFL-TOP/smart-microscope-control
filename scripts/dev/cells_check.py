"""Check the cell model against frames the owner annotated (m3 section 9.4, #342).

Two commands. ``export`` copies some frames of a finished scan into a COCO folder
that ``nikon-control``'s review page opens, with the model's (or the blob
detector's) boxes as a first draft to correct. ``measure`` reads that folder back
once the owner has marked frames *reviewed*, runs the detectors on them as the
``cells`` stage does, and prints precision and recall per class::

    cells_check.py export local/scans/D/run.json local/cells-check/D --count 15
    nikon-control-dashboard --data-dir local/cells-check --show     # /review: correct, mark reviewed, save
    cells_check.py measure local/cells-check/D --model-um-per-px native,0.35,0.49

Run it with an interpreter that has smc and torch: the stand's PC with the ``detect``
extra, or on a Mac ``../nikon-control/.venv/bin/python`` with ``PYTHONPATH=src``.
The blob detector needs no torch. Output is ASCII and files are UTF-8 (a Windows
console is not), and nothing here moves a stage or opens a stand.

Exit codes: 0 when everything asked ran, 1 when something is explained on stderr
(a folder, a scan, a file) or a configuration could not run, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

import numpy as np
import numpy.typing as npt
import tifffile

from smc.plugins.cells.algorithm import suppress_in_frame
from smc.plugins.cells.detector import (
    BLOB_DETECTOR,
    BLOB_DETECTOR_IDENTITY,
    BlobDetector,
    Detection,
    Detector,
    Preprocessing,
    default_model_path,
    load_detector,
    model_input_size_px,
)
from smc.plugins.cells.evaluate import (
    DEFAULT_MATCH_IOU,
    NO_BOX,
    ClassScore,
    LabelledBox,
    class_scores,
    confusion,
)
from smc.plugins.errors import PluginRunError
from smc.plugins.slots import ScanFrame, ScanResult, load_frame
from smc.run import RunDocument, RunDocumentError, StageStatus
from smc.run.files import resolve_in_run, retry_on_permission_error

#: COCO's category ids, as ``nikon-control`` trains and reviews with them.
CATEGORIES = {"single": 1, "doublet": 2, "debris": 3}
#: The in-frame suppression of the ``cells`` stage and of the prefill.
DEFAULT_MAX_OVERLAP = 0.7
_PROG = "cells_check"
_SPLITS = ("train", "val")
_NATIVE = "native"
_HASH_CHUNK_BYTES = 1 << 20
#: The largest ``model_um_per_px`` the plugin accepts.
_MAX_UM_PER_PX = 10.0
_TRAINED_PERCENTILES = (0.5, 99.5)


class CheckError(Exception):
    """Something the user can fix, told in one line: the script prints it and exits 1."""


def _write(text: str, stream: TextIO) -> None:
    """One line, ASCII whatever it holds: a path with a micro sign must not crash a cp1252 console."""
    stream.write(text.encode("ascii", "backslashreplace").decode("ascii") + "\n")


def _say(text: str = "") -> None:
    _write(text, sys.stdout)


def _fail(text: str) -> None:
    _write(f"{_PROG}: {text}", sys.stderr)


def _dumps(value: Any) -> str:
    """Pretty JSON, ASCII (``nikon-control`` reads it with the platform's encoding) and no ``nan``."""
    return json.dumps(value, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise CheckError(
            f"cannot read {path.name} to hash it ({exc.strerror or exc})"
        ) from exc
    return digest.hexdigest()


# ------------------------------------------------------------------ arguments


def _float_in(name: str, low: float, high: float, *, open_low: bool) -> Any:
    """An argparse type: a finite float in ``(low, high]`` (``[low, high]`` unless ``open_low``)."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
        above = value > low if open_low else value >= low
        if not (math.isfinite(value) and above and value <= high):
            bracket = "(" if open_low else "["
            raise argparse.ArgumentTypeError(
                f"{name} must be in {bracket}{low:g}, {high:g}], got {text}"
            )
        return value

    return parse


def _count(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text}")
    return value


def _sizes(text: str) -> list[float | None]:
    """``native,0.49,...``: the pixel sizes the model is shown, ``None`` for its own resize."""
    out: list[float | None] = []
    for part in text.split(","):
        part = part.strip()
        if part == _NATIVE:
            out.append(None)
            continue
        out.append(
            _float_in("a model pixel size", 0.0, _MAX_UM_PER_PX, open_low=True)(part)
        )
    return out


def _percentile_pairs(text: str) -> list[tuple[float, float]]:
    """``0.5:99.5,1:99``: the intensity percentiles, low then high."""
    out: list[tuple[float, float]] = []
    for part in text.split(","):
        low_text, colon, high_text = part.strip().partition(":")
        try:
            if not colon:
                raise ValueError("write each pair as low:high")
            pair = (float(low_text), float(high_text))
            Preprocessing(percentiles=pair)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{part.strip()!r}: {exc}") from None
        out.append(pair)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Export frames for the owner to annotate, and measure the cell detectors on them.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export",
        help="copy frames of a scan into a COCO folder for nikon-control's /review",
    )
    export.add_argument("run_json", type=Path, help="the run document of the scan")
    export.add_argument("out_dir", type=Path, help="a new or empty folder")
    export.add_argument("--wells", help="comma-separated wells (default: every well)")
    export.add_argument(
        "--count",
        type=_count,
        default=20,
        help="frames kept, evenly spaced (default 20)",
    )
    export.add_argument(
        "--prefill",
        choices=("model", "blobs", "none"),
        default="model",
        help="whose boxes are the first draft to correct (default model)",
    )
    export.add_argument(
        "--prefill-score",
        type=_float_in("--prefill-score", 0.0, 1.0, open_low=True),
        default=0.3,
        help="the lowest score the draft keeps (default 0.3)",
    )
    export.add_argument("--model-path", type=Path, help="the weights (--prefill model)")
    export.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), help="(--prefill model)"
    )
    export.add_argument(
        "--model-um-per-px",
        type=_float_in("--model-um-per-px", 0.0, _MAX_UM_PER_PX, open_low=True),
        help="the pixel size the model sees (--prefill model; default its own resize)",
    )

    measure = commands.add_parser(
        "measure", help="precision and recall per class on the reviewed frames"
    )
    measure.add_argument("dataset_dir", type=Path, help="a folder export made")
    measure.add_argument(
        "--detector",
        action="append",
        choices=("model", "blobs"),
        help="repeat for both (default: blobs and model)",
    )
    measure.add_argument(
        "--model-um-per-px",
        type=_sizes,
        help="comma-separated pixel sizes the model sees; native is its own resize (default native)",
    )
    measure.add_argument(
        "--percentiles",
        type=_percentile_pairs,
        help="comma-separated low:high pairs (default 0.5:99.5)",
    )
    measure.add_argument(
        "--score-threshold",
        type=_float_in("--score-threshold", 0.05, 1.0, open_low=False),
        default=0.5,
    )
    measure.add_argument(
        "--max-overlap",
        type=_float_in("--max-overlap", 0.0, 1.0, open_low=True),
        default=DEFAULT_MAX_OVERLAP,
    )
    measure.add_argument(
        "--match-iou",
        type=_float_in("--match-iou", 0.0, 1.0, open_low=True),
        default=DEFAULT_MATCH_IOU,
    )
    measure.add_argument("--model-path", type=Path, help="the weights")
    measure.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    measure.add_argument("--json", type=Path, help="also write the numbers here")
    return parser


# --------------------------------------------------------------------- export


def _load_scan(run_json: Path) -> ScanResult:
    """The finished scan of a run document, or what is wrong with it in a line."""
    try:
        document = RunDocument.load(run_json)
    except RunDocumentError as exc:
        raise CheckError(str(exc)) from exc
    record = document.provider("scan")
    if record is None:
        raise CheckError(f"{run_json.name} has no scan: run scan first")
    if record.status is not StageStatus.done or record.result is None:
        raise CheckError(
            f"the scan in {run_json.name} is {record.status.value}, not done: "
            f"run scan again (or --resume) before exporting its frames"
        )
    try:
        scan = ScanResult.model_validate(record.result)
    except ValueError as exc:
        raise CheckError(
            f"the scan in {run_json.name} does not validate: {exc}"
        ) from exc
    if not scan.frames:
        raise CheckError(
            f"the scan in {run_json.name} has no frame (a dry run, or every field failed)"
        )
    return scan


def _evenly(total: int, count: int) -> list[int]:
    """``count`` of ``total`` positions evenly spaced: ``round(linspace(0, total - 1, count))``; all when fewer."""
    if count >= total:
        return list(range(total))
    return sorted({round(float(v)) for v in np.linspace(0, total - 1, count)})


def _select(scan: ScanResult, wells: str | None, count: int) -> list[int]:
    """The positions in ``scan.frames`` of the frames to export, in the scan's order."""
    if wells is None:
        wanted = None
    else:
        wanted = [w.strip() for w in wells.split(",") if w.strip()]
        present = list(scan.frame_index())
        unknown = [w for w in wanted if w not in present]
        if unknown or not wanted:
            raise CheckError(
                f"no frame in well {', '.join(unknown) or '(none named)'}: "
                f"the scan has {', '.join(present)}"
            )
    positions = [
        p
        for p, frame in enumerate(scan.frames)
        if wanted is None or frame.well in wanted
    ]
    return [positions[i] for i in _evenly(len(positions), count)]


def _check_out_dir(out_dir: Path) -> None:
    """The folder is looked at first (FM-30): the owner's annotations are never overwritten."""
    if out_dir.exists():
        if not out_dir.is_dir():
            raise CheckError(f"{out_dir} is a file, not a folder")
        if any(out_dir.iterdir()):
            raise CheckError(
                f"{out_dir} is not empty: what is in it is never overwritten (it may "
                f"be the owner's annotations); give a new folder"
            )


def _image_names(frames: Sequence[ScanFrame]) -> list[str]:
    """The name each frame has in ``images/``: the basename of its file, and never twice."""
    names = [PurePosixPath(frame.file).name for frame in frames]
    seen: dict[str, str] = {}
    for name, frame in zip(names, frames, strict=True):
        # casefold: two names that differ in case are one file on NTFS and the default APFS (FM-24)
        first = seen.setdefault(name.casefold(), frame.file)
        if first != frame.file:
            raise CheckError(
                f"{first} and {frame.file} would both be images/{name}: "
                f"export the wells apart"
            )
    return names


def _prefill_detector(
    args: argparse.Namespace, scan: ScanResult
) -> tuple[Detector | None, dict[str, Any]]:
    """The detector that drafts the boxes (or none) and what the manifest says of it."""
    info: dict[str, Any] = {
        "detector": args.prefill,
        "model": None,
        "model_sha256": None,
        "score": args.prefill_score,
        "model_um_per_px": args.model_um_per_px,
    }
    if args.prefill != "model":
        for given, flag in (
            (args.model_path, "--model-path"),
            (args.device, "--device"),
            (args.model_um_per_px, "--model-um-per-px"),
        ):
            if given is not None:
                raise CheckError(f"{flag} is for --prefill model")
        if args.prefill == "blobs":
            info["model"] = BLOB_DETECTOR
            info["model_sha256"] = hashlib.sha256(
                BLOB_DETECTOR_IDENTITY.encode("utf-8")
            ).hexdigest()
            return BlobDetector(), info
        return None, info
    size = None
    if args.model_um_per_px is not None:
        try:
            size = model_input_size_px(
                scan.image_shape_px, scan.pixel_size_um, args.model_um_per_px
            )
        except ValueError as exc:
            raise CheckError(
                f"--model-um-per-px {args.model_um_per_px:g}: {exc}"
            ) from exc
    weights = args.model_path if args.model_path is not None else default_model_path()
    try:
        detector = load_detector(
            weights,
            device=args.device or "auto",
            score_threshold=args.prefill_score,
            preprocessing=Preprocessing(input_size_px=size),
        )
    except PluginRunError as exc:
        raise CheckError(exc.reason) from exc
    info["model"] = weights.name
    info["model_sha256"] = _sha256(weights)
    return detector, info


def _coco_annotation(
    number: int, image_id: int, detection: Detection
) -> dict[str, Any]:
    """A detection as ``nikon-control``'s review page keeps one: ``[x, y, w, h]``, columns first (FM-73)."""
    if detection.label not in CATEGORIES:
        raise CheckError(
            f"the detector gave the label {detection.label!r}; the review page knows "
            f"{', '.join(CATEGORIES)}"
        )
    width = detection.col1_px - detection.col0_px
    height = detection.row1_px - detection.row0_px
    return {
        "id": number,
        "image_id": image_id,
        "category_id": CATEGORIES[detection.label],
        "bbox": [detection.col0_px, detection.row0_px, width, height],
        "area": width * height,
        "iscrowd": 0,
        "cell": None,
        "auto": True,
        "score": detection.score,
    }


def _export(args: argparse.Namespace) -> int:
    run_json: Path = args.run_json
    out_dir: Path = args.out_dir
    _check_out_dir(out_dir)
    scan = _load_scan(run_json)
    frames = [scan.frames[p] for p in _select(scan, args.wells, args.count)]
    names = _image_names(frames)
    run_dir = run_json.parent
    sources: list[Path] = []
    for frame in frames:
        try:
            source = resolve_in_run(run_dir, frame.file)
        except ValueError as exc:
            raise CheckError(f"frame file {frame.file} is not usable: {exc}") from exc
        if not source.is_file():
            raise CheckError(
                f"frame file {frame.file} is missing from the run folder; run scan again"
            )
        sources.append(source)
    detector, prefill = _prefill_detector(args, scan)

    # Everything that can fail is done before the first file is written: a frame
    # that cannot be read leaves no half-made folder for the owner to find.
    annotations: list[dict[str, Any]] = []
    for image_id, frame in enumerate(frames, start=1):
        if detector is None:
            continue
        try:
            found = detector.detect(load_frame(run_dir, frame))
        except (ValueError, OSError, tifffile.TiffFileError) as exc:
            raise CheckError(f"frame file {frame.file} cannot be read ({exc})") from exc
        kept = [
            d
            for d in suppress_in_frame(found, max_overlap=DEFAULT_MAX_OVERLAP)
            if d.score >= args.prefill_score
        ]
        for detection in kept:
            annotations.append(
                _coco_annotation(len(annotations) + 1, image_id, detection)
            )

    height, width = scan.image_shape_px
    coco = {
        "categories": [{"id": i, "name": name} for name, i in CATEGORIES.items()],
        "images": [
            {
                "id": i,
                "file_name": name,
                "height": height,
                "width": width,
                "reviewed": False,
            }
            for i, name in enumerate(names, start=1)
        ],
        "annotations": annotations,
    }
    manifest = {
        "run_document": run_json.name,
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scan": {
            "objective": scan.objective,
            "pixel_size_um": scan.pixel_size_um,
            "image_shape_px": [height, width],
            "exposure_ms": scan.exposure_ms,
        },
        "prefill": prefill,
        "images": [
            {
                "file_name": name,
                "frame_file": frame.file,
                "well": frame.well,
                "x_um": frame.x_um,
                "y_um": frame.y_um,
            }
            for name, frame in zip(names, frames, strict=True)
        ],
    }
    created = not out_dir.exists()
    try:
        _write_dataset(out_dir, sources, frames, names, coco, manifest)
    except CheckError:
        # The same command must be runnable again: a half-made folder would be
        # refused as not empty, and it holds nothing of the owner's yet.
        _take_back(out_dir, created=created)
        raise

    wells = list(dict.fromkeys(frame.well for frame in frames))
    _say(f"exported {len(frames)} frames of {', '.join(wells)} to {out_dir}")
    _say(
        f"prefill: {args.prefill}"
        + (
            f", {len(annotations)} boxes to correct (auto)"
            if detector is not None
            else ""
        )
    )
    _say("next, in the review page:")
    _say(f"  nikon-control-dashboard --data-dir {_quote(out_dir.parent)} --show")
    _say(
        "  open /review, correct the boxes, mark each frame you checked as reviewed, save"
    )
    _say(f"then: python scripts/dev/cells_check.py measure {_quote(out_dir)}")
    return 0


def _quote(path: Path) -> str:
    """A path as the owner pastes it into a console: in double quotes when it holds a space.

    cmd, PowerShell and POSIX shells all read double quotes, and a folder under
    ``C:\\Users\\<first name> <last name>`` is two words without them. A character
    outside ASCII is still shown as an escape (the output is ASCII), so such a path
    is for reading, and the lines are for pasting only where the path is plain.
    """
    text = str(path)
    return f'"{text}"' if any(c.isspace() for c in text) else text


def _write_dataset(
    out_dir: Path,
    sources: Sequence[Path],
    frames: Sequence[ScanFrame],
    names: Sequence[str],
    coco: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Make the folder: the frames copied unchanged, then the two JSON files.

    Raises:
        CheckError: A frame cannot be copied (the message names it, not the folder:
            the file that is held is the source) or a file cannot be written.
    """
    images = out_dir / "images"
    try:
        images.mkdir(parents=True, exist_ok=True)
        (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckError(f"cannot write in {out_dir} ({exc.strerror or exc})") from exc
    for source, name, frame in zip(sources, names, frames, strict=True):
        target = images / name
        try:
            # Antivirus or a sync client may hold a frame the scan wrote a moment ago (FM-24).
            retry_on_permission_error(
                lambda source=source, target=target: shutil.copyfile(source, target)
            )
        except OSError as exc:
            raise CheckError(
                f"cannot copy frame file {frame.file} ({exc.strerror or exc}); "
                f"is another program using it?"
            ) from exc
    try:
        (out_dir / "annotations" / "val.json").write_text(
            _dumps(coco), encoding="utf-8"
        )
        (out_dir / "manifest.json").write_text(_dumps(manifest), encoding="utf-8")
    except OSError as exc:
        raise CheckError(f"cannot write in {out_dir} ({exc.strerror or exc})") from exc


def _take_back(out_dir: Path, *, created: bool) -> None:
    """Remove what a failed export made. The folder was absent or empty when it began, so all of it is ours."""
    for name in ("images", "annotations"):
        shutil.rmtree(out_dir / name, ignore_errors=True)
    with contextlib.suppress(OSError):
        (out_dir / "manifest.json").unlink(missing_ok=True)
    if created:
        with contextlib.suppress(OSError):
            out_dir.rmdir()


# -------------------------------------------------------------------- measure


@dataclass(frozen=True)
class ReviewedImage:
    """A frame the owner marked reviewed, its pixels and the reference boxes (rows first)."""

    name: str
    pixels: npt.NDArray[Any]
    reference: tuple[LabelledBox, ...]


def _read_json(path: Path) -> Any:
    try:
        # utf-8-sig: Notepad before Windows 10 1903 saves "UTF-8" with a byte-order
        # mark, and the owner may edit a file by hand (FM-80). The retry: the review
        # page saved the file a moment ago and antivirus may hold it (FM-24).
        data = retry_on_permission_error(path.read_bytes)
        return json.loads(data.decode("utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise CheckError(f"cannot read {path.name} ({exc})") from exc


def _reviewed_images(dataset: Path) -> tuple[list[ReviewedImage], int, list[str]]:
    """The reviewed images of ``val.json`` and ``train.json``, the number of images, the class names."""
    payloads = {
        split: _read_json(dataset / "annotations" / f"{split}.json")
        for split in _SPLITS
        if (dataset / "annotations" / f"{split}.json").is_file()
    }
    if not payloads:
        raise CheckError(
            f"no annotations/val.json or train.json under {dataset}: "
            f"is it a folder that export made?"
        )
    reviewed: list[ReviewedImage] = []
    total = 0
    classes: dict[int, str] = {}
    for split, payload in payloads.items():
        try:
            names = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
            classes.update(names)
            by_image: dict[int, list[LabelledBox]] = {}
            for a in payload["annotations"]:
                x, y, w, h = (float(v) for v in a["bbox"])
                # [x, y, w, h] is columns first; boxes here are (row0, col0, row1, col1).
                box = LabelledBox((y, x, y + h, x + w), names[int(a["category_id"])])
                by_image.setdefault(int(a["image_id"]), []).append(box)
            for image in payload["images"]:
                total += 1
                if not image.get("reviewed"):
                    continue
                name = str(image["file_name"])
                pixels = _read_image(dataset / "images" / name, name)
                if pixels.shape != (int(image["height"]), int(image["width"])):
                    raise CheckError(
                        f"{name} is {pixels.shape[0]}x{pixels.shape[1]} pixels (rows x columns) "
                        f"and {split}.json says {image['height']}x{image['width']}: "
                        f"the boxes would be read against the wrong axes"
                    )
                reviewed.append(
                    ReviewedImage(
                        name, pixels, tuple(by_image.get(int(image["id"]), []))
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckError(
                f"{split}.json is not a COCO file this script reads ({type(exc).__name__}: {exc})"
            ) from exc
    if not reviewed:
        raise CheckError(
            f"none of the {total} frames is marked reviewed: open the folder in /review, "
            f"correct the boxes, mark the frames you checked as reviewed and save"
        )
    return reviewed, total, [classes[i] for i in sorted(classes)]


def _read_image(path: Path, name: str) -> npt.NDArray[Any]:
    try:
        pixels = np.asarray(retry_on_permission_error(lambda: tifffile.imread(path)))
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise CheckError(f"image {name} cannot be read ({exc})") from exc
    if pixels.ndim != 2:
        raise CheckError(f"image {name} is not a 2-D frame: shape {pixels.shape}")
    return pixels


def _frames_pixel_size(dataset: Path) -> tuple[float, tuple[int, int]] | None:
    """The frames' pixel size and shape from ``manifest.json``, ``None`` without one."""
    path = dataset / "manifest.json"
    if not path.is_file():
        return None
    try:
        scan = _read_json(path)["scan"]
        pixel = float(scan["pixel_size_um"])
        shape = (int(scan["image_shape_px"][0]), int(scan["image_shape_px"][1]))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise CheckError(
            f"manifest.json has no usable scan pixel size and shape ({exc})"
        ) from exc
    if not (math.isfinite(pixel) and pixel > 0.0):
        raise CheckError(
            f"manifest.json gives a pixel size of {pixel}: not a usable one"
        )
    return pixel, shape


@dataclass(frozen=True)
class Configuration:
    """One thing to measure: the blob detector, or the model at a pixel size and percentiles."""

    detector: str
    model_um_per_px: float | None = None
    percentiles: tuple[float, float] = _TRAINED_PERCENTILES


@dataclass
class Outcome:
    """What a configuration gave: its numbers, or why it could not run."""

    configuration: Configuration
    heading: str
    error: str | None = None
    model: str | None = None
    model_sha256_12: str | None = None
    input_size_px: tuple[int, int] | None = None
    frames: int = 0
    first_s: float | None = None
    others_s: float | None = None
    scores: dict[str, ClassScore] | None = None
    any_box: ClassScore | None = None
    counts: Counter[tuple[str, str]] | None = None


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _run_configuration(
    configuration: Configuration,
    args: argparse.Namespace,
    images: Sequence[ReviewedImage],
    classes: Sequence[str],
    frames: tuple[float, tuple[int, int]] | None,
) -> Outcome:
    """Run one detector over the reviewed images as ``cells`` does and score it against the reference."""
    outcome = Outcome(configuration, f"=== {configuration.detector} ===")
    detector: Detector
    if configuration.detector == "blobs":
        detector = BlobDetector()
    else:
        low, high = configuration.percentiles
        weights = (
            args.model_path if args.model_path is not None else default_model_path()
        )
        shown = (
            "own resize (800 px)"
            if configuration.model_um_per_px is None
            else f"{configuration.model_um_per_px:g} um per pixel"
        )
        outcome.heading = (
            f"=== model: {weights.name}, {shown}, percentiles {low:g}:{high:g} ==="
        )
        size: tuple[int, int] | None = None
        try:
            if configuration.model_um_per_px is not None:
                assert frames is not None  # checked before the first configuration
                size = model_input_size_px(
                    frames[1], frames[0], configuration.model_um_per_px
                )
            outcome.input_size_px = size
            detector = load_detector(
                weights,
                device=args.device or "auto",
                score_threshold=args.score_threshold,
                preprocessing=Preprocessing(
                    percentiles=(low, high), input_size_px=size
                ),
            )
            outcome.model = weights.name
            outcome.model_sha256_12 = _sha256(weights)[:12]
        except (PluginRunError, ValueError, CheckError) as exc:
            outcome.error = exc.reason if isinstance(exc, PluginRunError) else str(exc)
            return outcome
        if size is not None:
            outcome.heading = (
                f"=== model: {weights.name} (sha256 {outcome.model_sha256_12}), "
                f"{shown} (input {size[0]}x{size[1]} px), percentiles {low:g}:{high:g} ==="
            )
        else:
            outcome.heading = (
                f"=== model: {weights.name} (sha256 {outcome.model_sha256_12}), "
                f"{shown}, percentiles {low:g}:{high:g} ==="
            )
    counts: Counter[tuple[str, str]] = Counter()
    seconds: list[float] = []
    try:
        for image in images:
            started = time.perf_counter()
            found = detector.detect(image.pixels)
            seconds.append(time.perf_counter() - started)
            kept = suppress_in_frame(found, max_overlap=args.max_overlap)
            detected = [
                LabelledBox((d.row0_px, d.col0_px, d.row1_px, d.col1_px), d.label)
                for d in kept
            ]
            counts += confusion(image.reference, detected, min_iou=args.match_iou)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome
    outcome.frames = len(images)
    outcome.first_s = seconds[0]
    outcome.others_s = sum(seconds[1:]) / len(seconds[1:]) if len(seconds) > 1 else None
    outcome.scores, outcome.any_box = class_scores(counts, classes)
    outcome.counts = counts
    return outcome


def _rank(label: str, classes: Sequence[str]) -> tuple[int, str]:
    if label == NO_BOX:
        return (len(classes) + 1, label)
    return (classes.index(label) if label in classes else len(classes), label)


def _print_outcome(
    outcome: Outcome,
    classes: Sequence[str],
    args: argparse.Namespace,
    images: int,
    total: int,
) -> None:
    _say(outcome.heading)
    if outcome.error is not None:
        _say(f"not run: {outcome.error}")
        _say()
        return
    assert outcome.scores is not None
    assert outcome.any_box is not None
    assert outcome.counts is not None
    _say(
        f"reviewed frames: {images} of {total} | match IoU {args.match_iou:g} | "
        f"score >= {args.score_threshold:g} | max overlap {args.max_overlap:g}"
    )
    warm = "-" if outcome.others_s is None else f"{outcome.others_s:.3f} s"
    _say(f"time per frame: first (cold) {outcome.first_s:.3f} s, others (warm) {warm}")
    rows = [(name, outcome.scores[name]) for name in classes] + [
        ("any box", outcome.any_box)
    ]
    width = max(len(name) for name, _ in rows)
    _say(
        f"{'class':<{width}}  {'reference':>9}  {'detected':>8}  {'matched':>7}  {'precision':>9}  {'recall':>6}"
    )
    for name, score in rows:
        _say(
            f"{name:<{width}}  {score.reference:>9}  {score.detected:>8}  {score.matched:>7}  "
            f"{_ratio(score.precision):>9}  {_ratio(score.recall):>6}"
        )
    _say("confusion (reference -> detected), the pairs that occur:")
    pairs = sorted(
        outcome.counts.items(),
        key=lambda item: (_rank(item[0][0], classes), _rank(item[0][1], classes)),
    )
    for (reference, detected), number in pairs:
        if number:
            _say(f"  {reference} -> {detected}: {number}")
    _say()


def _score_json(score: ClassScore) -> dict[str, Any]:
    return {
        "reference": score.reference,
        "detected": score.detected,
        "matched": score.matched,
        "precision": score.precision,
        "recall": score.recall,
    }


def _outcome_json(outcome: Outcome) -> dict[str, Any]:
    configuration = outcome.configuration
    is_model = configuration.detector == "model"
    return {
        "detector": configuration.detector,
        "model": outcome.model,
        "model_sha256_12": outcome.model_sha256_12,
        "model_um_per_px": configuration.model_um_per_px,
        "input_size_px": list(outcome.input_size_px) if outcome.input_size_px else None,
        "percentiles": list(configuration.percentiles) if is_model else None,
        "error": outcome.error,
        "frames": outcome.frames,
        "first_frame_s": outcome.first_s,
        "mean_other_frames_s": outcome.others_s,
        "classes": (
            {name: _score_json(s) for name, s in outcome.scores.items()}
            if outcome.scores is not None
            else None
        ),
        "any_box": _score_json(outcome.any_box)
        if outcome.any_box is not None
        else None,
        "confusion": (
            [[ref, det, n] for (ref, det), n in outcome.counts.items() if n]
            if outcome.counts is not None
            else None
        ),
    }


def _measure(args: argparse.Namespace) -> int:
    detectors = list(dict.fromkeys(args.detector or ["blobs", "model"]))
    if "model" not in detectors:
        for given, flag in (
            (args.model_um_per_px, "--model-um-per-px"),
            (args.percentiles, "--percentiles"),
            (args.model_path, "--model-path"),
            (args.device, "--device"),
        ):
            if given is not None:
                raise CheckError(f"{flag} is for --detector model")
    sizes: list[float | None] = args.model_um_per_px or [None]
    pairs: list[tuple[float, float]] = args.percentiles or [_TRAINED_PERCENTILES]
    dataset: Path = args.dataset_dir
    images, total, classes = _reviewed_images(dataset)
    frames = _frames_pixel_size(dataset)
    if "model" in detectors and any(s is not None for s in sizes) and frames is None:
        raise CheckError(
            f"{dataset.name}/manifest.json is missing: it holds the frames' pixel size, "
            f"which --model-um-per-px needs (export writes it)"
        )
    if frames is not None:
        for image in images:
            if image.pixels.shape != frames[1]:
                raise CheckError(
                    f"{image.name} is {image.pixels.shape[0]}x{image.pixels.shape[1]} pixels, "
                    f"manifest.json says the scan's frames are {frames[1][0]}x{frames[1][1]}"
                )
    configurations: list[Configuration] = []
    for detector in detectors:
        if detector == "blobs":
            configurations.append(Configuration("blobs"))
        else:
            configurations += [
                Configuration("model", size, pair) for size in sizes for pair in pairs
            ]
    outcomes = []
    for configuration in configurations:
        outcome = _run_configuration(configuration, args, images, classes, frames)
        _print_outcome(outcome, classes, args, len(images), total)
        outcomes.append(outcome)
    if args.json is not None:
        document = {
            "dataset": dataset.name,
            "reviewed_images": len(images),
            "images": total,
            "match_iou": args.match_iou,
            "max_overlap": args.max_overlap,
            "score_threshold": args.score_threshold,
            "classes": classes,
            "configurations": [_outcome_json(o) for o in outcomes],
        }
        try:
            args.json.write_text(_dumps(document), encoding="utf-8")
        except OSError as exc:
            raise CheckError(
                f"cannot write {args.json} ({exc.strerror or exc})"
            ) from exc
    return 1 if any(o.error is not None for o in outcomes) else 0


def main(argv: list[str] | None = None) -> int:
    """Run ``export`` or ``measure``; the exit code is 0, or 1 with the reason on stderr."""
    args = _parser().parse_args(argv)
    try:
        return _export(args) if args.command == "export" else _measure(args)
    except CheckError as exc:
        _fail(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
