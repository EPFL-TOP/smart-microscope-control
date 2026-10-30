"""The scan's pure part: which fields are left, where each frame goes, how much disk, what focus meant (m3 §8, #213).

Nothing here touches a stand. The paths are what a run folder copied between
PCs must survive (FM-24): a part that could name another folder, a drive or a
parent is refused before any file is written. The frames are not square
(FM-73), so a transposed encode is seen.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import tifffile

from smc.plugins.scan.algorithm import (
    CALIBRATION_TOLERANCE,
    PROFILE_TOLERANCE,
    RESERVED_NAMES,
    bytes_needed,
    collision_key,
    encode_frame,
    focus_state,
    frame_path,
    itemsize_of,
    landed_off_target,
    landing_error_um,
    landing_tolerance_um,
    size_differs,
    wells_to_scan,
)
from smc.plugins.slots import PlannedField, ScanFrame

TOKEN = "0123456789ab"


def _field(well: str, index: int) -> PlannedField:
    return PlannedField(
        well=well,
        index=index,
        name=f"{well}_{index:04d}",
        x_um=float(index),
        y_um=-float(index),
    )


def _frame(file: str = "scan/0123456789ab/A1/A1_0000.tif") -> ScanFrame:
    return ScanFrame(
        well="A1",
        index=0,
        name="A1_0000",
        file=file,
        x_um=12.5,
        y_um=-3.25,
        z_um=None,
        focus="none",
        pfs_offset=None,
        t_s=0.75,
    )


# ------------------------------------------------------------- the fields left


def test_fields_are_grouped_by_well_in_visiting_order_skipping_wells_done() -> None:
    # The plan's fields are in visiting order already; the wells' order is the plan's.
    fields = [_field("A1", 0), _field("A1", 1), _field("A2", 0), _field("B2", 0)]
    fields += [_field("B2", 1)]

    groups = wells_to_scan(("A1", "A2", "B2"), fields, done={"A2"})

    assert groups == [
        ("A1", (fields[0], fields[1])),
        ("B2", (fields[3], fields[4])),
    ]
    # The visiting order is the wells', not the order fields come in.
    reordered = wells_to_scan(("B2", "A1"), fields, done=())
    assert [well for well, _ in reordered] == ["B2", "A1"]
    assert reordered[0][1] == (fields[3], fields[4])


def test_a_well_with_no_field_is_kept_with_an_empty_group() -> None:
    """It is marked done like any other, so a resume does not visit it for nothing."""
    fields = [_field("A1", 0)]

    assert wells_to_scan(("A1", "A2"), fields, done=()) == [
        ("A1", (fields[0],)),
        ("A2", ()),
    ]


# ------------------------------------------------------------------- the paths


def test_a_frame_path_is_posix_and_relative() -> None:
    assert frame_path(TOKEN, "A1", "A1_0003") == "scan/0123456789ab/A1/A1_0003.tif"


@pytest.mark.parametrize(
    "bad", ["..", "a/b", "a\\b", "C:", "", "A1 ", ".", "A1.tif", "é1"]
)
def test_a_frame_path_refuses_a_part_that_could_leave_its_folder(bad: str) -> None:
    with pytest.raises(ValueError, match="cannot be a frame's"):
        frame_path(TOKEN, bad, "A1_0000")
    with pytest.raises(ValueError, match="cannot be a frame's"):
        frame_path(TOKEN, "A1", bad)


@pytest.mark.parametrize("device", ["CON", "nul", "Com1", "LPT0", "aux", "PRN", "com0"])
def test_a_frame_path_refuses_a_windows_device_name_in_any_case(device: str) -> None:
    """FM-24: ``NUL.tif`` is the null device on Windows, whatever the case."""
    quoted = repr(device)
    with pytest.raises(ValueError, match="a Windows device name") as as_well:
        frame_path(TOKEN, device, "A1_0000")
    with pytest.raises(ValueError, match="a Windows device name") as as_name:
        frame_path(TOKEN, "A1", device)

    assert str(as_well.value) == (
        f"{quoted} is a Windows device name and cannot be a frame's well in a file path"
    )
    assert str(as_name.value) == (
        f"{quoted} is a Windows device name and cannot be a frame's field name in a "
        "file path"
    )


@pytest.mark.parametrize(
    "name", ["COM10", "CONSOLE", "NULL", "AUX1", "LPT", "COM", "A1"]
)
def test_only_the_whole_part_is_a_device_name(name: str) -> None:
    assert frame_path(TOKEN, name, name) == f"scan/{TOKEN}/{name}/{name}.tif"


def test_every_device_name_of_the_list_is_refused_and_it_is_the_whole_list() -> None:
    expected = {"CON", "PRN", "AUX", "NUL"}
    expected |= {f"COM{n}" for n in range(10)} | {f"LPT{n}" for n in range(10)}

    assert expected == RESERVED_NAMES
    for device in sorted(expected):
        with pytest.raises(ValueError, match="is a Windows device name"):
            frame_path(TOKEN, "A1", device)


# ----------------------------------------------------------- the landing and names


def test_the_landing_error_is_the_distance_of_both_axes() -> None:
    assert landing_error_um((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert landing_error_um((10.0, -5.0), (10.0, -5.0)) == 0.0
    # Both axes count, in either direction.
    assert landing_error_um((1.0, 1.0), (-2.0, -3.0)) == 5.0


def test_the_landing_tolerance_is_a_fraction_of_the_smaller_side() -> None:
    """FM-73: the frame is not square; the larger side would let a field land short."""
    assert landing_tolerance_um((128.0, 96.0), 0.05) == pytest.approx(4.8)
    assert landing_tolerance_um((96.0, 128.0), 0.05) == pytest.approx(4.8)
    assert landing_tolerance_um((128.0, 96.0), 0.5) == pytest.approx(48.0)


def test_a_landing_that_is_not_a_number_is_off_target() -> None:
    """FM-104: ``nan <= tolerance`` is False, so a test written as ``>`` lets it pass."""
    assert landed_off_target(math.nan, 4.8)
    assert landed_off_target(math.inf, 4.8)
    assert not landed_off_target(4.8, 4.8)  # the tolerance itself is a landing
    assert landed_off_target(4.9, 4.8)
    assert not landed_off_target(0.0, 4.8)


def test_a_collision_key_ignores_case() -> None:
    """NTFS and the default APFS do not tell ``A1_0001`` from ``a1_0001``."""
    assert collision_key("A1", "A1_0001") == collision_key("a1", "a1_0001")
    assert collision_key("A1", "A1_0001") == ("a1", "a1_0001")
    assert collision_key("A1", "A1_0001") != collision_key("A1", "A1_0002")
    assert collision_key("A1", "X") != collision_key("A2", "X")


# -------------------------------------------------------------------- the disk


def test_the_disk_need_is_a_tenth_more_than_the_frames() -> None:
    """305 fields of the Ti2-E's assumed 1412 x 1412 camera at 16 bits: about 1.34 GB."""
    assert bytes_needed(305, (1412, 1412), 2) == math.ceil(1.1 * 305 * 1412**2 * 2)
    assert bytes_needed(305, (1412, 1412), 2) == pytest.approx(1.3378e9, rel=1e-4)
    # Height and width both count, and the itemsize.
    assert bytes_needed(1, (96, 128), 1) == math.ceil(1.1 * 96 * 128)
    assert bytes_needed(0, (96, 128), 2) == 0


@pytest.mark.parametrize(
    ("bits", "itemsize"), [(1, 1), (8, 1), (9, 2), (12, 2), (16, 2), (17, 4), (32, 4)]
)
def test_itemsize_follows_the_bit_depth(bits: int, itemsize: int) -> None:
    assert itemsize_of(bits) == itemsize


# --------------------------------------------------------------- the pixel size


def test_a_size_differs_beyond_the_tolerance_only() -> None:
    assert PROFILE_TOLERANCE == 0.01
    assert CALIBRATION_TOLERANCE == 0.10
    assert not size_differs(1.005, 1.0, 0.01)
    assert not size_differs(0.995, 1.0, 0.01)
    assert size_differs(1.015, 1.0, 0.01)
    assert size_differs(0.985, 1.0, 0.01)
    # Relative to the reference: the magnifier lever at 1.5x.
    assert size_differs(1 / 1.5, 1.0, 0.10)
    assert not size_differs(1.0, 1.03, 0.10)


# -------------------------------------------------------------------- the focus


def test_the_focus_state_names_what_the_scan_did() -> None:
    assert focus_state(False, False, False) == "none"
    assert focus_state(True, False, False) == "off"
    assert focus_state(True, False, True) == "off"  # locked, but not the scan's doing
    assert focus_state(True, True, True) == "locked"
    assert focus_state(True, True, False) == "unlocked"


# -------------------------------------------------------------------- the frame


@pytest.mark.parametrize("dtype", [np.uint16, np.uint8])
def test_an_encoded_frame_reads_back_with_its_dtype_and_metadata(
    dtype: type[np.unsignedinteger],
) -> None:
    pixels = (np.arange(96 * 128) % 251).astype(dtype).reshape(96, 128)
    meta = _frame()

    data = encode_frame(pixels, meta)

    with tifffile.TiffFile(io.BytesIO(data)) as tiff:
        back = tiff.asarray()
        stored = tiff.shaped_metadata
    assert back.dtype == dtype
    assert back.shape == (96, 128)
    np.testing.assert_array_equal(back, pixels)
    assert stored is not None
    written = dict(stored[0])
    written.pop("shape")  # tifffile's own key
    assert written == meta.model_dump(mode="json")
