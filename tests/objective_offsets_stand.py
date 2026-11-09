"""The demo stand with offsets between its objectives, for the tests of m3 §18 (#340).

Not a test module: ``goto-well`` and ``scan-plan`` both open the demo devices on a
profile whose ``[objectives]`` is filled, so that the profile, the facade, the
turret and the plugin are all in the path (FM-124). The values are the design's
worked example: the 20X is the reference, the 10X at ``(5.0, -2.5)`` and the 40X
at ``(-32.5, 9.75)``, which makes the shift 10X -> 40X ``(-37.5, 12.25)``,
anisotropic and far above goto-well's 0.02 µm tolerance (FM-73). The demo
turret's labels are ``Nikon 40X Plan Fluor ELWD``, ``Nikon 10X S Fluor`` (in
place at open), ``Objective-2``, ``Nikon 20X Plan Fluor ELWD``, ``Objective-4``
and ``Objective-5`` (measured on ``Microscope.open(Profile.demo())``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from smc.hardware import Microscope
from smc.hardware.profile import Profile

TEN = "Nikon 10X S Fluor"
TWENTY = "Nikon 20X Plan Fluor ELWD"
FORTY = "Nikon 40X Plan Fluor ELWD"
MEASURED = "2026-11-12, session 3 (#400)"

SHIFT_TEN_TO_FORTY = (-37.5, 12.25)
#: What a run logs and keeps when the offset between two objectives is not measured.
NOT_MEASURED = (
    "the offset between {origin} and {target} is not measured (the profile's "
    "[objectives]): the positions are {origin}'s, not corrected"
)


def profile_with_offsets(*, forty: bool = True) -> Profile:
    """``Profile.demo()`` with the 20X as reference, the 10X measured and the 40X too unless ``forty`` is false."""
    offsets: dict[str, dict[str, object]] = {
        TEN: {"xy_offset_um": [5.0, -2.5], "z_offset_um": -1.25, "measured": MEASURED}
    }
    if forty:
        offsets[FORTY] = {"xy_offset_um": [-32.5, 9.75], "measured": MEASURED}
    return Profile.model_validate(
        {
            **Profile.demo().model_dump(),
            "objectives": {"reference": TWENTY, "offsets": offsets},
        }
    )


@contextmanager
def opened(profile: Profile, *, dry_run: bool = False) -> Iterator[Microscope]:
    """The demo stand on ``profile``, closed afterwards, as ``demo_microscope`` is."""
    microscope = Microscope.open(profile, dry_run=dry_run)
    try:
        yield microscope
    finally:
        microscope.close()
