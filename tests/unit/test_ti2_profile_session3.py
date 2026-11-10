"""What Ti2-E session 3 measured (2026-11-10, #401), as the shipped ``profiles/nikon-ti2.toml`` holds it.

The session declared the first-lock keys by hand on the stand's copy of the profile, and a
``git checkout -- profiles\\nikon-ti2.toml`` before the next pull would have dropped them: the
first lock from the floor would have been refused again. These tests run the layer on the
*shipped* file, so that what the stand measured is in git and a first lock needs no hand edit.

The Ti2-E fake is the one of ``test_engage_guard.py`` (the plane PFS drives Z to follows the
offset); only what the fake lacks is changed in the profile (its roles, and a shorter settle).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest
from pymmcore_plus import CMMCorePlus
from test_engage_guard import LOCK_TIMEOUT_S, _plane_follows_the_offset, _stand
from test_mm_autofocus import LOCK_S

import smc.hardware.backends.mm as mm
from smc.hardware import Microscope
from smc.hardware.capabilities import ObjectiveTurret
from smc.hardware.errors import SafetyRefusedError
from smc.hardware.profile import Profile, RolesSection
from smc.testing import FakeCore

PROFILES = Path(__file__).resolve().parents[2] / "profiles"
TI2 = PROFILES / "nikon-ti2.toml"
#: Session 3's by-eye focus at the 10x, PFS off (step 5): the plane it focused on.
BY_EYE_Z_UM = 4132.38
FOUR_X, TEN_X, FORTY_X = (
    "1-Plan Fluor 4x",
    "2-Plan Fluor 10x",
    "3-Plan Apo LmbdD0.25 40x",
)


@pytest.fixture(autouse=True)
def short_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Read at call time (m2 §3.6): the profile sets no lock_timeout_s.
    monkeypatch.setattr(mm, "DEFAULT_LOCK_TIMEOUT_S", LOCK_S)


def _shipped() -> Profile:
    """The shipped Ti2-E profile, on the fake's devices.

    The fake has no ``FilterTurret2``, so the role overrides are dropped (the roles
    resolve from what the fake has), and the settle is 0.05 s instead of 1 s: only
    timing differs from the file; every limit and every key is the shipped one.
    """
    shipped = Profile.load(TI2)
    return shipped.model_copy(
        update={
            "roles": RolesSection(),
            "autofocus": shipped.autofocus.model_copy(
                update={"settle_s": 0.05, "lock_timeout_s": LOCK_TIMEOUT_S}
            ),
        }
    )


def _nosepiece_labels() -> list[str]:
    text = (PROFILES / "nikon-ti2.cfg").read_text(encoding="utf-8")
    return re.findall(r"^Label,Nosepiece,\d+,(.+)$", text, flags=re.MULTILINE)


def test_the_profile_declares_the_first_lock_keys_the_stand_measured() -> None:
    loaded = Profile.load(TI2)

    # #98: seen from the side, a falling Z lowers the nosepiece, away from the sample.
    assert loaded.safety.z_away_from_sample == "decreasing"
    # 0 to 20000 read back as written; 30000 read back 24370, where the stand clamps.
    assert loaded.safety.autofocus_offset_soft_limits == (0.0, 20000.0)


def test_a_first_lock_from_the_floor_runs_on_the_shipped_profile_alone() -> None:
    profile = _shipped()
    assert profile.safety.autofocus_offset_soft_limits is not None
    floor = profile.safety.autofocus_offset_soft_limits[0]

    with _stand(profile, offset=floor, z_um=BY_EYE_Z_UM) as stand:
        _plane_follows_the_offset(stand)
        assert stand.af.engage() is True
        # Away from the sample by more than the 50 µm a symmetric band would allow: the
        # one-sided band of a declared direction is what lets the drop through (m2 §12.10 D5).
        assert BY_EYE_Z_UM - stand.z.position_um() > 1000.0


@pytest.mark.parametrize("key", ["autofocus_offset_soft_limits", "z_away_from_sample"])
def test_a_first_lock_is_refused_when_the_profile_lacks_a_key_session_3_declared(
    key: str,
) -> None:
    # Without either key the stand refuses a first lock and names the key: the state of
    # a profile whose hand edit was dropped by a checkout. It proves the test above can fail.
    profile = _shipped()
    bare = profile.model_copy(
        update={"safety": profile.safety.model_copy(update={key: None})}
    )

    with _stand(bare, offset=0.0, z_um=BY_EYE_Z_UM) as stand:
        _plane_follows_the_offset(stand)
        with pytest.raises(SafetyRefusedError, match=re.escape(f"[safety] {key}")):
            stand.af.engage()
        assert stand.core.log == []


def test_every_objective_the_profile_names_is_a_label_of_the_nosepiece() -> None:
    # A label that is not the stand's own is an objective that is unknown, silently.
    labels = _nosepiece_labels()
    assert labels[:3] == [FOUR_X, TEN_X, FORTY_X]

    loaded = Profile.load(TI2)
    named = {*loaded.camera.pixel_size_um, *loaded.autofocus.offset_per_um}
    assert named <= set(labels), sorted(named - set(labels))


@pytest.mark.parametrize(
    ("label", "size_um"),
    [(FOUR_X, 2.73909), (TEN_X, 1.12447), (FORTY_X, 0.276136)],
)
def test_the_turret_gives_the_pixel_size_session_3_calibrated(
    label: str, size_um: float
) -> None:
    # camera-calibration with the lever at 1.0x and binning 1, each at its own step (900,
    # 350 and 100 µm); the fake has no magnifier, so the factor is 1.0 and nothing divides.
    core = FakeCore.demo_like()
    microscope = Microscope.from_core(cast("CMMCorePlus", core), _shipped())
    try:
        turret = microscope.require(ObjectiveTurret)
        assert turret.pixel_size_um(label) == size_um
        # The positions nobody calibrated stay unknown, never a nominal size.
        assert turret.pixel_size_um("4-") == 0.0
    finally:
        microscope.close()
