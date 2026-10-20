"""The pure diagnosis behind a probe report (design m2 §9.4).

A device list that is empty, a hub that will not open and a ``Native module
failed to load`` each have a likelier meaning than the raw text gives. These
tests pin that the stand's own words are used for a Nikon adapter and a
generic sentence otherwise, and that a hint never raises on adapter text.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from smc.discovery import diagnosis
from smc.hardware.backends import nikon


def test_the_empty_list_meaning_is_the_stands_for_a_nikon_adapter_and_generic_otherwise() -> (
    None
):
    assert diagnosis.empty_list_meaning("NikonTi2") == nikon.TI2.empty_list_meaning
    assert diagnosis.empty_list_meaning("NikonTI") == nikon.TIE.empty_list_meaning
    # Windows finds the adapter whatever its case, and so does the registry.
    assert diagnosis.empty_list_meaning("nikonti2") == nikon.TI2.empty_list_meaning
    assert diagnosis.empty_list_meaning("DemoCamera") == diagnosis.GENERIC_EMPTY_LIST
    assert diagnosis.empty_list_meaning("") == diagnosis.GENERIC_EMPTY_LIST
    # The two stands say different things: the test would pass with one text.
    assert nikon.TI2.empty_list_meaning != nikon.TIE.empty_list_meaning
    assert nikon.TI2.empty_list_meaning != diagnosis.GENERIC_EMPTY_LIST


def test_a_fixed_list_has_a_meaning_even_with_devices() -> None:
    # The Ti-E publishes the same 18 devices whatever is connected.
    assert diagnosis.listing_meaning("NikonTI", 18) == nikon.TIE.empty_list_meaning
    # The Ti2 asks the SDK, so two devices say nothing is wrong; same for others.
    assert diagnosis.listing_meaning("NikonTi2", 2) is None
    assert diagnosis.listing_meaning("DemoCamera", 20) is None
    # No device at all is the empty-list meaning, for any adapter.
    assert diagnosis.listing_meaning("NikonTi2", 0) == nikon.TI2.empty_list_meaning
    assert diagnosis.listing_meaning("NikonTI", 0) == nikon.TIE.empty_list_meaning
    assert diagnosis.listing_meaning("DemoCamera", 0) == diagnosis.GENERIC_EMPTY_LIST


def test_hub_failure_causes_are_the_stands_for_nikon_and_the_three_generic_ones_otherwise() -> (
    None
):
    ti2 = diagnosis.hub_failure_causes("NikonTi2")
    assert ti2 == [nikon.ONE_CONNECTION, *nikon.TI2.causes]
    assert diagnosis.hub_failure_causes("NikonTI") == [
        nikon.ONE_CONNECTION,
        *nikon.TIE.causes,
    ]

    generic = diagnosis.hub_failure_causes("HamamatsuHam")
    assert generic == list(diagnosis.GENERIC_HUB_CAUSES)
    assert len(generic) == 3
    # The order is the point (§9.4): another program first, then power and
    # cables, then the vendor runtime.
    assert "another program" in generic[0]
    assert "off" in generic[1]
    assert "driver" in generic[2]
    # A caller may append to what it gets; the module's tuple stays whole.
    generic.append("x")
    assert len(diagnosis.GENERIC_HUB_CAUSES) == 3


def test_an_error_hint_matches_a_fragment_in_any_case_and_no_match_is_none() -> None:
    error = "Native module failed to load (6)"

    other = diagnosis.error_hint("DemoCamera", error)
    assert other is not None
    # Only the runtime that was measured is named (#126); another adapter gets
    # the sentence without a vendor name.
    assert "DCAM" not in other

    hamamatsu = diagnosis.error_hint("HamamatsuHam", error)
    assert hamamatsu is not None
    assert "DCAM-API" in hamamatsu
    assert hamamatsu != other

    assert diagnosis.error_hint("HamamatsuHam", error.upper()) == hamamatsu
    assert diagnosis.error_hint("hamamatsuham", error.lower()) == hamamatsu
    assert diagnosis.error_hint("HamamatsuHam", "Device not found") is None
    assert diagnosis.error_hint("HamamatsuHam", "") is None


def test_a_lone_surrogate_in_an_error_does_not_break_the_hint() -> None:
    # FM-05: pymmcore decodes adapter text with surrogateescape.
    error = "Native module failed to load \udce9 (6)"

    assert diagnosis.error_hint("HamamatsuHam", error) is not None
    assert diagnosis.error_hint("HamamatsuHam", "caf\udce9") is None


@pytest.mark.parametrize(
    "order",
    [
        ("smc.hardware.backends.nikon", "smc.discovery.diagnosis"),
        ("smc.discovery.diagnosis", "smc.hardware.backends.nikon"),
    ],
)
def test_the_nikon_module_and_the_diagnosis_module_import_in_either_order(
    order: tuple[str, str],
) -> None:
    # nikon.py imports smc.discovery, so a module-level import of nikon in
    # diagnosis.py would close a cycle that only one order trips over.
    first, second = order
    code = f"import {first}\nimport {second}\n"

    run = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        # A Windows child writes the ANSI code page, and `-I` ignores
        # PYTHONIOENCODING: a traceback naming an accented profile folder must
        # show as text, not as a UnicodeDecodeError in place of the failure.
        errors="replace",
        timeout=120,
        check=False,
    )

    assert run.returncode == 0, run.stderr
