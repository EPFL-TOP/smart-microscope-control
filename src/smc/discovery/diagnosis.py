"""What a probe's raw findings usually mean (design m2 §9.4).

An empty device list, a hub that will not initialise and ``Native module failed
to load`` are each a finding whose cause is not in MMCore's text. This module
holds the sentences that say what is likelier, so that the walk, the listing
and the report agree. It is pure: no ``pymmcore_plus``, no pydantic, and the
Nikon stand registry is imported **inside** each function, because
``smc.hardware.backends.nikon`` imports ``smc.discovery`` and a module-level
import here would close a cycle that only one import order trips over.

Everything here is a likelihood, never a verdict: the raw error stays in the
row, and the diagnosis is printed beside it (unknown beats guessed).
"""

from __future__ import annotations

__all__ = [
    "GENERIC_EMPTY_LIST",
    "GENERIC_HUB_CAUSES",
    "empty_list_meaning",
    "error_hint",
    "hub_failure_causes",
    "listing_meaning",
]

GENERIC_EMPTY_LIST = (
    "the adapter lists no devices: if it asks a vendor SDK what is attached, "
    "the SDK could not be reached; otherwise this PC offers none"
)

#: In the order people hit them: the program that holds the device is usually
#: the operator's own, so it comes before any cable.
GENERIC_HUB_CAUSES: tuple[str, ...] = (
    "another program holds the device (the vendor's software, another smc or "
    "pymmcore session); close them and run this again",
    "the device or its controller is off, was powered in the wrong order, or "
    "its cable, USB port or serial port is wrong",
    "the vendor driver or runtime DLL is missing, or is not the version the "
    "adapter was built for (on Windows, `smc discover` shows what the loader "
    "says for an adapter that fails to load)",
)

_NATIVE_MODULE = "native module failed to load"

#: The vendor runtime an adapter wraps, for the one that was measured (#126:
#: ``HamamatsuHam_DCAM`` fails ``Native module failed to load (6)`` where
#: DCAM-API is not installed). No other vendor is named until one is measured.
_VENDOR_RUNTIME = {"hamamatsuham": "DCAM-API"}


def empty_list_meaning(adapter: str) -> str:
    """What an adapter that lists no device most likely says.

    A Nikon adapter has its stand's own text (the Ti2 asks the SDK, so an empty
    list means the SDK could not be reached); every other adapter gets the
    generic sentence.
    """
    from smc.hardware.backends import nikon

    stand = nikon.stand_for_adapter(adapter)
    return stand.empty_list_meaning if stand is not None else GENERIC_EMPTY_LIST


def listing_meaning(adapter: str, device_count: int) -> str | None:
    """What an adapter's device list says about the stand, or ``None``.

    The list is silent when it is long enough to say nothing is wrong. It is
    worth a line when it is empty, and when the adapter fixes its list whatever
    is connected (the Ti-E publishes the same devices attached or not): there a
    list of 18 devices proves nothing, which the plain listing must not let a
    reader forget.
    """
    from smc.hardware.backends import nikon

    if device_count == 0:
        return empty_list_meaning(adapter)
    stand = nikon.stand_for_adapter(adapter)
    if stand is not None and not stand.dynamic:
        return stand.empty_list_meaning
    return None


def hub_failure_causes(adapter: str) -> list[str]:
    """The likely causes of a hub that will not initialise, most likely first.

    For a Nikon adapter, the one-connection rule and then the stand's own list;
    otherwise the three generic ones. A new list each time, so a caller that
    appends to it cannot change the module's constants.
    """
    from smc.hardware.backends import nikon

    stand = nikon.stand_for_adapter(adapter)
    if stand is not None:
        return [nikon.ONE_CONNECTION, *stand.causes]
    return list(GENERIC_HUB_CAUSES)


def error_hint(adapter: str, error: str) -> str | None:
    """What a device error usually means, or ``None`` when nothing is known.

    One entry today: a native module that will not load is the vendor runtime
    the adapter wraps, which is not installed or not the version the adapter
    was built for. ``str.casefold`` never raises on the lone surrogates that
    pymmcore leaves in adapter text (FM-05).
    """
    if _NATIVE_MODULE not in error.casefold():
        return None
    runtime = _VENDOR_RUNTIME.get(adapter.strip().casefold())
    named = f" ({runtime})" if runtime else ""
    return (
        f"the vendor runtime{named} the adapter wraps could not be loaded: it "
        "is not installed on this PC, or is not the version the adapter was "
        "built for"
    )
