---
name: new-backend
description: Scaffold a hardware bridge for a stand that Micro-Manager cannot reach natively (e.g. Zeiss MTB, Viventis PyMCS) as pymmcore-plus unicore Python devices, plus profile, contract tests and hardware-session checklist. Argument: vendor slug.
---

# /new-backend <vendor>

Read ADR-0002, ADR-0003 and ADR-0007 first. Prefer **unicore Python
devices** so the stand lives in the same core as its camera; fall back to a
direct capability implementation only for what unicore cannot express.

Create:

```
src/smc/hardware/backends/<vendor>/
  __init__.py
  session.py       owns the single vendor connection (login once, share, close at exit)
  devices.py       unicore devices: XYStageDevice / StageDevice / StateDevice / ShutterDevice / GenericDevice
  autofocus.py     Autofocus capability if the stand has one (until unicore has an autofocus type)
profiles/<vendor>-<stand>.toml
tests/backends/test_<vendor>_devices.py     against a fake of the vendor API
tests/contracts/                             the existing suite runs against the new devices
docs/hardware/inventory.md                   new section or additions
```

Rules:
- The vendor SDK is imported lazily and only inside the backend package;
  the rest of the project must import without it (Windows-only .NET, etc.).
- Every quirk learned at the microscope becomes a comment **and** a test.
- Wrap vendor exceptions into `smc.hardware` errors with a diagnosis
  (what to check, in order of likelihood).
- Add the `[<vendor>]` optional-dependency extra to `pyproject.toml`.
- Plan the first hardware session with `/hardware-session`.
