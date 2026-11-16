"""The Zeiss Axio Observer 7's MTB 2011 components as pymmcore-plus unicore devices.

MTB (Zeiss's hardware service, ``MTBApi.dll``) is reached through pythonnet,
and its components become Python devices in the same ``UniMMCore`` as the
PVCAM camera, so roles, capabilities, ``useq`` MDA and every plugin work
unchanged (ADR-0007, design m4).

The modules, from the bottom up: ``api`` is the narrow seam the devices speak
to (Protocols, value types, ``MTBError``); ``dotnet`` implements it over
``MTBApi.dll``; ``session`` owns the process's one MTB login; ``devices`` holds
the unicore devices a ``.cfg`` names with ``#py pyDevice`` lines.

This package re-exports nothing: importing it must never import pythonnet,
which only ``dotnet`` does, inside its functions (FM-40).
"""
