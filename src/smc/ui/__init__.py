"""The plate page (m3 §15, ADR-0006): a view over plugins and capabilities.

``smc.ui.controller`` holds the page's logic and imports no Panel, so it is
tested everywhere; ``smc.ui.plate_page`` is the Panel layer and the only module
(besides the ``smc ui`` command) that imports ``panel`` or ``bokeh``;
``smc.ui.frames`` is numpy only. The page keeps its M3 names through M5 (m5 §0).
Nothing is imported here, so ``import smc`` and ``smc --help`` work without the
``ui`` extra.
"""
