"""The plate page (m3 §15, ADR-0006): a view over plugins and capabilities.

``smc.ui.controller`` holds the page's logic and imports no Panel, so it is
tested everywhere; ``smc.ui.plate_page`` is the Panel layer and the only module
(besides the ``smc ui`` command and the next one) that imports ``panel`` or
``bokeh``; ``smc.ui.image_view`` is the image as a component of its own (m5 §24),
the second module that imports Panel and Bokeh, and the one a notebook and
``smc.analysis`` may import: of ``smc`` it imports ``smc.ui.frames`` alone, which is
numpy only. The page keeps its M3 names through M5 (m5 §0). Nothing is imported
here, so ``import smc`` and ``smc --help`` work without the ``ui`` extra.
"""
