"""Report (m3 §13): ``svg`` and ``render`` are pure, ``plugin`` reads the records and writes one file.

Nothing is imported here, so a test of the drawing primitives or of the page
imports ``svg`` or ``render`` alone, and importing the package does not load the
plugin.
"""
