"""Camera-to-stage calibration (m3 §4): ``algorithm`` is pure, ``plugin`` moves the stage.

Nothing is imported here, so importing the package does not load scikit-image
twice over: ``plugin`` pulls ``algorithm`` in, and a test of the arithmetic
imports ``algorithm`` alone.
"""
