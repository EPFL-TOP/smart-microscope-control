"""Cell detection on a scan's frames (m3 §9): ``algorithm`` is pure, ``plugin`` runs it, the model is ``smc.inference``'s.

Nothing is imported here, so importing the package never loads torch: the
plugin is listed by ``smc run --list`` on every PC, and torch is imported only
inside ``smc.inference.local.TorchDetector`` (the ``detect`` extra).
"""
