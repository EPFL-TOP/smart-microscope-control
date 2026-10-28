"""Cell detection on a scan's frames (m3 §9): ``algorithm`` is pure, ``detector`` wraps the model, ``plugin`` runs it.

Nothing is imported here, so importing the package never loads torch: the
plugin is listed by ``smc run --list`` on every PC, and torch is imported only
inside ``TorchDetector`` (the ``detect`` extra).
"""
