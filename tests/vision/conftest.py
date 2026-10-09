"""Vision tests draw synthetic packs as coloured rectangles, whatever data/gallery holds.

The classic detector and the pixel checks are tuned for plain packs; real gallery photos would
make results depend on what's in the gallery folder. test_synth checks gallery pasting itself.
"""

from __future__ import annotations

import pytest

import tools.synth.make as synth


@pytest.fixture(autouse=True)
def no_gallery(monkeypatch, request):
    if "uses_gallery" not in request.keywords:
        monkeypatch.setattr(synth, "load_gallery", lambda root: {})


@pytest.fixture(autouse=True)
def no_identifier(monkeypatch):
    """Detection tests don't load DINOv2, even if a gallery index was built locally."""
    import shelfpulse.perception.analyze as analyze
    import shelfpulse.perception.run as run

    plain = analyze.load_pipeline

    def load_pipeline(name="perception", backend=None, identify=False):
        return plain(name, backend, identify)

    cached = {}
    monkeypatch.setattr(analyze, "load_pipeline", load_pipeline)
    monkeypatch.setattr(run, "load_pipeline", load_pipeline)
    monkeypatch.setattr(analyze, "default_pipeline", lambda: cached.setdefault(0, load_pipeline()))


def pytest_configure(config):
    config.addinivalue_line("markers", "uses_gallery: let synthetic shelves paste gallery photos")
