"""Vision tests draw synthetic packs as coloured rectangles, whatever data/gallery holds.

The classic detector and the pixel checks are tuned for plain packs; real gallery photos would
make results depend on what's in the gallery folder. Tests that want photos pass their own
gallery folder (test_synth checks pasting that way).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.synth.make as synth
from shelfpulse.config import REPO_ROOT

_REAL_LOAD_GALLERY = synth.load_gallery
_DEFAULT_GALLERY = (REPO_ROOT / "data" / "gallery").resolve()


def _no_default_gallery(root):
    """Ignore the real data/gallery; a gallery folder a test passes explicitly still loads."""
    if Path(root).resolve() in (_DEFAULT_GALLERY, Path("data/gallery").resolve()):
        return {}
    return _REAL_LOAD_GALLERY(root)


@pytest.fixture(autouse=True, scope="session")
def no_gallery():
    # Session scope: module-scoped fixtures that render shelves are built before any
    # function-scoped patch would apply.
    mp = pytest.MonkeyPatch()
    mp.setattr(synth, "load_gallery", _no_default_gallery)
    yield
    mp.undo()


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
