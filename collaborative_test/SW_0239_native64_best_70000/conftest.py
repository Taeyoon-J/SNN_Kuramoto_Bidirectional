"""Keep generated fixtures inside the research artifact directory on every platform."""
from pathlib import Path
import uuid
import pytest


@pytest.fixture
def tmp_path():
    root = Path(__file__).resolve().parents[2]
    path = root / 'collaborative_testing/generated/t239' / uuid.uuid4().hex[:10]
    path.mkdir(parents=True)
    return path
