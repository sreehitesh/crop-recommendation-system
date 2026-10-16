"""Shared pytest fixtures for the Crop Recommendation System test suite.

The tests exercise the real modules. Dataset-dependent tests are skipped with a
clear message when the CSV is missing, and model-dependent tests are skipped
when the trained artifact has not been produced yet, so the suite always reports
honest results instead of pretending to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import app as app_module  # noqa: E402
from common import DATASET_PATH, METRICS_PATH, MODEL_PATH  # noqa: E402

requires_dataset = pytest.mark.skipif(
    not DATASET_PATH.is_file(), reason=f"dataset not found at {DATASET_PATH}"
)
requires_model = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason=f"trained model not found at {MODEL_PATH} (run python src/train.py)"
)
requires_metrics = pytest.mark.skipif(
    not METRICS_PATH.is_file(), reason=f"metrics not found at {METRICS_PATH} (run python src/train.py)"
)


@pytest.fixture(scope="session")
def flask_app():
    app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    return app_module.app


@pytest.fixture()
def client(flask_app):
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="session")
def valid_payload() -> dict[str, float]:
    """The documentation example values (not a guaranteed prediction)."""
    return {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 25,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 200,
    }


@pytest.fixture(scope="session")
def dataset_df():
    from common import load_dataset

    if not DATASET_PATH.is_file():
        pytest.skip("dataset not available")
    return load_dataset()
