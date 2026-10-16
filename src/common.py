"""Shared configuration, paths and dataset helpers for the Crop Recommendation System.

Every module (training, prediction, Flask app, tests) imports its paths and
constants from here so that feature ordering and file locations are defined in
exactly one place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

DATASET_FILENAME = "Crop_recommendation.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

MODEL_FILENAME = "crop_pipeline.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"

METRICS_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
DATASET_PROFILE_PATH = REPORTS_DIR / "dataset_profile.json"

# --------------------------------------------------------------------------- #
# Dataset contract
# --------------------------------------------------------------------------- #
# Order matters: this is the exact order in which features are passed to the
# model at inference time, matching the training column order.
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_SCORING = "f1_macro"

APP_NAME = "Crop Recommendation System"
APP_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file (used to version the dataset)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_available() -> bool:
    return DATASET_PATH.is_file()


def model_available() -> bool:
    return MODEL_PATH.is_file()


def metrics_available() -> bool:
    return METRICS_PATH.is_file()


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the crop recommendation CSV.

    Raises:
        FileNotFoundError: if the CSV is not present (with a helpful message).
    """
    csv_path = Path(path) if path else DATASET_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place '{DATASET_FILENAME}' in the "
            f"'{DATA_DIR.name}' folder or run: python src/download_dataset.py"
        )
    return pd.read_csv(csv_path)


def package_versions() -> dict[str, str]:
    """Record library versions used for a training run (reproducibility)."""
    import platform

    import joblib
    import numpy
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
