"""Load the trained pipeline and produce crop predictions.

Used by the Flask API (``app.py``) and available as a command line tool:

    python src/predict.py --json "{\"N\":90,\"P\":42,\"K\":43,\"temperature\":25,\"humidity\":80,\"ph\":6.5,\"rainfall\":200}"
    python src/predict.py --n 90 --p 42 --k 43 --temperature 25 --humidity 80 --ph 6.5 --rainfall 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make sibling modules importable when this file is executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from common import FEATURE_COLUMNS, MODEL_INFO_PATH, MODEL_PATH  # noqa: E402
from crop_info import get_crop_info  # noqa: E402
from validate_data import validate_prediction_payload  # noqa: E402

TRAIN_COMMAND = "python src/train.py"

# Explain honestly what a "score" means for each algorithm family.
SCORE_MEANINGS = {
    "Random Forest": (
        "share of the forest's trees that voted for that crop. It is a model score, "
        "not a calibrated probability that the crop will succeed in the field."
    ),
    "Decision Tree": (
        "fraction of training samples of that crop in the leaf the input reached. "
        "It is a model score, not a calibrated probability of agronomic success."
    ),
    "Support Vector Machine (SVM)": (
        "Platt-scaled decision score produced by SVC(probability=True). It is a model "
        "score, not a calibrated probability of agronomic success."
    ),
}


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained pipeline is missing or cannot be loaded."""


# --------------------------------------------------------------------------- #
# Pipeline loading (cached by file modification time)
# --------------------------------------------------------------------------- #
_CACHE: dict[str, Any] = {"mtime": None, "pipeline": None}


def load_pipeline(path: Path | None = None, use_cache: bool = True):
    """Load the saved scikit-learn pipeline.

    Raises:
        ModelNotAvailableError: if the artifact is missing or unreadable, with
            an explanation of how to create it.
    """
    model_path = Path(path) if path else MODEL_PATH

    if not model_path.is_file():
        raise ModelNotAvailableError(
            f"The trained model file was not found at '{model_path}'. "
            f"Train it first with: {TRAIN_COMMAND}"
        )

    mtime = model_path.stat().st_mtime
    if use_cache and _CACHE["pipeline"] is not None and _CACHE["mtime"] == mtime:
        return _CACHE["pipeline"]

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        raise ModelNotAvailableError(
            f"The model file '{model_path}' exists but could not be loaded ({exc}). "
            f"Re-train it with: {TRAIN_COMMAND}"
        ) from exc

    if not hasattr(pipeline, "predict"):
        raise ModelNotAvailableError(
            f"The file '{model_path}' does not contain a usable scikit-learn pipeline. "
            f"Re-train it with: {TRAIN_COMMAND}"
        )

    _CACHE["mtime"] = mtime
    _CACHE["pipeline"] = pipeline
    return pipeline


def model_display_name(pipeline: Any | None = None) -> str:
    """Human readable name of the deployed algorithm."""
    try:
        if MODEL_INFO_PATH.is_file():
            with open(MODEL_INFO_PATH, encoding="utf-8") as handle:
                name = json.load(handle).get("selected_model")
            if name:
                return str(name)
    except (OSError, json.JSONDecodeError):
        pass

    if pipeline is not None:
        estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
        return type(estimator).__name__
    return "Unknown"


def _score_meaning(name: str) -> str:
    return SCORE_MEANINGS.get(
        name,
        "model score for that class. It is not a calibrated probability of agronomic success.",
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_crop(
    payload: dict[str, Any],
    feature_ranges: dict[str, dict[str, float]] | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the payload, run the trained pipeline and describe the result.

    The features are always passed in ``FEATURE_COLUMNS`` order, which matches
    the column order used during training.

    Returns:
        A dictionary with the predicted crop, model name, optional per-class
        scores and any validation warnings.

    Raises:
        InputValidationError: hard validation failure (bad payload).
        ModelNotAvailableError: the model artifact is missing/unreadable.
    """
    values, warnings = validate_prediction_payload(payload, feature_ranges)
    pipeline = load_pipeline(model_path)

    # Single-row DataFrame guarantees the training-time feature order and names.
    frame = pd.DataFrame([[values[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    prediction = str(pipeline.predict(frame)[0])

    name = model_display_name(pipeline)
    top_scores: list[dict[str, Any]] = []
    scores_available = False

    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(frame)[0]
            classes = [str(c) for c in pipeline.classes_]
            ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
            top_scores = [
                {"crop": crop, "score": round(float(score), 4)} for crop, score in ranked[:3]
            ]
            scores_available = True
        except Exception:  # noqa: BLE001 - scores are optional, never fail a prediction
            top_scores = []
            scores_available = False

    return {
        "predicted_crop": prediction,
        "crop_info": get_crop_info(prediction),
        "model_name": name,
        "input_values": values,
        "warnings": warnings,
        "top_scores": top_scores,
        "scores_available": scores_available,
        "scores_meaning": _score_meaning(name) if scores_available else None,
    }


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a suitable crop with the trained model."
    )
    parser.add_argument("--json", help="Prediction payload as a JSON object.")
    for field in FEATURE_COLUMNS:
        parser.add_argument(f"--{field}", type=float, help=f"Value for {field}.")
    parser.add_argument(
        "--model", help=f"Optional alternative path to a joblib pipeline (default: {MODEL_PATH})."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.json:
        try:
            payload = json.loads(args.json)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON: {exc}", file=sys.stderr)
            return 2
    else:
        payload = {}
        for field in FEATURE_COLUMNS:
            value = getattr(args, field.replace(" ", "_"))
            if value is None:
                print(f"Missing required value --{field}", file=sys.stderr)
                return 2
            payload[field] = value

    try:
        result = predict_crop(payload, model_path=Path(args.model) if args.model else None)
    except Exception as exc:  # noqa: BLE001 - CLI feedback
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
