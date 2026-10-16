"""Dataset schema validation and user-input validation.

Two clearly separated jobs live here:

1. ``validate_dataset`` - checks the training CSV (columns, types, missing
   values, duplicates, class balance) and returns a factual profile.
2. ``validate_prediction_payload`` - validates a JSON payload coming from the
   web form before it is handed to the model.

Validation policy for user input (documented in docs/USER_GUIDE.md):

* Hard rejection (HTTP 400) for missing fields, non-numeric text, NaN/Infinity
  and values outside *physically meaningful* limits (for example pH outside
  0-14, or relative humidity outside 0-100 %).
* Soft warning (HTTP 200 with a ``warnings`` list) when a value is numeric and
  physically possible but falls outside the range observed in the training
  dataset. Dataset minimum/maximum values are NOT treated as universal
  agricultural limits - they only mark where the model has no example data.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from common import EXPECTED_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN

# --------------------------------------------------------------------------- #
# Feature metadata used by the API, the UI and the validation layer
# --------------------------------------------------------------------------- #
FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "N": {
        "label": "Nitrogen (N)",
        "unit": "soil nitrogen content (dataset documents this as a ratio/index)",
        "hard_min": 0.0,
        "hard_max": 1000.0,
        "help": "Nitrogen content reported for the soil sample.",
        "group": "soil",
    },
    "P": {
        "label": "Phosphorus (P)",
        "unit": "soil phosphorus content (dataset documents this as a ratio/index)",
        "hard_min": 0.0,
        "hard_max": 1000.0,
        "help": "Phosphorus content reported for the soil sample.",
        "group": "soil",
    },
    "K": {
        "label": "Potassium (K)",
        "unit": "soil potassium content (dataset documents this as a ratio/index)",
        "hard_min": 0.0,
        "hard_max": 1000.0,
        "help": "Potassium content reported for the soil sample.",
        "group": "soil",
    },
    "temperature": {
        "label": "Temperature",
        "unit": "°C",
        "hard_min": -50.0,
        "hard_max": 60.0,
        "help": "Average air temperature while the crop is grown.",
        "group": "environment",
    },
    "humidity": {
        "label": "Relative humidity",
        "unit": "%",
        "hard_min": 0.0,
        "hard_max": 100.0,
        "help": "Relative humidity of the air. Must be between 0 and 100 %.",
        "group": "environment",
    },
    "ph": {
        "label": "Soil pH",
        "unit": "pH scale (0-14)",
        "hard_min": 0.0,
        "hard_max": 14.0,
        "help": "Soil acidity/alkalinity on the pH scale.",
        "group": "soil",
    },
    "rainfall": {
        "label": "Rainfall",
        "unit": "mm (period not stated in the dataset documentation)",
        "hard_min": 0.0,
        "hard_max": 20000.0,
        "help": "Rainfall value in millimetres, as measured the same way as the training data.",
        "group": "environment",
    },
}


class DatasetValidationError(ValueError):
    """Raised when the training CSV does not match the expected schema."""


class InputValidationError(ValueError):
    """Raised when a prediction payload fails hard validation."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


# --------------------------------------------------------------------------- #
# 1. Dataset validation
# --------------------------------------------------------------------------- #
def validate_dataset(df: pd.DataFrame, strict: bool = True) -> dict[str, Any]:
    """Validate the dataset schema and return a factual profile dictionary.

    Args:
        df: dataframe read from the crop recommendation CSV.
        strict: when True, raise ``DatasetValidationError`` if the schema or
            data quality is not acceptable for training.
    """
    problems: list[str] = []

    columns = list(df.columns)
    if columns != EXPECTED_COLUMNS:
        problems.append(
            f"Unexpected columns. Expected {EXPECTED_COLUMNS} but found {columns}."
        )

    if len(df) == 0:
        problems.append("Dataset contains no rows.")

    missing = {col: int(df[col].isna().sum()) for col in df.columns if col in df.columns}
    if any(missing.values()):
        problems.append(f"Missing values detected: {missing}")

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    non_numeric: list[str] = []
    infinite: dict[str, int] = {}
    for col in feature_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            non_numeric.append(col)
        series = coerced.replace([math.inf, -math.inf], None).dropna()
        inf_count = int((~series.apply(math.isfinite)).sum())
        if inf_count:
            infinite[col] = inf_count
    if non_numeric:
        problems.append(f"Non-numeric values in feature columns: {non_numeric}")
    if infinite:
        problems.append(f"Infinite values detected: {infinite}")

    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].isna().any():
            problems.append("Target column contains missing labels.")
        if df[TARGET_COLUMN].nunique() < 2:
            problems.append("Target column needs at least two classes.")
    else:
        problems.append(f"Target column '{TARGET_COLUMN}' is missing.")

    duplicate_rows = int(df.duplicated().sum())

    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": columns,
        "missing_values": missing,
        "duplicate_rows": duplicate_rows,
        "problems": problems,
    }

    if not problems and feature_cols and TARGET_COLUMN in df.columns:
        class_counts = (
            df[TARGET_COLUMN].value_counts().sort_index().to_dict()
        )
        profile.update(
            {
                "n_classes": int(df[TARGET_COLUMN].nunique()),
                "classes": sorted(str(c) for c in df[TARGET_COLUMN].unique()),
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
                "feature_ranges": {
                    col: {
                        "min": float(pd.to_numeric(df[col]).min()),
                        "max": float(pd.to_numeric(df[col]).max()),
                    }
                    for col in feature_cols
                },
            }
        )

    if problems and strict:
        raise DatasetValidationError("Dataset validation failed:\n- " + "\n- ".join(problems))

    return profile


# --------------------------------------------------------------------------- #
# 2. Prediction payload validation
# --------------------------------------------------------------------------- #
def validate_prediction_payload(
    payload: Any, feature_ranges: dict[str, dict[str, float]] | None = None
) -> tuple[dict[str, float], list[str]]:
    """Validate one prediction request.

    Args:
        payload: the decoded JSON body (must be a mapping).
        feature_ranges: optional ``{feature: {"min": .., "max": ..}}`` taken from
            the training dataset; used to emit *warnings* only.

    Returns:
        (values, warnings) - ``values`` keeps ``FEATURE_COLUMNS`` ordering.

    Raises:
        InputValidationError: with a ``{field: message}`` mapping.
    """
    errors: dict[str, str] = {}
    warnings: list[str] = []

    if not isinstance(payload, dict):
        raise InputValidationError({"_request": "Request body must be a JSON object."})

    values: dict[str, float] = {}

    for field in FEATURE_COLUMNS:
        spec = FEATURE_SPECS[field]
        label = spec["label"]

        if field not in payload:
            errors[field] = f"{label} is required."
            continue

        raw = payload[field]

        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors[field] = f"{label} is required and cannot be empty."
            continue

        # Strings are only accepted if they look like a plain decimal number.
        if isinstance(raw, bool):
            errors[field] = f"{label} must be a number."
            continue

        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[field] = f"{label} must be a number (received {raw!r})."
            continue

        if math.isnan(value) or math.isinf(value):
            errors[field] = f"{label} must be a finite number (NaN/Infinity is not allowed)."
            continue

        if value < spec["hard_min"] or value > spec["hard_max"]:
            errors[field] = (
                f"{label} must be between {spec['hard_min']:g} and {spec['hard_max']:g} "
                f"{spec['unit'].split(' ')[0]}. Received {value:g}."
            )
            continue

        if feature_ranges and field in feature_ranges:
            observed_min = feature_ranges[field]["min"]
            observed_max = feature_ranges[field]["max"]
            if value < observed_min or value > observed_max:
                warnings.append(
                    f"{label} ({value:g}) is outside the range seen in the training "
                    f"data ({observed_min:g} to {observed_max:g}). The prediction may be "
                    "less reliable for this value - the dataset range is not an "
                    "agricultural limit, it only shows where the model has examples."
                )

        values[field] = value

    if errors:
        raise InputValidationError(errors)

    # Guarantee deterministic feature ordering for the model.
    ordered = {field: values[field] for field in FEATURE_COLUMNS}
    return ordered, warnings
