"""Tests for dataset schema validation and prediction-payload validation."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from common import EXPECTED_COLUMNS, FEATURE_COLUMNS
from validate_data import (
    DatasetValidationError,
    InputValidationError,
    validate_dataset,
    validate_prediction_payload,
)

from conftest import requires_dataset


# --------------------------------------------------------------------------- #
# Dataset validation
# --------------------------------------------------------------------------- #
@requires_dataset
def test_real_dataset_passes_schema_validation(dataset_df):
    profile = validate_dataset(dataset_df, strict=True)

    assert list(dataset_df.columns) == EXPECTED_COLUMNS
    assert profile["rows"] == len(dataset_df)
    assert profile["problems"] == []
    assert profile["duplicate_rows"] == 0
    assert all(count == 0 for count in profile["missing_values"].values())
    assert profile["n_classes"] >= 2
    assert set(profile["classes"]) == set(dataset_df["label"].unique())
    assert sum(profile["class_counts"].values()) == profile["rows"]
    assert set(profile["feature_ranges"]) == set(FEATURE_COLUMNS)


@requires_dataset
def test_dataset_has_no_missing_or_infinite_feature_values(dataset_df):
    for column in FEATURE_COLUMNS:
        series = pd.to_numeric(dataset_df[column], errors="coerce")
        assert series.isna().sum() == 0, f"{column} contains non-numeric values"
        assert all(math.isfinite(float(v)) for v in series), f"{column} contains non-finite values"


def test_validate_dataset_rejects_wrong_columns():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(DatasetValidationError):
        validate_dataset(frame, strict=True)


def test_validate_dataset_reports_missing_values():
    frame = pd.DataFrame({column: [1.0, 2.0] for column in FEATURE_COLUMNS})
    frame["label"] = ["rice", "maize"]
    frame.loc[0, "N"] = None

    profile = validate_dataset(frame, strict=False)
    assert profile["missing_values"]["N"] == 1
    assert profile["problems"]

    with pytest.raises(DatasetValidationError):
        validate_dataset(frame, strict=True)


def test_validate_dataset_reports_non_numeric_features():
    frame = pd.DataFrame({column: [1.0, 2.0] for column in FEATURE_COLUMNS})
    frame["label"] = ["rice", "maize"]
    frame["ph"] = ["acidic", 6.5]

    profile = validate_dataset(frame, strict=False)
    assert any("Non-numeric" in problem for problem in profile["problems"])


# --------------------------------------------------------------------------- #
# Prediction payload validation
# --------------------------------------------------------------------------- #
def test_valid_payload_is_accepted_in_feature_order(valid_payload):
    values, warnings = validate_prediction_payload(valid_payload)

    assert list(values.keys()) == FEATURE_COLUMNS
    assert values["N"] == 90.0
    assert values["ph"] == 6.5
    assert warnings == []


def test_numeric_strings_are_accepted():
    payload = {column: "10" for column in FEATURE_COLUMNS}
    values, _ = validate_prediction_payload(payload)
    assert values == {column: 10.0 for column in FEATURE_COLUMNS}


@pytest.mark.parametrize("field", FEATURE_COLUMNS)
def test_missing_field_is_rejected(field, valid_payload):
    payload = dict(valid_payload)
    payload.pop(field)

    with pytest.raises(InputValidationError) as excinfo:
        validate_prediction_payload(payload)
    assert field in excinfo.value.errors
    assert "required" in excinfo.value.errors[field].lower()


@pytest.mark.parametrize("bad_value", ["", "   ", None])
def test_blank_field_is_rejected(bad_value, valid_payload):
    payload = dict(valid_payload)
    payload["rainfall"] = bad_value

    with pytest.raises(InputValidationError) as excinfo:
        validate_prediction_payload(payload)
    assert "rainfall" in excinfo.value.errors


def test_non_numeric_value_is_rejected_with_clear_message(valid_payload):
    payload = dict(valid_payload)
    payload["N"] = "hello"

    with pytest.raises(InputValidationError) as excinfo:
        validate_prediction_payload(payload)
    message = excinfo.value.errors["N"]
    assert "number" in message.lower()
    assert "Nitrogen" in message


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinite_values_are_rejected(bad_value, valid_payload):
    payload = dict(valid_payload)
    payload["humidity"] = bad_value

    with pytest.raises(InputValidationError) as excinfo:
        validate_prediction_payload(payload)
    assert "finite" in excinfo.value.errors["humidity"].lower()


def test_booleans_are_not_accepted_as_numbers(valid_payload):
    payload = dict(valid_payload)
    payload["P"] = True

    with pytest.raises(InputValidationError):
        validate_prediction_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("humidity", 150),      # above 100 %
        ("humidity", -5),       # below 0 %
        ("ph", 20),             # above the pH scale
        ("ph", -1),             # below the pH scale
        ("rainfall", -1),       # negative rainfall
        ("temperature", -100),  # physically impossible
        ("N", -3),
    ],
)
def test_physically_impossible_values_are_rejected(field, value, valid_payload):
    payload = dict(valid_payload)
    payload[field] = value

    with pytest.raises(InputValidationError) as excinfo:
        validate_prediction_payload(payload)
    assert field in excinfo.value.errors


def test_multiple_errors_are_reported_together(valid_payload):
    payload = dict(valid_payload)
    payload["N"] = "abc"
    payload["ph"] = 99

    with pytest.raises(InputValidationError) as excinfo:
        validate_prediction_payload(payload)
    assert set(excinfo.value.errors) == {"N", "ph"}


def test_values_outside_training_range_warn_but_still_validate(valid_payload):
    ranges = {
        "N": {"min": 0.0, "max": 140.0},
        "rainfall": {"min": 20.0, "max": 300.0},
    }
    payload = dict(valid_payload)
    payload["rainfall"] = 1500  # physically possible, far outside the dataset

    values, warnings = validate_prediction_payload(payload, ranges)
    assert values["rainfall"] == 1500.0
    assert any("training" in warning.lower() for warning in warnings)


def test_values_are_never_silently_modified(valid_payload):
    payload = dict(valid_payload)
    payload["ph"] = 6.5234567
    values, _ = validate_prediction_payload(payload)
    assert values["ph"] == 6.5234567


def test_extra_keys_are_ignored_and_target_cannot_be_injected(valid_payload):
    payload = dict(valid_payload)
    payload["label"] = "rice"  # the target must never be an input
    payload["district"] = "unknown"

    values, _ = validate_prediction_payload(payload)
    assert "label" not in values
    assert list(values.keys()) == FEATURE_COLUMNS


@pytest.mark.parametrize("bad_payload", [None, [], "text", 42])
def test_non_object_payload_is_rejected(bad_payload):
    with pytest.raises(InputValidationError):
        validate_prediction_payload(bad_payload)
