"""Tests for model loading and the prediction layer.

These tests use the genuinely trained artifact in ``models/crop_pipeline.joblib``.
A hardcoded prediction is never substituted for the real model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import predict as predict_module
from common import FEATURE_COLUMNS, METRICS_PATH, MODEL_INFO_PATH, MODEL_PATH
from predict import (
    ModelNotAvailableError,
    load_pipeline,
    model_display_name,
    predict_crop,
)
from validate_data import InputValidationError

from conftest import requires_metrics, requires_model


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
@requires_model
def test_model_artifact_exists_and_is_non_empty():
    assert MODEL_PATH.is_file()
    assert MODEL_PATH.stat().st_size > 0


@requires_model
def test_pipeline_loads_and_exposes_predict():
    pipeline = load_pipeline()
    assert hasattr(pipeline, "predict")
    assert getattr(pipeline, "n_features_in_", len(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)


@requires_model
def test_pipeline_was_fitted_on_the_documented_feature_order():
    pipeline = load_pipeline()
    names = getattr(pipeline, "feature_names_in_", None)
    if names is not None:
        assert [str(name) for name in names] == FEATURE_COLUMNS


@requires_model
@requires_metrics
def test_model_info_matches_metrics_file():
    with open(MODEL_INFO_PATH, encoding="utf-8") as handle:
        info = json.load(handle)
    with open(METRICS_PATH, encoding="utf-8") as handle:
        metrics = json.load(handle)

    assert info["selected_model"] == metrics["selected_model"]["name"]
    assert info["dataset_sha256"] == metrics["dataset"]["sha256"]
    assert info["feature_columns"] == FEATURE_COLUMNS
    assert info["target_column"] == "label"
    assert pytest.approx(info["test_accuracy"], rel=1e-9) == metrics["selected_model"]["test"]["accuracy"]


@requires_model
def test_model_display_name_is_reported():
    assert model_display_name(load_pipeline()) in {
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
    }


def test_missing_model_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.joblib"

    with pytest.raises(ModelNotAvailableError) as excinfo:
        load_pipeline(missing)
    message = str(excinfo.value)
    assert "not found" in message.lower()
    assert "src/train.py" in message


def test_corrupt_model_file_raises_helpful_error(tmp_path):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"this is not a joblib artifact")

    with pytest.raises(ModelNotAvailableError) as excinfo:
        load_pipeline(corrupt)
    assert "train" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- #
# Prediction behaviour
# --------------------------------------------------------------------------- #
@requires_model
def test_prediction_returns_a_label_from_the_trained_model(valid_payload):
    result = predict_crop(valid_payload)

    assert isinstance(result["predicted_crop"], str)
    assert result["predicted_crop"].strip() != ""
    assert result["input_values"] == {k: float(v) for k, v in valid_payload.items()}
    assert list(result["input_values"].keys()) == FEATURE_COLUMNS
    assert result["model_name"]


@requires_model
def test_prediction_matches_a_direct_pipeline_call(valid_payload):
    """The service must not fabricate, reorder or post-process the label."""
    import pandas as pd

    pipeline = load_pipeline()
    frame = pd.DataFrame([[valid_payload[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    expected = str(pipeline.predict(frame)[0])

    assert predict_crop(valid_payload)["predicted_crop"] == expected


@requires_model
def test_predicted_label_exists_in_the_dataset_classes():
    from common import load_dataset

    result = predict_crop(
        {"N": 90, "P": 42, "K": 43, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 200}
    )
    dataset_classes = set(load_dataset()["label"].astype(str))
    assert result["predicted_crop"] in dataset_classes


@requires_model
def test_scores_are_reported_when_the_model_supports_them(valid_payload):
    result = predict_crop(valid_payload)
    if not result["scores_available"]:
        pytest.skip("the deployed algorithm does not expose class scores")

    assert 1 <= len(result["top_scores"]) <= 3
    scores = [entry["score"] for entry in result["top_scores"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert sum(scores) <= 1.000001
    # The highest-scoring class must be the label the pipeline predicted.
    assert result["top_scores"][0]["crop"] == result["predicted_crop"]
    assert result["scores_meaning"]


@requires_model
def test_prediction_includes_crop_reference_notes(valid_payload):
    result = predict_crop(valid_payload)
    info = result["crop_info"]
    assert set(info) == {"display", "category", "note"}
    assert info["note"]


@requires_model
def test_prediction_uses_training_data_ranges_for_warnings(valid_payload):
    ranges = {"rainfall": {"min": 20.0, "max": 300.0}}
    payload = dict(valid_payload)
    payload["rainfall"] = 1900

    result = predict_crop(payload, feature_ranges=ranges)
    assert result["predicted_crop"]
    assert any("outside the range" in warning for warning in result["warnings"])


def test_prediction_rejects_invalid_input_before_touching_the_model(valid_payload):
    payload = dict(valid_payload)
    payload["N"] = "hello"

    with pytest.raises(InputValidationError):
        predict_crop(payload)


def test_prediction_without_model_raises_model_not_available(monkeypatch, valid_payload, tmp_path):
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "missing.joblib")
    predict_module._CACHE.update({"pipeline": None, "mtime": None})

    with pytest.raises(ModelNotAvailableError):
        predict_crop(valid_payload)

    predict_module._CACHE.update({"pipeline": None, "mtime": None})


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
@requires_model
def test_cli_predicts_from_json_argument(capsys, valid_payload):
    exit_code = predict_module.main(["--json", json.dumps(valid_payload)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["predicted_crop"]
    assert payload["model_name"]


@requires_model
def test_cli_predicts_from_named_arguments(capsys):
    exit_code = predict_module.main(
        ["--N", "90", "--P", "42", "--K", "43", "--temperature", "25",
         "--humidity", "80", "--ph", "6.5", "--rainfall", "200"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["predicted_crop"]


def test_cli_rejects_invalid_json(capsys):
    exit_code = predict_module.main(["--json", "{not json"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Invalid JSON" in captured.err


def test_cli_requires_all_values(capsys):
    exit_code = predict_module.main(["--N", "90"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Missing required value" in captured.err
