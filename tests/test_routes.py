"""HTTP-level tests for the Flask pages, the JSON API and error handling."""

from __future__ import annotations

import json

import pytest

import app as app_module
from common import DATASET_PATH, FEATURE_COLUMNS, METRICS_PATH

from conftest import requires_dataset, requires_metrics, requires_model


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("path", "expected_text"),
    [
        ("/", "Crop Recommendation System"),
        ("/recommend", "Predict Suitable Crop"),
        ("/performance", "Model"),
        ("/dataset", "Dataset"),
        ("/docs", "Problem statement"),
        ("/about", "About this project"),
    ],
)
def test_pages_render(client, path, expected_text):
    response = client.get(path)
    assert response.status_code == 200
    assert expected_text.lower() in response.get_data(as_text=True).lower()


def test_unknown_page_returns_html_404(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
    body = response.get_data(as_text=True)
    assert "404" in body
    assert "<html" in body.lower()


def test_recommend_page_has_all_seven_inputs(client):
    body = client.get("/recommend").get_data(as_text=True)
    for field in FEATURE_COLUMNS:
        assert f'id="input-{field}"' in body


# --------------------------------------------------------------------------- #
# Health and summary
# --------------------------------------------------------------------------- #
def test_api_health_reports_readiness(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    for key in ("dataset_ready", "model_ready", "metrics_ready", "train_command"):
        assert key in payload


def test_api_summary_uses_real_numbers(client):
    response = client.get("/api/summary")
    assert response.status_code == 200
    payload = response.get_json()

    assert "dataset" in payload and "setup" in payload
    if DATASET_PATH.is_file():
        assert payload["dataset"]["available"] is True
        assert payload["dataset"]["rows"] > 0
        assert payload["dataset"]["n_classes"] > 1
        assert sum(payload["dataset"]["class_counts"].values()) == payload["dataset"]["rows"]


@requires_metrics
def test_api_summary_metrics_match_the_stored_report(client):
    with open(METRICS_PATH, encoding="utf-8") as handle:
        metrics = json.load(handle)

    payload = client.get("/api/summary").get_json()
    assert payload["metrics"]["selected_model"] == metrics["selected_model"]["name"]
    assert payload["metrics"]["accuracy"] == pytest.approx(metrics["selected_model"]["test"]["accuracy"])
    assert payload["metrics"]["macro_f1"] == pytest.approx(metrics["selected_model"]["test"]["macro_f1"])
    assert len(payload["metrics"]["comparison"]) == 3


# --------------------------------------------------------------------------- #
# Metrics and dataset endpoints
# --------------------------------------------------------------------------- #
@requires_metrics
def test_api_model_metrics_returns_all_three_models(client):
    response = client.get("/api/model-metrics")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["available"] is True
    assert set(payload["models"]) == {"decision_tree", "random_forest", "svm"}
    for model in payload["models"].values():
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            assert 0.0 <= model["test"][key] <= 1.0
    assert payload["selected_model"]["name"]
    assert payload["feature_columns"] == FEATURE_COLUMNS


def test_api_model_metrics_returns_503_when_report_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "METRICS_PATH", tmp_path / "missing_metrics.json")
    app_module._CACHE.update({"metrics": None, "metrics_mtime": None})

    response = client.get("/api/model-metrics")
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["available"] is False
    assert "train_command" in payload


@requires_dataset
def test_api_dataset_insights_matches_the_csv(client, dataset_df):
    response = client.get("/api/dataset-insights")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["available"] is True
    assert payload["rows"] == len(dataset_df)
    assert payload["n_classes"] == dataset_df["label"].nunique()
    assert sum(payload["class_counts"].values()) == len(dataset_df)
    assert payload["columns"] == list(dataset_df.columns)
    assert set(payload["histograms"]) == set(FEATURE_COLUMNS)
    assert set(payload["describe"]) == set(FEATURE_COLUMNS)
    assert payload["duplicate_rows"] == 0


def test_api_dataset_insights_returns_503_when_dataset_missing(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "DATASET_PATH", tmp_path / "missing.csv")
    app_module._CACHE.update({"dataset": None, "dataset_mtime": None})

    response = client.get("/api/dataset-insights")
    assert response.status_code == 503
    assert response.get_json()["available"] is False


# --------------------------------------------------------------------------- #
# Prediction endpoint
# --------------------------------------------------------------------------- #
@requires_model
@requires_dataset
def test_api_predict_returns_a_real_model_prediction(client, valid_payload, dataset_df):
    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["status"] == "success"
    assert payload["predicted_crop"] in set(dataset_df["label"].astype(str))
    assert payload["model_name"]
    assert payload["input_values"] == {k: float(v) for k, v in valid_payload.items()}
    assert isinstance(payload["warnings"], list)
    assert payload["crop_info"]["note"]


def test_api_predict_rejects_non_numeric_input(client, valid_payload):
    payload = dict(valid_payload)
    payload["N"] = "hello"

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400

    body = response.get_json()
    assert body["status"] == "error"
    assert "N" in body["field_errors"]
    assert "number" in body["field_errors"]["N"].lower()


def test_api_predict_rejects_missing_field(client, valid_payload):
    payload = dict(valid_payload)
    payload.pop("ph")

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "ph" in response.get_json()["field_errors"]


def test_api_predict_rejects_out_of_range_value(client, valid_payload):
    payload = dict(valid_payload)
    payload["humidity"] = 150

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "humidity" in response.get_json()["field_errors"]


def test_api_predict_rejects_non_json_body(client):
    response = client.post("/api/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400
    assert "json" in response.get_json()["error"].lower()


@requires_model
def test_api_predict_warns_about_values_outside_the_training_range(client):
    response = client.post(
        "/api/predict",
        json={"N": 90, "P": 42, "K": 43, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 5000},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["predicted_crop"]
    assert payload["warnings"], "expected a warning for rainfall far outside the dataset range"


def test_api_predict_returns_503_when_model_is_missing(client, monkeypatch, tmp_path, valid_payload):
    import predict as predict_module

    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "missing.joblib")
    predict_module._CACHE.update({"pipeline": None, "mtime": None})

    response = client.post("/api/predict", json=valid_payload)
    assert response.status_code == 503

    body = response.get_json()
    assert body["status"] == "error"
    assert "train" in body["error"].lower()
    assert "Traceback" not in response.get_data(as_text=True)
    assert "src/train.py" in body["error"]


def test_api_predict_wrong_method_is_json_405(client):
    response = client.get("/api/predict")
    assert response.status_code == 405
    assert response.get_json()["status"] == "error"


def test_api_unknown_endpoint_returns_json_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["status"] == "error"


def test_error_responses_never_leak_a_traceback(client, valid_payload):
    from validate_data import FEATURE_COLUMNS as columns  # noqa: F401  (import check)

    response = client.post("/api/predict", json={"N": object().__str__()})
    assert response.status_code == 400
    text = response.get_data(as_text=True)
    for leak in ("Traceback", "File \"", "werkzeug"):
        assert leak not in text


# --------------------------------------------------------------------------- #
# Report image serving
# --------------------------------------------------------------------------- #
def test_report_image_whitelist(client, monkeypatch, tmp_path):
    png = tmp_path / "confusion_matrix.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(app_module, "REPORTS_DIR", tmp_path)

    assert client.get("/reports/confusion_matrix.png").status_code == 200
    assert client.get("/reports/../../app.py").status_code == 404
    assert client.get("/reports/secret.png").status_code == 404
