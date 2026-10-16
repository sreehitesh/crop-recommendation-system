"""Flask web application for the Crop Recommendation System.

Routes
------
Pages:
    GET /            overview dashboard
    GET /recommend   prediction form
    GET /performance model comparison and confusion matrix
    GET /dataset     dataset insights
    GET /docs        project documentation
    GET /about       about the project

JSON API:
    GET  /api/health           application/dataset/model readiness
    GET  /api/summary          headline statistics for the dashboard
    GET  /api/model-metrics    measured model comparison metrics
    GET  /api/dataset-insights dataset statistics and distributions
    POST /api/predict          predict a crop from seven numeric features
    GET  /reports/<file.png>   generated evaluation images (whitelisted)

The model is loaded once and re-used for inference; it is never re-trained by a
web request. If the artifacts are missing the UI shows a setup status instead of
invented numbers.

Run from the project root:

    python app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request, send_from_directory  # noqa: E402

from common import (  # noqa: E402
    APP_NAME,
    APP_VERSION,
    DATASET_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_INFO_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    dataset_available,
    load_dataset,
    metrics_available,
    model_available,
)
from crop_info import CROP_INFO  # noqa: E402
from predict import ModelNotAvailableError, model_display_name, predict_crop  # noqa: E402
from validate_data import FEATURE_SPECS, InputValidationError, validate_dataset  # noqa: E402

# Report images that the web app is allowed to serve (no path traversal).
SERVABLE_REPORTS = {
    "confusion_matrix.png",
    "confusion_matrix_decision_tree.png",
    "confusion_matrix_random_forest.png",
    "confusion_matrix_svm.png",
    "model_comparison.png",
    "feature_importance.png",
    "dataset_overview.png",
}

TRAIN_COMMAND = "python src/train.py"

# Title/subtitle for each page, resolved from the request path.
PAGE_META: dict[str, tuple[str, str]] = {
    "/": ("Overview", "What the system knows right now, read from the real dataset and the trained model"),
    "/recommend": ("Crop Recommendation", "Enter soil nutrients and environmental conditions to get a model prediction"),
    "/performance": ("Model Performance", "Measured results of the Decision Tree, Random Forest and SVM models"),
    "/dataset": ("Dataset Insights", "Descriptive statistics computed from the training dataset"),
    "/docs": ("Documentation", "How the project works, from dataset to prediction"),
    "/about": ("About Project", "Purpose, technology, scope and limitations"),
}

app = Flask(__name__)
app.config.update(
    JSON_SORT_KEYS=False,
    TEMPLATES_AUTO_RELOAD=False,
    MAX_CONTENT_LENGTH=64 * 1024,  # prediction payloads are tiny
)


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #
_CACHE: dict[str, Any] = {"dataset": None, "dataset_mtime": None, "metrics": None, "metrics_mtime": None}


@app.context_processor
def inject_globals() -> dict[str, Any]:
    """Values every template needs (navigation, titles, model readiness)."""
    title, subtitle = PAGE_META.get(request.path, (APP_NAME, ""))
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "page_title": title,
        "page_subtitle": subtitle,
        "model_ready": model_available(),
    }


def _file_mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.is_file() else None


def dataset_insights(force: bool = False) -> dict[str, Any]:
    """Factual dataset profile, computed from the CSV and cached by mtime."""
    mtime = _file_mtime(DATASET_PATH)
    if mtime is None:
        return {"available": False, "error": f"Dataset not found at {DATASET_PATH}."}

    if not force and _CACHE["dataset"] is not None and _CACHE["dataset_mtime"] == mtime:
        return _CACHE["dataset"]

    df = load_dataset()
    profile = validate_dataset(df, strict=False)
    features_present = [c for c in FEATURE_COLUMNS if c in df.columns]

    describe = (
        df[features_present].astype(float).describe().round(4).to_dict()
        if features_present
        else {}
    )

    histograms: dict[str, Any] = {}
    for column in features_present:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        counts, edges = np.histogram(series.to_numpy(dtype=float), bins=12)
        histograms[column] = {
            "labels": [f"{edges[i]:.2f} - {edges[i + 1]:.2f}" for i in range(len(counts))],
            "counts": [int(c) for c in counts],
        }

    insights = {
        "available": True,
        "file": DATASET_PATH.name,
        "rows": profile["rows"],
        "columns": profile["columns"],
        "n_classes": profile.get("n_classes"),
        "classes": profile.get("classes", []),
        "class_counts": profile.get("class_counts", {}),
        "missing_values": profile["missing_values"],
        "duplicate_rows": profile["duplicate_rows"],
        "problems": profile["problems"],
        "feature_ranges": profile.get("feature_ranges", {}),
        "describe": describe,
        "histograms": histograms,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "feature_specs": FEATURE_SPECS,
        "target_column": TARGET_COLUMN,
    }
    _CACHE["dataset"] = insights
    _CACHE["dataset_mtime"] = mtime
    return insights


def model_metrics(force: bool = False) -> dict[str, Any] | None:
    """Measured training metrics, read from reports/metrics.json when present."""
    mtime = _file_mtime(METRICS_PATH)
    if mtime is None:
        return None
    if not force and _CACHE["metrics"] is not None and _CACHE["metrics_mtime"] == mtime:
        return _CACHE["metrics"]

    try:
        with open(METRICS_PATH, encoding="utf-8") as handle:
            metrics = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    _CACHE["metrics"] = metrics
    _CACHE["metrics_mtime"] = mtime
    return metrics


def model_status() -> dict[str, Any]:
    info: dict[str, Any] = {}
    if MODEL_INFO_PATH.is_file():
        try:
            with open(MODEL_INFO_PATH, encoding="utf-8") as handle:
                info = json.load(handle)
        except (OSError, json.JSONDecodeError):
            info = {}
    return {
        "ready": model_available(),
        "path": str(MODEL_PATH),
        "name": info.get("selected_model") or model_display_name(None),
        "trained_at_utc": info.get("trained_at_utc"),
    }


def build_summary() -> dict[str, Any]:
    """Headline numbers for the overview page - never invented, always sourced."""
    insights = dataset_insights()
    metrics = model_metrics()
    status = model_status()

    summary: dict[str, Any] = {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "dataset": {
            "available": insights.get("available", False),
            "rows": insights.get("rows"),
            "n_classes": insights.get("n_classes"),
            "class_counts": insights.get("class_counts", {}),
            "file": insights.get("file"),
        },
        "model": {
            "available": status["ready"],
            "name": status["name"] if status["ready"] else None,
            "trained_at_utc": status["trained_at_utc"],
        },
        "metrics": None,
        "setup": {
            "dataset_ready": bool(insights.get("available")),
            "model_ready": bool(status["ready"]),
            "metrics_ready": bool(metrics),
            "train_command": TRAIN_COMMAND,
            "download_command": "python src/download_dataset.py",
        },
    }

    if metrics:
        selected = metrics.get("selected_model", {})
        test = selected.get("test", {})
        summary["metrics"] = {
            "trained_at_utc": metrics.get("trained_at_utc"),
            "selected_model": selected.get("name"),
            "accuracy": test.get("accuracy"),
            "macro_precision": test.get("macro_precision"),
            "macro_recall": test.get("macro_recall"),
            "macro_f1": test.get("macro_f1"),
            "test_rows": metrics.get("split", {}).get("test_rows"),
            "comparison": {
                key: {
                    "name": payload.get("name"),
                    "accuracy": payload.get("test", {}).get("accuracy"),
                    "macro_precision": payload.get("test", {}).get("macro_precision"),
                    "macro_recall": payload.get("test", {}).get("macro_recall"),
                    "macro_f1": payload.get("test", {}).get("macro_f1"),
                }
                for key, payload in metrics.get("models", {}).items()
            },
        }
    return summary


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html", page="overview", summary=build_summary())


@app.route("/recommend")
def recommend():
    return render_template(
        "recommend.html",
        page="recommend",
        feature_specs=FEATURE_SPECS,
        model_ready=model_available(),
        model_name=model_status()["name"] if model_available() else None,
    )


@app.route("/performance")
def performance():
    return render_template(
        "performance.html", page="performance", metrics=model_metrics(), train_command=TRAIN_COMMAND
    )


@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html", page="dataset", insights=dataset_insights())


@app.route("/docs")
def docs_page():
    metrics = model_metrics()
    return render_template(
        "docs.html",
        page="docs",
        metrics=metrics,
        summary=build_summary(),
        feature_specs=FEATURE_SPECS,
        train_command=TRAIN_COMMAND,
    )


@app.route("/about")
def about_page():
    return render_template(
        "about.html",
        page="about",
        summary=build_summary(),
        crop_count=len(CROP_INFO),
        app_version=APP_VERSION,
    )


# --------------------------------------------------------------------------- #
# JSON API
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def api_health():
    return jsonify(
        {
            "status": "ok",
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "dataset_ready": dataset_available(),
            "model_ready": model_available(),
            "metrics_ready": metrics_available(),
            "train_command": TRAIN_COMMAND,
        }
    )


@app.get("/api/summary")
def api_summary():
    return jsonify(build_summary())


@app.get("/api/model-metrics")
def api_model_metrics():
    metrics = model_metrics()
    if metrics is None:
        return (
            jsonify(
                {
                    "available": False,
                    "error": "Model metrics are not available yet.",
                    "train_command": TRAIN_COMMAND,
                }
            ),
            503,
        )
    return jsonify({"available": True, **metrics})


@app.get("/api/dataset-insights")
def api_dataset_insights():
    insights = dataset_insights()
    if not insights.get("available"):
        return jsonify({"available": False, "error": insights.get("error"), "download_command": "python src/download_dataset.py"}), 503
    return jsonify(insights)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Request body must be valid JSON with Content-Type: application/json.",
                }
            ),
            400,
        )

    try:
        insights = dataset_insights()
        ranges = insights.get("feature_ranges") if insights.get("available") else None
        result = predict_crop(payload, feature_ranges=ranges)
    except InputValidationError as exc:
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "Input validation failed.",
                    "field_errors": exc.errors,
                }
            ),
            400,
        )
    except ModelNotAvailableError as exc:
        return (
            jsonify({"status": "error", "error": str(exc), "train_command": TRAIN_COMMAND}),
            503,
        )
    except Exception:  # noqa: BLE001 - never leak a traceback to the browser
        app.logger.exception("Unexpected error during prediction")
        return (
            jsonify(
                {
                    "status": "error",
                    "error": "The prediction could not be completed because of an unexpected server error.",
                }
            ),
            500,
        )

    return jsonify({"status": "success", **result})


@app.get("/reports/<path:filename>")
def report_file(filename: str):
    """Serve only the generated PNG report images from reports/."""
    if filename not in SERVABLE_REPORTS:
        return jsonify({"status": "error", "error": "Report file not available."}), 404
    return send_from_directory(REPORTS_DIR, filename, mimetype="image/png")


# --------------------------------------------------------------------------- #
# Error handlers
# --------------------------------------------------------------------------- #
def _wants_json() -> bool:
    return request.path.startswith("/api/") or request.path.startswith("/reports/")


@app.errorhandler(404)
def handle_404(_error):
    if _wants_json():
        return jsonify({"status": "error", "error": "Endpoint not found."}), 404
    return render_template("error.html", page="", code=404,
                           message="We could not find that page."), 404


@app.errorhandler(405)
def handle_405(_error):
    if _wants_json():
        return jsonify({"status": "error", "error": "Method not allowed for this endpoint."}), 405
    return render_template("error.html", page="", code=405,
                           message="That action is not allowed here."), 405


@app.errorhandler(413)
def handle_413(_error):
    if _wants_json():
        return jsonify({"status": "error", "error": "Request body is too large."}), 413
    return render_template("error.html", page="", code=413,
                           message="The request body was too large."), 413


@app.errorhandler(500)
def handle_500(_error):
    if _wants_json():
        return jsonify({"status": "error", "error": "Internal server error."}), 500
    return render_template("error.html", page="", code=500,
                           message="Something went wrong on the server."), 500


if __name__ == "__main__":
    print(f"{APP_NAME} v{APP_VERSION}")
    print(f"Dataset ready : {dataset_available()}  ({DATASET_PATH})")
    print(f"Model ready   : {model_available()}  ({MODEL_PATH})")
    if not model_available():
        print(f"  -> train the model first with: {TRAIN_COMMAND}")
    # debug=False: never expose the Werkzeug debugger on a local demo machine.
    app.run(host="127.0.0.1", port=5000, debug=False)
