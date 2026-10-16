"""Train, tune, evaluate and persist the crop recommendation models.

Pipeline
--------
1. Load ``data/Crop_recommendation.csv`` and validate its schema.
2. Separate the seven features from the ``label`` target.
3. Stratified 80/20 train/test split with a fixed random seed.
4. Tune Decision Tree, Random Forest and SVM with 5-fold cross-validation
   **inside the training partition only** (the test set is never used here).
5. Evaluate the three tuned models once each on the shared held-out test set.
6. Select the deployed model from those measured test metrics and save the
   complete fitted pipeline with joblib.
7. Write the evaluation reports (metrics.json, classification_report.txt,
   confusion matrices and comparison charts) into ``reports/``.

Run from the project root:

    python src/train.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make sibling modules importable when this file is executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend: no GUI window, safe on any machine
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from common import (  # noqa: E402
    APP_VERSION,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    CV_FOLDS,
    CV_SCORING,
    DATASET_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_INFO_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    load_dataset,
    package_versions,
    sha256_file,
)
from validate_data import validate_dataset  # noqa: E402

SELECTION_POLICY = (
    "Hyperparameters are tuned with 5-fold stratified cross-validation on the "
    "training partition only (scoring = macro F1). The test set is untouched "
    "during tuning. All three tuned models are then evaluated once on the same "
    "held-out test set, and the model with the highest held-out macro F1 is "
    "selected (ties broken by accuracy, then by cross-validation score)."
)


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #
def build_model_specs() -> list[dict[str, Any]]:
    """Return the three candidate algorithms with their search grids."""
    return [
        {
            "key": "decision_tree",
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(random_state=RANDOM_SEED),
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 4],
            },
            "summary": (
                "A single decision tree that splits the data on threshold questions "
                "such as 'is rainfall above 150 mm?'. It is easy to visualise and "
                "explain, but a single tree can overfit the training data."
            ),
        },
        {
            "key": "random_forest",
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 15, 30],
                "min_samples_leaf": [1, 2],
            },
            "summary": (
                "An ensemble (bag) of many decision trees whose votes are pooled. "
                "Averaging many decorrelated trees usually reduces the overfitting "
                "of a single tree at the cost of a larger, less transparent model."
            ),
        },
        {
            "key": "svm",
            "name": "Support Vector Machine (SVM)",
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(probability=True, random_state=RANDOM_SEED)),
                ]
            ),
            "param_grid": {
                "svc__C": [1, 10, 50],
                "svc__gamma": ["scale"],
                "svc__kernel": ["rbf", "linear"],
            },
            "summary": (
                "A Support Vector Machine finds decision boundaries that separate "
                "the crop classes with the widest margin. It is wrapped in a "
                "StandardScaler pipeline because SVM is sensitive to feature scale."
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def evaluate_on_test(
    model: BaseEstimator, x_test: pd.DataFrame, y_test: pd.Series, labels: list[str]
) -> dict[str, Any]:
    """Compute the full metric set for one fitted model on the held-out test set."""
    predictions = model.predict(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = float(
        precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )[2]
    )
    report = classification_report(
        y_test, predictions, labels=labels, target_names=labels, zero_division=0, output_dict=True
    )
    report_text = classification_report(
        y_test, predictions, labels=labels, target_names=labels, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": weighted_f1,
        "per_class": report,
        "classification_report_text": report_text,
        "confusion_matrix": matrix.tolist(),
    }


def feature_importance(model: BaseEstimator, feature_names: list[str]) -> dict[str, float] | None:
    """Return normalised feature importances when the model exposes them."""
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0)
    else:
        return None

    if values.ndim != 1 or len(values) != len(feature_names):
        return None
    total = float(values.sum()) or 1.0
    return {name: float(v / total) for name, v in zip(feature_names, values)}


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
        .strip("_")
    )


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    """Render one confusion matrix as a PNG heatmap (matplotlib Agg backend)."""
    size = max(8.0, 0.42 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size * 0.85), dpi=130)
    im = ax.imshow(matrix, cmap="Greens", aspect="auto")
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted crop")
    ax.set_ylabel("Actual crop")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = int(matrix[i, j])
            if value == 0:
                continue
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "#14311a",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Test samples")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_comparison_chart(results: dict[str, dict[str, Any]], path: Path) -> None:
    """Grouped bar chart comparing the three models on the held-out test set."""
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("macro_f1", "Macro F1"),
    ]
    model_names = [payload["name"] for payload in results.values()]
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(model_names), 1)
    colors = ["#1b5e20", "#4caf50", "#9ccc65"]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=130)
    for index, payload in enumerate(results.values()):
        values = [payload["test"][key] for key, _ in metric_keys]
        offsets = x - 0.4 + width / 2 + index * width
        bars = ax.bar(offsets, values, width, label=payload["name"], color=colors[index % len(colors)])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score on held-out test set")
    ax.set_title("Model comparison - held-out test metrics (80/20 stratified split)")
    ax.axhline(1.0, color="#bdbdbd", linewidth=0.8, linestyle="--")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance_chart(importance: dict[str, float], model_name: str, path: Path) -> None:
    names = list(importance.keys())
    values = [importance[name] for name in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=130)
    bars = ax.barh(names[::-1], values[::-1], color="#2e7d32")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xlabel("Relative importance")
    ax.set_title(f"Feature importance - {model_name}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_dataset_overview(df: pd.DataFrame, labels: list[str], path: Path) -> None:
    """Class distribution plus one histogram per input feature."""
    counts = df[TARGET_COLUMN].value_counts().reindex(labels)
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), dpi=120)

    ax = axes[0, 0]
    ax.bar(counts.index, counts.values, color="#2e7d32")
    ax.set_title("Crop class distribution (records per class)")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=90, labelsize=8)

    positions = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    for feature, position in zip(FEATURE_COLUMNS, positions):
        ax = axes[position]
        ax.hist(df[feature], bins=25, color="#66bb6a", edgecolor="white")
        ax.set_title(f"Distribution of {feature}")
        ax.set_ylabel("Records")

    fig.suptitle("Crop Recommendation Dataset - descriptive overview", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_classification_report_file(
    payloads: dict[str, dict[str, Any]],
    selected_key: str,
    dataset_profile: dict[str, Any],
    split: dict[str, Any],
    trained_at: str,
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("CROP RECOMMENDATION SYSTEM - CLASSIFICATION REPORT")
    lines.append("=" * 78)
    lines.append(f"App version          : {APP_VERSION}")
    lines.append(f"Trained at (UTC)     : {trained_at}")
    lines.append(f"Dataset              : {DATASET_PATH.name}")
    lines.append(f"Dataset rows         : {dataset_profile['rows']}")
    lines.append(f"Crop classes         : {dataset_profile['n_classes']}")
    lines.append(f"Train / test rows    : {split['train_rows']} / {split['test_rows']}")
    lines.append(f"Split                : stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, "
                 f"random_state={RANDOM_SEED}")
    lines.append(f"Cross-validation     : StratifiedKFold(n_splits={CV_FOLDS}), scoring={CV_SCORING} "
                 "(training partition only)")
    lines.append("")
    lines.append("MODEL COMPARISON (held-out test set)")
    lines.append("-" * 78)
    header = f"{'Model':<34}{'Accuracy':>10}{'Macro P':>10}{'Macro R':>10}{'Macro F1':>10}"
    lines.append(header)
    for payload in payloads.values():
        test = payload["test"]
        lines.append(
            f"{payload['name']:<34}{test['accuracy']:>10.4f}{test['macro_precision']:>10.4f}"
            f"{test['macro_recall']:>10.4f}{test['macro_f1']:>10.4f}"
        )
    lines.append("")
    lines.append(f"Selected model: {payloads[selected_key]['name']}")
    lines.append(f"Selection policy: {SELECTION_POLICY}")
    lines.append("")

    for payload in payloads.values():
        lines.append("=" * 78)
        lines.append(f"{payload['name']} - per-class test metrics")
        lines.append("=" * 78)
        lines.append(f"Best hyperparameters : {payload['best_params']}")
        lines.append(
            f"Cross-validation     : {payload['cv_best_score']:.4f} (macro F1, "
            f"+/- {payload['cv_std']:.4f}, best of {payload['cv_candidates']} candidates)"
        )
        lines.append(f"Fit time (seconds)   : {payload['fit_seconds']:.2f}")
        lines.append("")
        lines.append(payload["test"]["classification_report_text"])
        lines.append("")

    lines.append("=" * 78)
    lines.append("NOTE")
    lines.append("=" * 78)
    lines.append(
        "Metrics describe performance on this dataset's held-out test split. They are\n"
        "not a guarantee of agronomic success on any specific farm or season."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Crop Recommendation System - training")
    print("=" * 70)

    # 1. Load + validate -----------------------------------------------------
    print(f"\n[1/7] Loading dataset: {DATASET_PATH}")
    df = load_dataset()
    profile = validate_dataset(df, strict=True)
    dataset_hash = sha256_file(DATASET_PATH)
    print(f"      rows={profile['rows']}  classes={profile['n_classes']}  "
          f"duplicates={profile['duplicate_rows']}")
    print(f"      sha256={dataset_hash}")

    labels = sorted(str(c) for c in df[TARGET_COLUMN].unique())

    # 2. Features / target --------------------------------------------------
    x = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(str)
    if TARGET_COLUMN in FEATURE_COLUMNS:  # defensive: never leak the target
        raise RuntimeError("Target column must not be part of the input features.")

    # 3. Stratified split ---------------------------------------------------
    print(f"\n[2/7] Stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} split "
          f"(random_state={RANDOM_SEED})")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    split = {"train_rows": int(len(x_train)), "test_rows": int(len(x_test))}
    print(f"      train={split['train_rows']}  test={split['test_rows']}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # 4. Tune on training data only ----------------------------------------
    results: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(build_model_specs(), start=1):
        print(f"\n[3/7] ({index}/3) GridSearchCV - {spec['name']} "
              f"({CV_FOLDS}-fold, scoring={CV_SCORING}, training data only)")
        started = time.perf_counter()
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            scoring=CV_SCORING,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        best_index = search.best_index_
        print(f"      best params : {search.best_params_}")
        print(f"      CV macro F1 : {search.best_score_:.4f} "
              f"(+/- {search.cv_results_['std_test_score'][best_index]:.4f}), "
              f"{len(search.cv_results_['params'])} candidates, {fit_seconds:.1f}s")
        results[spec["key"]] = {
            "key": spec["key"],
            "name": spec["name"],
            "summary": spec["summary"],
            "param_grid": spec["param_grid"],
            "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                            for k, v in search.best_params_.items()},
            "cv_best_score": float(search.best_score_),
            "cv_std": float(search.cv_results_["std_test_score"][best_index]),
            "cv_candidates": int(len(search.cv_results_["params"])),
            "fit_seconds": float(fit_seconds),
            "estimator": search.best_estimator_,
        }

    # 5. Evaluate each tuned model once on the held-out test set -------------
    print("\n[4/7] Evaluating the tuned models on the held-out test set")
    for spec in results.values():
        spec["test"] = evaluate_on_test(spec["estimator"], x_test, y_test, labels)
        spec["supports_probability"] = hasattr(spec["estimator"], "predict_proba")
        spec["feature_importance"] = feature_importance(spec["estimator"], FEATURE_COLUMNS)
        print(f"      {spec['name']:<32} accuracy={spec['test']['accuracy']:.4f}  "
              f"macro F1={spec['test']['macro_f1']:.4f}")

    # 6. Select the deployed model -----------------------------------------
    print("\n[5/7] Selecting the deployed model (test metrics compared once)")
    ranking = sorted(
        results.values(),
        key=lambda payload: (
            payload["test"]["macro_f1"],
            payload["test"]["accuracy"],
            payload["cv_best_score"],
        ),
        reverse=True,
    )
    selected = ranking[0]
    print(f"      selected: {selected['name']}")

    # 7. Persist artifacts --------------------------------------------------
    print("\n[6/7] Saving artifacts")
    joblib.dump(selected["estimator"], MODEL_PATH)
    print(f"      model      -> {MODEL_PATH}")

    for payload in results.values():
        matrix = np.array(payload["test"]["confusion_matrix"])
        matrix_path = REPORT_MATRIX_PATHS[payload["key"]]
        save_confusion_matrix(
            matrix, labels, f"Confusion matrix - {payload['name']} (test set)", matrix_path
        )
        if payload["key"] == selected["key"]:
            save_confusion_matrix(
                matrix, labels, f"Confusion matrix - {selected['name']} (selected model, test set)",
                CONFUSION_MATRIX_PATH,
            )
    print(f"      confusion matrices -> {REPORTS_DIR}")

    save_comparison_chart(results, REPORTS_DIR / "model_comparison.png")
    if selected["feature_importance"]:
        save_feature_importance_chart(
            selected["feature_importance"], selected["name"], REPORTS_DIR / "feature_importance.png"
        )
    save_dataset_overview(df, labels, REPORTS_DIR / "dataset_overview.png")
    print(f"      comparison / importance / dataset charts -> {REPORTS_DIR}")

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    metrics: dict[str, Any] = {
        "app_version": APP_VERSION,
        "trained_at_utc": trained_at,
        "training_config": {
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "train_test_split": "stratified",
            "cv_folds": CV_FOLDS,
            "cv_scoring": CV_SCORING,
            "selection_policy": SELECTION_POLICY,
        },
        "dataset": {
            "file": DATASET_PATH.name,
            "sha256": dataset_hash,
            "rows": profile["rows"],
            "n_features": len(FEATURE_COLUMNS),
            "n_classes": profile["n_classes"],
            "classes": profile["classes"],
            "class_counts": profile["class_counts"],
            "feature_ranges": profile["feature_ranges"],
            "missing_values": profile["missing_values"],
            "duplicate_rows": profile["duplicate_rows"],
        },
        "feature_columns": FEATURE_COLUMNS,
        "split": split,
        "models": {
            payload["key"]: {
                "name": payload["name"],
                "summary": payload["summary"],
                "best_params": payload["best_params"],
                "cv_best_score": payload["cv_best_score"],
                "cv_std": payload["cv_std"],
                "cv_candidates": payload["cv_candidates"],
                "fit_seconds": payload["fit_seconds"],
                "supports_probability": payload["supports_probability"],
                "feature_importance": payload["feature_importance"],
                "test": {
                    key: value
                    for key, value in payload["test"].items()
                    if key not in {"classification_report_text"}
                },
            }
            for payload in results.values()
        },
        "selected_model": {
            "key": selected["key"],
            "name": selected["name"],
            "test": {
                key: value
                for key, value in selected["test"].items()
                if key not in {"classification_report_text"}
            },
            "cv_best_score": selected["cv_best_score"],
            "best_params": selected["best_params"],
        },
        "ranking": [payload["name"] for payload in ranking],
        "package_versions": package_versions(),
        "artifacts": {
            "model": "models/crop_pipeline.joblib",
            "metrics": "reports/metrics.json",
            "classification_report": "reports/classification_report.txt",
            "confusion_matrix": "reports/confusion_matrix.png",
            "model_comparison": "reports/model_comparison.png",
            "dataset_overview": "reports/dataset_overview.png",
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"      metrics    -> {METRICS_PATH}")

    model_info = {
        "selected_model": selected["name"],
        "selected_model_key": selected["key"],
        "trained_at_utc": trained_at,
        "dataset_sha256": dataset_hash,
        "dataset_rows": profile["rows"],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "classes": profile["classes"],
        "test_accuracy": selected["test"]["accuracy"],
        "test_macro_f1": selected["test"]["macro_f1"],
        "metrics_file": "reports/metrics.json",
    }
    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as handle:
        json.dump(model_info, handle, indent=2)
    print(f"      model info -> {MODEL_INFO_PATH}")

    write_classification_report_file(
        results, selected["key"], profile, split, trained_at, CLASSIFICATION_REPORT_PATH
    )
    print(f"      report     -> {CLASSIFICATION_REPORT_PATH}")

    print("\n[7/7] Done. Summary")
    for payload in ranking:
        print(
            f"      {payload['name']:<34} accuracy={payload['test']['accuracy']:.4f}  "
            f"macro_f1={payload['test']['macro_f1']:.4f}"
        )
    print(f"\nSelected model: {selected['name']}  ({MODEL_PATH.name})")
    print("Start the web app with: python app.py\n")
    return metrics


# Paths for the per-model confusion matrices (module level so train() stays tidy)
MODEL_SLUGS = ["decision_tree", "random_forest", "svm"]
REPORT_MATRIX_PATHS = {
    slug: REPORTS_DIR / f"confusion_matrix_{slug}.png" for slug in MODEL_SLUGS
}


if __name__ == "__main__":
    train()
