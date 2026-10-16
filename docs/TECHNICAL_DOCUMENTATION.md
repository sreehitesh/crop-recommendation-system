# Technical Documentation - Crop Recommendation System

**Version:** 1.0.0  
**Date:** 2026-10-16  
**Author:** Implementation based on project requirements

## 1. Architecture Overview

The system follows a layered architecture:

```
Browser (HTML/CSS/JS) ←→ Flask API (app.py) ←→ src modules ←→ ML artifacts (joblib/json)
                                        ↓
                                      Static/Reports
```

### Layers
1. **Presentation Layer**: Jinja2 templates, custom CSS, vanilla JS, Chart.js
2. **Application Layer**: Flask routes, request validation, error handling
3. **Business/ML Layer**: Training, prediction, validation logic in src/
4. **Data Layer**: CSV dataset, joblib models, JSON reports

## 2. Project Structure

```
Default Project/
├── app.py
├── requirements.txt
├── README.md
├── README_PROJECT.md
├── .gitignore
├── data/
│   └── Crop_recommendation.csv
├── models/
│   ├── crop_pipeline.joblib
│   └── model_info.json
├── src/
│   ├── common.py
│   ├── download_dataset.py
│   ├── validate_data.py
│   ├── train.py
│   ├── predict.py
│   └── crop_info.py
├── templates/
│   ├── base.html, index.html, recommend.html, performance.html,
│   ├── dataset.html, docs.html, about.html, error.html, _icons.html
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── images/favicon.svg
├── reports/
│   ├── metrics.json, classification_report.txt
│   ├── *.png (model_comparison, confusion matrices, feature_importance, dataset_overview)
├── docs/
│   └── (PRD.md, DATASET.md, MODEL_CARD.md, USER_GUIDE.md, TECHNICAL_DOCUMENTATION.md, TEST_REPORT.md)
└── tests/
    ├── conftest.py, test_validation.py, test_prediction.py, test_routes.py
```

## 3. Core Modules

### 3.1 common.py
**Purpose:** Shared constants and utilities

**Key elements:**
- `FEATURE_COLUMNS = ["N","P","K","temperature","humidity","ph","rainfall"]` - Feature order (critical)
- `RANDOM_SEED = 42`, `TEST_SIZE = 0.2`, `CV_FOLDS = 5`
- `load_dataset()` - Load CSV with validation
- `package_versions()` - Get package versions for health checks
- Path constants: `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, etc.

### 3.2 validate_data.py
**Purpose:** Input and dataset validation

**FEATURE_SPECS:** Defines for each feature:
- type, required flag, group, help text
- hard_min/hard_max (reject if violated)
- warn_min/warn_max (warn if outside observed range)

**Key functions:**
- `validate_dataset(df)` - Validate dataset structure, missing values, dtypes
- `validate_payload(payload)` - Validate API input (returns (ok, errors, warnings, cleaned))
- `get_observed_ranges(df)` - Extract min/max from data

**Validation logic:**
1. Check required fields exist
2. Convert to numeric (int/float)
3. Check for NaN/inf
4. Apply hard limits
5. Check observed ranges for warnings

### 3.3 download_dataset.py
**Purpose:** Robust dataset acquisition

**Strategy:**
1. If local file exists, verify and skip
2. Try two public mirrors
3. Download both, compare after whitespace normalization
4. Write only if they match (ensures integrity)
5. Compute SHA-256
6. Basic validation

Mirrors used: Google Drive direct link style + raw content mirrors (implementation handles redirects).

### 3.4 train.py
**Purpose:** Full training pipeline

**Workflow:**
1. Load and validate dataset
2. Split stratified 80/20 (random_state=42)
3. Define parameter grids for DT, RF, SVM
4. For each: GridSearchCV with StratifiedKFold(5, shuffle=False or with seed) scoring f1_macro
5. Evaluate best estimator on test set
6. Compute metrics: accuracy, macro P/R/F1, weighted F1, per-class, confusion matrix
7. Compute feature importances
8. Select best by test macro F1 (tie-break: accuracy, CV score)
9. Save artifacts: joblib pipeline, model_info.json, metrics.json, classification reports, plots
10. Generate matplotlib visualizations

**Models:**
- DT/RF: Direct classifier in pipeline conceptually (no scaler)
- SVM: Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])

**Plots:** model comparison, confusion matrices (4), feature importance (RF), dataset overview

### 3.5 predict.py
**Purpose:** Prediction service

**Classes:**
- `ModelNotAvailableError(Exception)` - Raised when model missing/corrupt

**Functions:**
- `load_pipeline(reload=False)` - Load joblib, cache in module
- `predict_crop(inputs)` - Validate inputs, predict, return result with scores
- CLI interface via argparse for testing

**Prediction flow:**
1. Validate inputs (reuse validate_payload logic)
2. Load pipeline
3. Get prediction: `pipeline.predict(X)[0]`
4. Get probabilities: `pipeline.predict_proba(X)[0]`
5. Pair with class names, sort desc
6. Return top-k scores

### 3.6 crop_info.py
**Purpose:** Crop reference information (display only)

Contains brief neutral descriptions for each of 22 crops. Not used by ML model - purely for UI context if needed.

### 3.7 app.py
**Purpose:** Flask application

**Configuration:**
- `MAX_CONTENT_LENGTH = 64 * 1024` (64KB)
- `TEMPLATES_AUTO_RELOAD = True`

**Context processor:** Injects PAGE_META and app_version into templates

**PAGE_META:** Defines titles/descriptions for each page

**Routes (GET):**
- `/` → index.html
- `/recommend` → recommend.html
- `/performance` → performance.html
- `/dataset` → dataset.html
- `/docs` → docs.html
- `/about` → about.html

**API Routes:**
- `GET /api/health` - Returns readiness status
- `GET /api/summary` - Summary stats
- `GET /api/dataset-insights` - Dataset details
- `GET /api/model-metrics` - Model metrics from reports/metrics.json
- `POST /api/predict` - Prediction endpoint

**Static/Reports:**
- `/static/<path>` - Static files
- `/reports/<filename>` - Whitelisted PNG files only (basename check)

**Error handlers:** 400, 404, 500 render templates/error.html

## 4. Frontend Implementation

### 4.1 Templates
- `base.html` - Base layout with sidebar, header, footer, meta tags
- `_icons.html` - SVG icon definitions (inline)
- Page templates extend base
- Error pages standalone

### 4.2 CSS (static/css/style.css)
Custom design system with:
- CSS variables for colors (greens, accents, grays)
- Card-based layouts
- Grid/Flexbox
- Responsive breakpoints
- Form styling
- Table styling
- Badge/pill components
- Alert/warning styles
- Agricultural theme (deep greens)

### 4.3 JavaScript (static/js/app.js)
**Features:**
- Sidebar toggle for mobile
- Chart.js integration from CDN
- **Chart registry (WeakMap)** to prevent duplicate charts
- Canvas element creation (fixes Chart.js v4 context issue)
- HTML bar fallback if Chart.js unavailable
- Form handling for recommendation page
- Load example values / Reset
- Client-side validation
- AJAX POST to /api/predict
- Render prediction results
- Fetch and render charts on performance/dataset pages

**Key fix:** Chart.js v4 requires a `<canvas>` element; code now creates canvas dynamically and passes to Chart.js. Also tracks charts by canvas in WeakMap.

## 5. Data Flow Details

### Prediction Request Flow
1. User fills form → JS validates
2. JS POSTs JSON to /api/predict
3. Flask receives, checks Content-Type
4. validate_payload() validates all fields
5. If hard errors → 400 with errors dict
6. Check warnings (outside observed ranges)
7. predict_crop() called with cleaned inputs
8. load_pipeline() loads joblib (cached)
9. Pipeline predicts and returns proba
10. Build response with crop, model, top-3 scores, warnings, disclaimer
11. Return JSON (200 or error code)
12. JS renders result card

### Training Flow
1. download_dataset.py (optional) gets CSV
2. train.py loads CSV
3. Validates schema
4. Stratified split
5. GridSearchCV per model (CV on train only)
6. Test evaluation
7. Selection by test macro F1
8. Persist artifacts
9. Generate reports

## 6. API Specification

### GET /api/health
**Response:**
```json
{
  "status": "ok",
  "app_version": "1.0.0",
  "python_version": "...",
  "model_ready": true/false,
  "model_info": {...} or null,
  "dataset": {"exists": true, "rows": N, "valid": true, "columns": [...]}
}
```

### GET /api/summary
```json
{
  "dataset": {"rows": 2200, "n_classes": 22, "balanced": true},
  "model": {"selected": "Random Forest", "test_accuracy": 0.9955, "test_macro_f1": 0.9955}
}
```

### GET /api/dataset-insights
Returns dataset profile, describe stats, class distribution, feature ranges

### GET /api/model-metrics
Returns full metrics.json content (503 if missing)

### POST /api/predict
**Request:**
```json
{"N":90,"P":42,"K":43,"temperature":25,"humidity":80,"ph":6.5,"rainfall":200}
```

**Success 200:**
```json
{
  "predicted_crop": "rice",
  "model_used": "Random Forest",
  "scores": [{"label":"rice","probability":0.52}, ...],
  "input": {...},
  "warnings": [],
  "disclaimer": "..."
}
```

**Error 400:**
```json
{"errors": {"N": "Nitrogen (N) must be a number (received 'hello')."}}
```

**Error 503:** Model not available

**Error 500:** Unexpected error

## 7. Testing

### Test Structure
- `conftest.py`: Fixtures (app, client, sample_payloads)
- `test_validation.py`: 37 tests for validation logic
- `test_prediction.py`: 19 tests for prediction service
- `test_routes.py`: 26 tests for Flask routes/API

### Running Tests
```bash
pytest -q  # or with .venv\Scripts\python.exe
```

**Result:** 82 passed, 0 failed, 0 skipped

### Test Coverage Areas
- Input validation (types, ranges, NaN/inf)
- Hard limits enforcement
- Warning generation for out-of-range
- Model loading/caching
- Prediction with valid inputs
- Error handling (missing model)
- All page routes return 200
- API endpoints behavior
- 404/400 cases
- Reports whitelist
- Large payload rejection

## 8. Configuration & Environment

### Dependencies (requirements.txt)
```
Flask==3.0.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
matplotlib==3.9.2
pytest==8.3.3
```

### Environment Variables/Settings
None required. All config in code.

### Paths
All paths resolved via `pathlib.Path` relative to project root. Windows and POSIX compatible.

## 9. Performance Considerations

- **Model loading:** Cached after first load (module-level)
- **Training:** Fast enough (DT~4s, RF~5s, SVM~0.75s total ~10s)
- **Prediction:** Near-instant (< 1ms per request)
- **Reports:** PNGs generated once during training (~few seconds for matplotlib)
- **Memory:** Small (models < 10MB, data < 1MB CSV)

## 10. Security Considerations

- **Input validation:** Strict server-side validation (type, ranges, NaN/inf)
- **Path traversal:** `/reports/<filename>` checks basename only, rejects paths with `/\`
- **Payload size:** 64KB limit via Flask MAX_CONTENT_LENGTH
- **No eval/exec:** No dynamic code execution
- **CORS:** Not configured (localhost-only usage)
- **Error messages:** Generic for unexpected errors, specific for validation

## 11. Deployment Considerations

**Development (current):**
- Flask dev server: `python app.py` (host 127.0.0.1, port 5000)
- Debug mode not enabled in production style

**Production-ready options:**
- Gunicorn/uWSGI behind reverse proxy (nginx)
- Set `debug=False`
- Use environment-specific config
- Ensure `models/*.joblib` present (or retrain)
- Consider containerization (Docker)

But this project is designed for educational/demo use with dev server.

## 12. Debugging & Logging

- Flask default logging for requests/errors
- Training script prints progress to stdout
- Validation errors returned in API responses
- Exception handlers log errors and return safe messages

## 13. Known Issues & Fixes

**Issue:** Chart.js v4 passed div instead of canvas → "can't acquire context"  
**Fix:** Create `<canvas>` element dynamically, pass to Chart.js; added WeakMap chart registry

**Issue:** Browser screenshots not possible in headless env → Documented honestly

**Issue:** Test had `pytest.approx(...) <= float` TypeError  
**Fix:** Corrected comparison in test

## 14. Future Improvements (Optional)

- Add feature scaling explanation in UI
- Export prediction history
- Add more visualizations (ROC curves per class - though multi-class)
- Model explainability (SHAP)
- Form validation with better UX
- Dockerfile
- CI/CD with GitHub Actions

But current implementation meets all PRD requirements.

## 15. References

- Flask: https://flask.palletsprojects.com/
- scikit-learn: https://scikit-learn.org/
- pandas: https://pandas.pydata.org/
- Chart.js: https://www.chartjs.org/
- matplotlib: https://matplotlib.org/
- pytest: https://pytest.org/