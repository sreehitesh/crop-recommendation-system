# Product Requirements Document (PRD) - Crop Recommendation System

**Project:** Crop Recommendation System Using Machine Learning  
**Date:** 2026-10-16  
**Implementation Status:** Complete

## 1. Executive Summary

The Crop Recommendation System is a web-based application that recommends suitable crops based on soil nutrients and environmental conditions using machine learning. Users input seven parameters (N, P, K, temperature, humidity, pH, rainfall), and the system predicts one of 22 crop classes using a trained classifier served via Flask API with an interactive dashboard.

## 2. Objectives

1. Build a complete, runnable "Crop Recommendation System Using Machine Learning" project (Flask + scikit-learn dashboard)
2. Train and compare three machine learning algorithms: Decision Tree, Random Forest, and Support Vector Machine
3. Evaluate models on held-out test set with real measured metrics
4. Provide a user-friendly web dashboard with multiple pages
5. Create comprehensive documentation including model card, dataset card, user guide, technical docs, and test report
6. Ensure reproducibility with pinned dependencies and clear setup instructions
7. Follow honest reporting - no hardcoded predictions, no invented numbers

## 3. Target Users

- Agricultural students (for learning and demo/viva)
- Farmers (as educational decision support - not for production agronomic decisions)
- Researchers/practitioners learning ML applied to agriculture

## 4. Core Features

### 4.1 ML Features
- Train Decision Tree, Random Forest, SVM with hyperparameter tuning via GridSearchCV
- 5-fold stratified cross-validation on training partition only
- Evaluate all models on same held-out test set (stratified 80/20 split, random_state=42)
- Select best model by highest held-out macro F1 (tie-break: accuracy, then CV score)
- Use StandardScaler pipeline for SVM (as SVM is distance-sensitive)
- Save trained model as joblib, export comprehensive metrics and reports

### 4.2 Web Application Features
- 6 dashboard pages: Overview, Crop Recommendation, Model Performance, Dataset Insights, Documentation, About
- Responsive design with deep-green agricultural theme
- Interactive charts using Chart.js (CDN) with HTML fallback
- Form validation (client + server side)
- Real-time prediction via AJAX to REST API
- Display prediction with top-3 class probabilities/scores
- Show validation warnings for out-of-range values
- Clear error messages and setup guidance when dataset/model missing

### 4.3 API Endpoints
- GET `/api/health` - App, dataset, and model readiness
- GET `/api/summary` - Headline statistics
- GET `/api/dataset-insights` - Full dataset profile
- GET `/api/model-metrics` - Measured model comparison metrics
- POST `/api/predict` - Validate inputs and return prediction
- GET `/reports/<file>.png` - Whitelisted report images

## 5. Input Requirements

Users must provide 7 numeric inputs:
1. N (Nitrogen) - soil nutrient content
2. P (Phosphorus) - soil nutrient content
3. K (Potassium) - soil nutrient content
4. temperature - °C
5. humidity - % (relative humidity, 0-100 expected)
6. ph - soil pH (0-14 scale)
7. rainfall - mm (period not specified in dataset)

## 6. Validation Rules

**Hard limits (reject with 400):**
- All fields required
- Must be numeric (int/float, not strings)
- No NaN, Infinity, -Infinity
- humidity: 0-100
- ph: 0-14
- rainfall: >= 0
- temperature: -50 to 60
- N/P/K: 0-1000

**Soft warnings (allow with warning, HTTP 200):**
- Values outside observed dataset ranges (from actual data min/max)
- Warn but don't block prediction

## 7. Dataset

- Source: Kaggle "Crop Recommendation Dataset" by Atharva Ingle
- File: `data/Crop_recommendation.csv`
- Records: 2,200
- Features: 7 numeric inputs
- Target: 22 crop classes (balanced: 100 each)
- No missing values, no duplicates
- SHA-256: `808e1d84a544647f1ab2fdae04f434b1842692d6be09eee18a5c2d43eb78d77c`
- Units: N/P/K ratio values, temp °C, humidity %, pH 0-14, rainfall mm (period undocumented)

## 8. Technical Requirements

**Backend:** Flask 3.0.3  
**ML:** scikit-learn 1.5.1, pandas 2.2.2, numpy 1.26.4  
**Persistence:** joblib 1.4.2  
**Visualization/Reports:** matplotlib 3.9.2  
**Testing:** pytest 8.3.3  
**Frontend:** HTML5, CSS3 (custom), vanilla JavaScript, Chart.js (CDN)

**Python:** 3.12.10 (tested)  
**Platform:** Windows (friendly paths), cross-platform compatible

## 9. Non-Functional Requirements

- **Reproducibility:** Pinned deps, fixed seeds, documented workflow
- **Honesty:** Real metrics only, no hardcoded predictions
- **Modularity:** Separation of training/serving code
- **Testability:** 82 comprehensive tests
- **Documentation:** Two READMEs + 6 docs files
- **Error handling:** Clear messages with appropriate HTTP codes
- **Security:** Path traversal prevention for reports, payload size limit (64KB), input validation
- **Usability:** Clear UI, helpful setup status, example values

## 10. Success Criteria

1. ✅ All 3 models trained and evaluated with real metrics
2. ✅ Best model selected by held-out macro F1
3. ✅ All 82 tests pass
4. ✅ All 6 pages functional with no console errors
5. ✅ API endpoints work correctly with validation
6. ✅ Predictions come from trained model (not hardcoded)
7. ✅ Comprehensive documentation created
8. ✅ Reproducible via documented steps
9. ✅ Two READMEs present

## 11. Implementation Decisions

- **Direct workspace root** (not subfolder) as per user instruction
- **Python 3.12.10 via winget** installed for environment
- **Custom CSS** (no Bootstrap) for agricultural theme
- **Chart.js from CDN** with offline HTML-bar fallback
- **Inline SVG icons** (Lucide-style)
- **.gitignore excludes models/*.joblib** (reproducible via training)
- **reports/*.png and metrics.json committed** to show results
- **Browser screenshots not possible** in environment - documented honestly
- **Background Flask server** used during verification

## 12. Measured Results (Final)

- Dataset: 2200 rows, 22 classes, 7 features, 0 missing, 0 duplicates
- Split: 1760 train / 440 test (stratified, seed=42)
- Decision Tree: test acc 0.9795, macro F1 0.9794
- **Random Forest (deployed):** test acc 0.9955, macro F1 0.9955 (2/440 errors)
- SVM (linear, C=10): test acc 0.9886, macro F1 0.9886 (5/440 errors)
- Feature importance (RF): rainfall 0.2196, humidity 0.2171, K 0.1808, P 0.1513, N 0.1034, temp 0.0755, pH 0.0523
- All tests: 82 passed, 0 failed, 0 skipped
