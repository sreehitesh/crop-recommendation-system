# Model Card - Crop Recommendation System

**Model Card Date:** 2026-10-16  
**Model Version:** 1.0.0  
**Trained On:** 2026-10-16T06:01:29+00:00 UTC  
**Dataset SHA-256:** 808e1d84a544647f1ab2fdae04f434b1842692d6be09eee18a5c2d43eb78d77c

## 1. Model Overview

This model card describes the machine learning models trained for crop recommendation. Three classifiers were trained, tuned, and evaluated; the **Random Forest** classifier was selected for deployment based on held-out test performance.

**Task:** Multi-class classification (22 classes)  
**Problem Type:** Supervised learning  
**Input:** 7 numeric features (N, P, K, temperature, humidity, pH, rainfall)  
**Output:** Predicted crop label (one of 22)

## 2. Models Trained

1. **Decision Tree (CART)** - `sklearn.tree.DecisionTreeClassifier`
2. **Random Forest (Ensemble)** - `sklearn.ensemble.RandomForestClassifier` **[DEPLOYED]**
3. **Support Vector Machine (SVM)** - `sklearn.svm.SVC` with `StandardScaler` in Pipeline

All models configured with `probability=True` to provide class probability scores.

## 3. Data Split

- **Split Method:** Stratified train-test split
- **Train/Test Ratio:** 80/20
- **Random Seed:** 42
- **Training Samples:** 1,760 (80 per class)
- **Test Samples:** 440 (20 per class)
- **Stratification:** Preserved class proportions

## 4. Hyperparameter Tuning

Hyperparameters tuned using **GridSearchCV** with **5-fold StratifiedKFold** on the training partition only (test set never used during tuning). **Scoring metric:** `f1_macro`.

### Decision Tree
**Grid:** criterion=['gini','entropy'], max_depth=[None,3,5,10], min_samples_leaf=[1,2,4]  
**Candidates:** 18 | **Fit time:** 3.93s  
**Best params:** criterion=gini, max_depth=None, min_samples_leaf=2  
**CV macro F1:** 0.9858 ± 0.0051

### Random Forest
**Grid:** n_estimators=[100,200], max_depth=[None,10], min_samples_leaf=[1,2]  
**Candidates:** 12 | **Fit time:** 4.97s  
**Best params:** n_estimators=200, max_depth=None, min_samples_leaf=1  
**CV macro F1:** 0.9943 ± 0.0048

### SVM (Pipeline: StandardScaler + SVC)
**Grid:** svc__kernel=['linear','rbf'], svc__C=[1,10], svc__gamma=['scale','auto']  
**Candidates:** 6 | **Fit time:** 0.75s  
**Best params:** svc__C=10, svc__gamma=scale, svc__kernel=linear  
**CV macro F1:** 0.9857 ± 0.0018

## 5. Performance on Held-Out Test Set

All models evaluated once on test set (440 samples). Selection: **highest held-out macro F1**.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | 0.9795 | 0.9806 | 0.9795 | 0.9794 | 0.9794 |
| **Random Forest (deployed)** | **0.9955** | **0.9957** | **0.9955** | **0.9955** | 0.9955 |
| SVM (linear) | 0.9886 | 0.9897 | 0.9886 | 0.9886 | 0.9886 |

**Selected:** Random Forest (test macro F1 = 0.9954517)

## 6. Confusion Analysis (Random Forest - Selected)

**Test errors: 2 out of 440 samples (0.45% error rate)**

1. True label: `rice` → Predicted: `jute` (1 instance)
2. True label: `blackgram` → Predicted: `maize` (1 instance)

All other 438 samples correctly classified. See `reports/confusion_matrix_random_forest.png` and `reports/classification_report.txt` for full details.

## 7. Feature Importance (Random Forest)

Computed via mean decrease in impurity (Gini importance) on the trained RF model:

| Feature | Importance |
| --- | --- |
| rainfall | 0.2196 |
| humidity | 0.2171 |
| K (Potassium) | 0.1808 |
| P (Phosphorus) | 0.1513 |
| N (Nitrogen) | 0.1034 |
| temperature | 0.0755 |
| ph | 0.0523 |

**Interpretation:** Environmental factors (rainfall, humidity) and potassium show relatively higher importance in this model, though all features contribute.

## 8. Intended Use

- **Primary purpose:** Educational demonstration and learning (college projects, viva demos)
- **Use cases:** Understanding ML for crop recommendation, prototyping, teaching
- **Decision support:** Should be used as supplementary information, not as sole decision-maker

## 9. Out-of-Scope Uses

- Production agricultural advisory systems without local validation
- Yield prediction or economic optimization
- Replacing professional agronomic advice
- Use in regions with conditions far outside training data ranges without caution

## 10. Limitations & Risks

- **Dataset scope:** Limited to 22 crops with balanced samples; real-world distribution differs
- **Geographic generalization:** No location data - may not generalize to all agro-climatic zones
- **Data age/source:** Based on the specific Kaggle dataset; may not reflect current varieties/practices
- **High accuracy caveat:** Strong performance on this dataset partly due to balanced, structured data
- **Uncalibrated probabilities:** `predict_proba` scores are model-internal, not calibrated success probabilities
- **No uncertainty:** No confidence intervals or abstention mechanism
- **Input sensitivity:** Garbage-in, garbage-out - quality of input measurements matters

## 11. Fairness & Bias

- **Class balance:** Equal representation (100 each) minimizes class bias
- **Feature coverage:** All 7 features used uniformly
- **No demographic data:** Dataset doesn't contain farmer demographics, region, etc.
- **Domain bias:** Dataset reflects common crop-soil-climate relationships from its source

## 12. Model Artifacts

- `models/crop_pipeline.joblib` - Fitted pipeline for deployed model (Random Forest)
- `models/model_info.json` - Metadata (timestamp, hashes, metrics, classes)
- `reports/metrics.json` - Complete measured metrics for all models
- `reports/classification_report.txt` - Per-class detailed reports
- `reports/*.png` - Visualizations (comparison, confusion matrices, feature importance)

## 13. Reproducibility

- Fixed random seeds: 42
- Deterministic train-test split (stratified)
- Pinned library versions in requirements.txt
- Training procedure fully documented in `src/train.py`
- Dataset integrity verified via SHA-256

## 14. Maintenance

- Model retrainable by running: `python src/train.py`
- Dataset can be re-downloaded/verified via: `python src/download_dataset.py`
- All artifacts regenerated deterministically

## 15. Contact & Attribution

- **Implementation:** Complete, reproducible project as specified
- **Dataset:** Crop Recommendation Dataset by Atharva Ingle (Kaggle)
- **License considerations:** Check Kaggle dataset license for redistribution terms
