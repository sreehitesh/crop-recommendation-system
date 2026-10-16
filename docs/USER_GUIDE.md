# User Guide - Crop Recommendation System

**Version:** 1.0.0  
**Last Updated:** 2026-10-16

## 1. Getting Started

The Crop Recommendation System is a web application that predicts suitable crops based on soil and environmental parameters. This guide walks you through using the dashboard and understanding results.

## 2. Accessing the Application

1. Start the server: `python app.py`
2. Open browser: http://127.0.0.1:5000
3. You'll see the Overview page with system status

## 3. Dashboard Navigation

The app has 6 main pages accessible via the sidebar:

- **Overview** (`/`) - System status, key metrics, class distribution
- **Crop Recommendation** (`/recommend`) - Make predictions
- **Model Performance** (`/performance`) - Model comparison and metrics
- **Dataset Insights** (`/dataset`) - Explore the dataset
- **Documentation** (`/docs`) - Project documentation links
- **About Project** (`/about`) - About the system

## 4. Making a Crop Recommendation

### Step 1: Go to Recommendation Page
Navigate to "Crop Recommendation" from sidebar or click "Try Recommendation" on Overview page.

### Step 2: Enter Input Values

You need to provide 7 numeric values:

| Field | Label | Expected Range | Unit | Notes |
| --- | --- | --- | --- | --- |
| N | Nitrogen (N) | 0-1000 (typical 0-140) | ratio/value | Soil nitrogen content |
| P | Phosphorus (P) | 0-1000 (typical 5-145) | ratio/value | Soil phosphorus content |
| K | Potassium (K) | 0-1000 (typical 5-205) | ratio/value | Soil potassium content |
| temperature | Temperature | -50 to 60 °C (typical 8-44) | °C | Ambient temperature |
| humidity | Humidity | 0-100% | % | Relative humidity |
| ph | Soil pH | 0-14 | pH | Soil acidity/alkalinity |
| rainfall | Rainfall | >= 0 (typical 20-299) | mm | Rainfall amount (period not specified) |

### Step 3: Use Example Values (Optional)
Click "Load Example Values" to auto-fill with tested values: N=90, P=42, K=43, temp=25, humidity=80, pH=6.5, rainfall=200. This example predicts **rice**.

### Step 4: Get Recommendation
Click "Get Recommendation". The form validates inputs client-side and server-side.

### Step 5: Review Results

The result card shows:
- **Predicted Crop** - The recommended crop name (e.g., rice)
- **Model Used** - Algorithm that made prediction (Random Forest - deployed)
- **Top 3 Scores** - Probability-like scores for top 3 crops (highest first)
- **Submitted Values** - Your input values for reference
- **Warnings** - If values outside dataset observed range (soft warning)
- **Disclaimer** - Educational decision support notice

## 5. Understanding Validation

### Hard Validation Errors (400) - Cannot Predict
These indicate impossible/improper values:
- Empty fields or non-numeric input (e.g., "hello")
- NaN or Infinity
- humidity > 100 or < 0
- pH < 0 or > 14
- rainfall < 0
- temperature outside -50 to 60 range
- N/P/K outside 0-1000 range

You'll see inline field errors or API error messages.

### Soft Warnings (200) - Prediction Still Happens
These appear as yellow warning boxes:
- Values outside **observed dataset range** (e.g., rainfall=5000 when max observed is ~298.56)
- Warnings mean the input is unusual compared to training data; prediction is still returned but may be less reliable

## 6. Reading the Overview Page

- **Setup Status**: Shows if dataset and model are ready. If not, it gives exact commands to fix.
- **Key Metrics**: Dataset size, classes, deployed model, test accuracy, macro F1
- **Class Distribution**: Bar chart of 22 crops (balanced at 100 each)
- **How It Works**: 4-step explanation of the system

## 7. Understanding Model Performance

- **Comparison Table**: All 3 models with accuracy, macro precision/recall/F1, CV scores
- **Metric Chart**: Visual comparison of key metrics
- **Selected Model**: Highlighted with green badge (Random Forest)
- **Algorithm Explanations**: Plain-English descriptions of DT, RF, SVM
- **Hyperparameters**: Best params found via CV
- **Confusion Matrices**: Show correct/wrong predictions per class
- **Feature Importance**: Which features most influence RF predictions

**Key takeaway:** Random Forest was selected because it has the highest held-out macro F1 (0.9955).

## 8. Exploring Dataset Insights

- **Overview Cards**: Rows, features, classes, balance status
- **Columns Table**: Column names, data types, non-null counts, observed min/max
- **Class Distribution Chart**: Visual of 22 classes (all 100)
- **Descriptive Statistics**: Mean, std, min, 25%, 50%, 75%, max for each feature
- **Feature Histograms**: Distribution of each of 7 features

## 9. Documentation Page

Links to all documentation files:
- README.md (GitHub quick start)
- README_PROJECT.md (Detailed explanation)
- docs/PRD.md, DATASET.md, MODEL_CARD.md, USER_GUIDE.md, TECHNICAL_DOCUMENTATION.md, TEST_REPORT.md

## 10. Tips for Better Results

1. **Use realistic values**: Enter actual soil test results and weather measurements
2. **Check units**: Ensure temperature in °C, humidity %, pH 0-14, rainfall in mm
3. **Note warnings**: If you see out-of-range warnings, interpret prediction cautiously
4. **Compare top scores**: Look at top 3 - if scores are close, crops may be similarly suitable
5. **Understand limitations**: This is educational - validate with local agronomic advice

## 11. Troubleshooting

**Problem:** "Model not ready" or 503 on predict  
**Solution:** Run `python src/train.py` to train and save the model

**Problem:** "Dataset not found"  
**Solution:** Run `python src/download_dataset.py` to download the CSV

**Problem:** Form won't submit / validation errors  
**Solution:** Ensure all fields are filled with valid numbers. Check for typos like commas instead of decimals (use periods)

**Problem:** Charts not showing  
**Solution:** Check internet connection (Chart.js loads from CDN). App has HTML fallback if CDN blocked.

**Problem:** Port 5000 already in use  
**Solution:** Edit `app.py` and change port, or kill the existing process

## 12. Interpreting Scores

The "Top 3 Scores" show class probabilities from the model's `predict_proba`. For example:
- rice: 0.52 means model assigns ~52% confidence weight to rice
- jute: 0.475 means ~47.5% to jute

**Important:** These are **model confidence scores**, not calibrated probabilities of real-world success. They reflect similarity to patterns in training data.

## 13. Example Test Case

Input: N=90, P=42, K=43, temperature=25, humidity=80, pH=6.5, rainfall=200  
→ Predicted: **rice**  
→ Top scores: rice (0.52), jute (0.475), watermelon (0.005)

This matches verified behavior from live testing.

## 14. Responsible Use

This system is for **educational and demonstration purposes only**. It does not:
- Predict crop yield
- Recommend fertilizers, pesticides, or irrigation
- Account for market prices or economic factors
- Replace soil testing by experts or local extension services

Always use alongside professional agronomic advice for real farming decisions.