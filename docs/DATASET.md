# Dataset Card - Crop Recommendation Dataset

**Dataset Name:** Crop Recommendation Dataset  
**Source:** Kaggle (Atharva Ingle)  
**URL:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset  
**Local File:** `data/Crop_recommendation.csv`  
**Date Verified:** 2026-10-16

## 1. Dataset Summary

The Crop Recommendation Dataset contains soil and environmental parameters along with recommended crop labels. It is designed for building ML models to suggest suitable crops based on NPK values and climate conditions.

- **Total Records:** 2,200
- **Features (Inputs):** 7 numeric
- **Target:** 1 categorical (crop label)
- **Classes:** 22 crop types
- **Class Distribution:** Balanced - 100 samples per class
- **Missing Values:** 0
- **Duplicate Rows:** 0
- **File Format:** CSV

## 2. Data Verification

**SHA-256 (verified):** `808e1d84a544647f1ab2fdae04f434b1842692d6be09eee18a5c2d43eb78d77c`

Verification performed by:
1. Comparing two public mirrors (after whitespace normalization)
2. Computing SHA-256 of the saved file
3. Schema validation in `src/validate_data.py`

## 3. Features and Units

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| N | numeric | ratio/value | Nitrogen content in soil |
| P | numeric | ratio/value | Phosphorus content in soil |
| K | numeric | ratio/value | Potassium content in soil |
| temperature | numeric | °C | Ambient temperature |
| humidity | numeric | % | Relative humidity |
| ph | numeric | pH | Soil pH (0-14 scale) |
| rainfall | numeric | mm | Rainfall amount (accumulation period not specified in dataset) |

## 4. Target Classes

All 22 classes (alphabetically): apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

Each class has exactly 100 records - perfectly balanced dataset.

## 5. Statistical Summary (Observed Ranges)

Computed from actual data:

| Feature | Min | Max | Mean | Std Dev | Median (approx) |
| --- | --- | --- | --- | --- | --- |
| N | 0.000 | 140.000 | 50.552 | 36.917 | ~50 |
| P | 5.000 | 145.000 | 53.363 | 32.986 | ~52 |
| K | 5.000 | 205.000 | 48.149 | 50.648 | ~32 |
| temperature | 8.826 | 43.675 | 25.616 | 5.064 | ~25.6 |
| humidity | 14.258 | 99.982 | 71.482 | 22.264 | ~80.5 |
| ph | 3.505 | 9.935 | 6.469 | 0.774 | ~6.4 |
| rainfall | 20.211 | 298.560 | 103.464 | 54.958 | ~94.5 |

## 6. Data Quality

- **Completeness:** 100% - no missing values in any column
- **Consistency:** All features numeric, label non-empty strings
- **Validity:** Values within reasonable agricultural ranges (with some extremes)
- **Uniqueness:** 0 duplicate rows
- **Balance:** Perfect class balance (stratified splitting ideal)

## 7. Known Limitations

1. **No geographic information** - No location, region, or soil type classification
2. **No temporal information** - No season, date, or year of observation
3. **Unclear rainfall period** - Dataset states "rainfall (mm)" but doesn't specify if daily, monthly, seasonal, or annual. UI explicitly notes this.
4. **Limited soil parameters** - Only NPK and pH; missing organic matter, soil texture, electrical conductivity, micronutrients
5. **Balanced distribution** - All classes have equal samples; real-world crop occurrence is imbalanced
6. **Possible synthetic augmentation** - Values appear structured around typical crop requirements; this contributes to high model performance
7. **No outcome data** - No actual yield, profit, or success/failure - just "recommended crop" based on conditions

## 8. Usage Considerations

- This dataset is suitable for **educational and demonstration** purposes
- Model performance on this dataset should not be extrapolated directly to all agricultural regions
- Treat predictions as **decision support**, not prescriptive advice
- Always validate with local soil tests and agronomic expertise
- Check Kaggle page for dataset license before redistribution

## 9. References

- Atharva Ingle. "Crop Recommendation Dataset." Kaggle. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
