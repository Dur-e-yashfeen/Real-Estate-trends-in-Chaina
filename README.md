# Real-Estate-trends-in-Chaina
This project tackles the Kaggle competition “China Real Estate Demand Prediction”, where the goal is to forecast future housing demand

```markdown
# 🏠 China Real Estate Demand Prediction

This notebook builds a baseline model to forecast the **number of new house transactions** across Chinese cities using historical real estate data.  
The goal is to predict transaction counts **12 months into the future** for each sector.

---

## 📦 1. Import Libraries

Essential libraries are imported for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`), time series cross-validation (`TimeSeriesSplit`), and colored outputs (`colorama`).

---

## 📂 2. Data Verification

All dataset files are listed from the Kaggle input directory to ensure correct path access:
```

/kaggle/input/china-real-estate-demand-prediction/train/...

```

---

## 🧮 3. Custom Scoring Function

A competition-style metric is implemented to evaluate model predictions:

- Calculates **Absolute Percentage Error (APE)** between actual and predicted values.  
- Defines a **good rate** = fraction of predictions within ±100% error.  
- If good rate < 0.7 → score = 0  
- Otherwise:  
```

score = 1 - (mean(APE_good) / good_rate)

```

This balances accuracy and stability across sectors.

---

## 🧹 4. Load Data

All provided training files are loaded:
- `city_indexes.csv`, `city_search_index.csv`, `sector_POI.csv`
- `land_transactions.csv`, `land_transactions_nearby_sectors.csv`
- `pre_owned_house_transactions.csv`, `pre_owned_house_transactions_nearby_sectors.csv`
- `new_house_transactions.csv` (target variable)
- `test.csv`

---

## ⚙️ 5. Preprocessing

1. Convert month names (`Jan`, `Feb`, etc.) to numeric month codes.  
2. Extract `sector_id` from the sector column.  
3. Split month strings into `year`, `month`, and generate a continuous time index:
```

time = (year - 2019) * 12 + month - 1

````
4. Prepare target data:
- Pivot `amount_new_house_transactions` into a `(time × sector_id)` matrix.  
- Fill missing values with zeros and ensure consistent 96 sectors.  

---

## 📈 6. Visualization

Plots the total transaction volume across time to visualize trends and seasonality.

```python
plt.plot(amount_new_house_transactions.sum(axis=1))
````

---

## 🧠 7. Baseline Model (Geometric Mean Forecast)

A simple baseline model is built using geometric mean forecasting:

* Uses the **last `t1=6` months** to compute geometric mean per sector.
* If a sector had **zeros in the last `t2=6` months**, its prediction is set to zero.
* Cross-validation is done using `TimeSeriesSplit(n_splits=4, test_size=12)`.

Each fold prints:

```
# Fold X: train on months ..., validate on months ...
# Fold X: score=..., good_rate=...
```

Finally, overall performance across folds is summarized.

---

## 📊 8. Test Prediction

Predictions for **months 67–78** (the next 12 months) are generated using the same geometric mean logic:

```python
a_pred = np.exp(np.log(a_tr.tail(t1)).mean(axis=0))
```

Sectors with zero activity remain zero for the forecast horizon.

---

## 📤 9. Submission File

Final predictions are reshaped and merged with the test dataset:

```
test['amount_new_house_transactions_12_months_later'] = predictions
```

Output CSV:

```
submission.csv
```

✅ Successfully created with shape `(rows, 2)`
Columns:

| Column                                        | Description                           |
| --------------------------------------------- | ------------------------------------- |
| id                                            | Unique test identifier (month_sector) |
| amount_new_house_transactions_12_months_later | Predicted transaction count           |

---

## 🧩 10. Summary

* Implements full preprocessing pipeline from raw data to submission.
* Includes competition-like evaluation metric.
* Provides reproducible **geometric mean baseline** using time-based CV.
* Ready for extension to ML models (e.g., LightGBM with Optuna tuning).

---

✅ **Submission file created successfully: `submission.csv`**

```
print(submission.shape)
submission.head()
```

