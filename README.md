# Activity Monitoring — CogAge Human Activity Recognition 

> Goal: implement the **Pattern Recognition Chain (PRC)** for **human activity recognition** on the **Cognitive Village (CogAge)** dataset and report **Accuracy** and **macro-F1** on the provided test split. 

## Project overview
This repository contains code and data to classify **55 behavioural activities** (4-second executions) using **wearable sensors** from **smartphone, smartwatch, and smartglasses**. We follow the PRC: **pre-processing → feature extraction → classifier training → evaluation**. The dataset is pre-split into **training** and **testing** numpy arrays for each sensor, and labels are provided. 

## Key characteristics :
- Devices & sensors: smartphone (**Accelerometer, Gravity, Gyroscope, LinearAcceleration 200 Hz; Magnetometer 50 Hz**), smartwatch (**Accelerometer, Gyroscope 67 Hz**), smartglasses (**Accelerometer, Gyroscope 20 Hz**). 4 s windows per execution. 
- Behavioural classes: **55** (IDs **0–54**). 4 subjects, ≥ 20 repetitions per activity.
- Required metrics: **Accuracy** and **Average F1-score** (macro F1). (If using DNNs, also **Mean Average Precision**, MAP.)

## Feature engineering

Handcrafted features are computed **per sensor channel** and concatenated across all sensors:
- **Time-domain statistics:** max, min, mean, std, **variance**, **IQR**, **percentiles** (20/50/80).  
- **Signal morphology:** **zero-crossing rate**, counts of **positive/negative peaks**.  
- **TSFRESH** descriptors: `mean_abs_change`, `cid_ce` (complexity-invariant distance).  
- Optional **resampling** within a sensor to standardize `T` where needed.  
(See `RDFAllFeat.ipynb` / `XGBAllFeat.ipynb` code cells for exact implementation.)

For the “TopFeat” runs, we additionally use:
- **RFECV** (recursive feature elimination with cross-validation) to select an informative subset,  
- **StandardScaler** to normalize selected features, and  
- persisting the **selector** and **scaler** to `models/*.pkl` (via `joblib`).

## Models

Two classical classifiers are evaluated:
- **RandomForestClassifier** (scikit-learn) — robust to heterogeneous features.  
- **XGBClassifier** (XGBoost) — gradient-boosted trees, strong baseline on tabular features.

Both variants are evaluated in **AllFeat** and **TopFeat** configurations.
## How to run

1) Open one of the notebooks in `02_ActivityMonitoring/`:
   - `RDFAllFeat.ipynb` or `XGBAllFeat.ipynb` (all features)
   - `RDFTopFeat.ipynb` or `XGBTopFeat.ipynb` (RFECV-selected features + scaling)

2) Execute cells in order:
   - **Load data** from `training/` and `testing/`  
   - **Extract features** per sensor/channel and **concatenate**  
   - (TopFeat) **RFECV + StandardScaler** → save to `models/`  
   - **Train** classifier, **predict** on test, **report** Accuracy & macro-F1  
   - Plot **confusion matrix**

> The notebooks are Colab-friendly. If running on Colab, mount Drive as needed in the first cell.


