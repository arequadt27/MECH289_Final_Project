# MECH289 Final Project — Wrist-Based Stress Detection (WESAD)

Stress classification using the WESAD dataset and Empatica E4 wrist wearable signals only (EDA, BVP, TEMP, ACC). Four models are compared — Logistic Regression, Random Forest, XGBoost, and a 1D CNN — using Leave-One-Subject-Out (LOSO) cross-validation across 15 subjects.

## Setup

### 1. Install dependencies

```
pip install numpy pandas scipy scikit-learn xgboost optuna shap matplotlib tensorflow
```

### 2. Place the dataset

The `WESAD` folder must sit **inside the same folder as the scripts**:

```
MECH289 FP/
├── WESAD/
│   ├── S2/
│   │   ├── S2.pkl
│   │   └── ...
│   ├── S3/
│   ├── ...
│   └── S17/
├── wesad_pipeline.py
├── model_comparison.py
├── cnn1d_pipeline.py
├── motion_stratification.py
└── README.md
```

No path configuration is needed — all scripts locate the dataset automatically.

---

### Step 1 — Feature extraction

```
python wesad_pipeline.py
```

Loads all 15 subjects (S2–S17, S12 absent), preprocesses wrist signals, and extracts 19 features from 60-second windows with 50% overlap. Only baseline (1), stress (2), and amusement (3) conditions are used; transitions and meditation windows are excluded.

**Outputs:**
| File | Description |
|---|---|
| `wesad_features.csv` | Feature matrix (1105 windows × 19 features + label + subject_id) |
| `Subject Signals/signal_comparison_S*.png` | Diagnostic signal plots for all 15 subjects |

**Features extracted (19 total):**
- EDA × 7: mean, std, min, max, slope, peak count, mean peak amplitude
- BVP × 5: mean, std, peak count, mean RR interval, std RR interval
- TEMP × 3: mean, std, slope
- ACC × 4: magnitude mean, std, max, energy

---

### Step 2 — 1D CNN

```
python cnn1d_pipeline.py
```

Trains a 1D convolutional neural network on raw multi-modal wrist signals (EDA, BVP, TEMP, ACC magnitude) resampled to a common 64 Hz grid. Each window is 60 seconds × 4 channels. LOSO cross-validation with early stopping and class-weighted loss. Does not depend on `wesad_features.csv` — reads directly from the raw `.pkl` files.


**Expected runtime: ~30–60 minutes (GPU recommended)**

**Outputs:**
| File | Description |
|---|---|
| `cnn1d_results.csv` | Per-subject metrics for the 1D CNN |
| `cnn1d_window_predictions.csv` | Per-window CNN predictions |
| `cnn1d_confusion_matrix.png` | Aggregated confusion matrix |

---

### Step 3 — Model comparison (LR, Random Forest, XGBoost)

```
python model_comparison.py
```

Runs LOSO cross-validation across all 15 subjects. LR uses inner 3-fold grid search over C. RF uses Optuna TPE search (50 trials). XGBoost uses Optuna TPE search (150 trials) over 9 hyperparameters: `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`. StandardScaler is fit on training folds only (no data leakage). Results are reported both including and excluding S14.

**Expected runtime: ~60–90 minutes**

**Outputs:**
| File | Description |
|---|---|
| `model_comparison_results.csv` | Per-subject metrics for LR, RF, XGBoost |
| `all_models_results.csv` | Combined per-subject metrics including 1D-CNN (requires Step 2) |
| `window_predictions.csv` | Per-window predictions (used by motion_stratification.py) |
| `shap_beeswarm.png` | XGBoost SHAP feature effects across all folds |
| `shap_bar.png` | XGBoost global feature importance (mean \|SHAP\|) |
| `lr_coefficients.png` | LR feature coefficients (mean ± std across folds) |
| `feature_importance_comparison.png` | Side-by-side LR coefficients vs XGBoost SHAP |
| `confusion_matrices.png` | Aggregated confusion matrices for all three models |
| `all_models_f1_by_subject.png` | Per-subject F1 bar chart for all four models |

---

### Step 4 — Motion artifact stratification

```
python motion_stratification.py
```

Loads `window_predictions.csv` and splits each subject's windows into low-motion and high-motion halves using a **per-subject median** of `acc_magnitude_mean`. Computes F1, recall, and precision per model per stratum to assess whether wrist motion degrades classification performance.

**Outputs:**
| File | Description |
|---|---|
| `motion_stratification_results.csv` | Metrics per model per stratum |
| `motion_stratification.png` | Grouped bar chart comparing low vs. high motion |


## Label mapping

| WESAD label | Condition | Binary class |
|---|---|---|
| 1 | Baseline | 0 (non-stress) |
| 2 | TSST stress task | 1 (stress) |
| 3 | Amusement | 0 (non-stress) |
| 4 | Meditation | excluded |
| 0, 5, 6, 7 | Transitions / non-TSST | excluded |


## Key design decisions

- **Wrist-only**: chest sensor data is never accessed at any point in the pipeline
- **LOSO cross-validation**: each subject is held out once as the test set; no subject's data appears in both training and test within a fold
- **Class imbalance**: handled via `class_weight='balanced'` (LR, RF) and `scale_pos_weight` computed per fold (XGBoost); inner CV optimises F1 not accuracy
- **No data leakage**: StandardScaler fit on training fold only; windowing does not cross subject boundaries
- **Per-subject motion threshold**: motion stratification uses each subject's own median to control for inter-subject differences in baseline acceleration amplitude
- **S14 note**: RF and XGBoost predict all non-stress for S14 (F1=0.0) despite high PR-AUC (~0.98), indicating a probability calibration failure specific to that subject rather than a ranking failure; LR is unaffected
