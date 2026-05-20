# MECH289 Final Project — Wrist-Based Stress Detection (WESAD)

Stress classification using the WESAD dataset and Empatica E4 wrist wearable signals only (EDA, BVP, TEMP, ACC). Three models are compared — Logistic Regression, Random Forest, and XGBoost — using Leave-One-Subject-Out (LOSO) cross-validation.

---

## Setup

### 1. Install dependencies

```
pip install numpy pandas scipy scikit-learn xgboost optuna shap matplotlib
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
├── lr_pipeline.py
├── motion_stratification.py
└── README.md
```

No path configuration is needed — all scripts locate the dataset automatically.

---

## Running the pipeline

Run scripts **in this order**. Each step depends on outputs from the previous one.

### Step 1 — Feature extraction

```
python wesad_pipeline.py
```

Loads all 15 subjects (S2–S17, S12 absent), preprocesses wrist signals, and extracts 19 features from 60-second windows with 50% overlap. Meditation windows (label 4) are excluded; only baseline (1), stress (2), and amusement (3) conditions are used.

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

### Step 2 — Model comparison (LR, Random Forest, XGBoost)

```
python model_comparison.py
```

Runs LOSO cross-validation across all 15 subjects. Hyperparameters for RF and XGBoost are tuned per fold using Optuna Bayesian search (50 trials). LR uses inner 3-fold grid search. StandardScaler is fit on training folds only (no data leakage).

**Expected runtime: ~15–20 minutes**

**Outputs:**
| File | Description |
|---|---|
| `model_comparison_results.csv` | Per-subject metrics for all three models |
| `window_predictions.csv` | Per-window predictions (used by motion_stratification.py) |
| `shap_beeswarm.png` | XGBoost SHAP feature effects across all folds |
| `shap_bar.png` | XGBoost global feature importance (mean \|SHAP\|) |
| `confusion_matrices.png` | Aggregated confusion matrices for LR, RF, XGBoost |

---

### Step 3 — Logistic Regression detailed analysis

```
python lr_pipeline.py
```

Re-runs LR-only LOSO with the same protocol and adds a feature coefficient analysis — showing which features most strongly drive stress predictions across folds.

**Outputs:**
| File | Description |
|---|---|
| `logistic_regression_results.csv` | Per-subject LR metrics |
| `lr_coefficients.png` | Mean ± std LR coefficients across 15 LOSO folds |

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

---

## Label mapping

| WESAD label | Condition | Binary class |
|---|---|---|
| 1 | Baseline | 0 (non-stress) |
| 2 | TSST stress task | 1 (stress) |
| 3 | Amusement | 0 (non-stress) |
| 4 | Meditation | excluded |
| 0, 5, 6, 7 | Transitions / non-TSST | excluded |

---

## Key design decisions

- **Wrist-only**: chest sensor data is never accessed
- **LOSO cross-validation**: each subject is held out once as the test set; no subject's data appears in both training and test within a fold
- **Class imbalance**: handled via `class_weight='balanced'` (LR, RF) and `scale_pos_weight` per fold (XGBoost); inner CV optimises F1 not accuracy
- **No data leakage**: StandardScaler fit on training fold only; windowing does not cross subject boundaries
- **Per-subject motion threshold**: motion stratification uses each subject's own median to control for inter-subject differences in baseline acceleration amplitude
