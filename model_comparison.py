#!/usr/bin/env python3
"""
model_comparison.py
────────────────────
Compares Logistic Regression, Random Forest, and XGBoost for WESAD stress
detection using LOSO cross-validation.

Additionally:
  - Saves per-window predictions to window_predictions.csv (used by
    motion_stratification.py)
  - Computes XGBoost SHAP values across all LOSO folds and saves
    shap_beeswarm.png and shap_bar.png

Install:
    pip install scikit-learn xgboost optuna shap

Expected runtime: ~20-40 min on a modern CPU.
"""

import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# CONFIGURATION
DATA_ROOT       = os.path.dirname(os.path.abspath(__file__))  # project folder
RANDOM_SEED     = 42
INNER_CV_FOLDS  = 3
N_OPTUNA_TRIALS = 50
C_GRID          = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# LOAD DATA

csv_path = os.path.join(DATA_ROOT, 'wesad_features.csv')
df       = pd.read_csv(csv_path)

meta_cols = ['label', 'subject_id']
feat_cols = [c for c in df.columns if c not in meta_cols]

X      = df[feat_cols].values.astype(np.float64)
y      = df['label'].values.astype(np.int32)
groups = df['subject_id'].values

n_subjects = len(np.unique(groups))
label_cts  = dict(zip(*np.unique(y, return_counts=True)))

print('=' * 70)
print('  LR vs Random Forest vs XGBoost  —  LOSO Cross-Validation')
print('=' * 70)
print(f'  Windows     : {len(df)}')
print(f'  Features    : {len(feat_cols)}')
print(f'  Subjects    : {n_subjects}')
print(f'  Classes     : {label_cts}  (0=non-stress, 1=stress)')
print(f'  Optuna trials per fold: {N_OPTUNA_TRIALS}')
print(f'  Expected runtime: 20-40 min\n')



# METRIC HELPER

def compute_metrics(y_true, y_pred, y_prob):
    return {
        'f1':        round(f1_score(y_true, y_pred,        pos_label=1, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred,    pos_label=1, zero_division=0), 4),
        'precision': round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'pr_auc':    round(average_precision_score(y_true, y_prob), 4),
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
    }



# MODEL TRAINERS

def train_lr(X_tr, y_tr, X_te, y_te, inner_cv):
    gs = GridSearchCV(
        LogisticRegression(class_weight='balanced', max_iter=2000,
                           solver='lbfgs', random_state=RANDOM_SEED),
        param_grid={'C': C_GRID},
        cv=inner_cv, scoring='f1', n_jobs=-1, refit=True,
    )
    gs.fit(X_tr, y_tr)
    y_pred = gs.predict(X_te)
    y_prob = gs.predict_proba(X_te)[:, 1]
    return compute_metrics(y_te, y_pred, y_prob), {'C': gs.best_params_['C']}, y_pred, y_prob


def train_rf(X_tr, y_tr, X_te, y_te, inner_cv):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 50, 500),
            max_depth        = trial.suggest_int('max_depth', 3, 20),
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20),
        )
        clf = RandomForestClassifier(**params, class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_SEED)
        return cross_val_score(clf, X_tr, y_tr, cv=inner_cv,
                               scoring='f1', n_jobs=1).mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    clf  = RandomForestClassifier(**best, class_weight='balanced',
                                  n_jobs=-1, random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return compute_metrics(y_te, y_pred, y_prob), best, y_pred, y_prob


def train_xgb(X_tr, y_tr, X_te, y_te, inner_cv, scale_pos_weight):
    """
    Optuna TPE search. Returns fitted clf as 5th element for SHAP.
    min_child_weight is XGBoost's equivalent of LightGBM's min_child_samples.
    """
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int('n_estimators', 50, 500),
            learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int('max_depth', 3, 10),
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
            subsample        = trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        )
        clf = XGBClassifier(**params, scale_pos_weight=scale_pos_weight,
                            eval_metric='logloss', verbosity=0,
                            n_jobs=1, random_state=RANDOM_SEED)
        return cross_val_score(clf, X_tr, y_tr, cv=inner_cv,
                               scoring='f1', n_jobs=1).mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    clf  = XGBClassifier(**best, scale_pos_weight=scale_pos_weight,
                         eval_metric='logloss', verbosity=0,
                         n_jobs=-1, random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    best_with_spw = {**best, 'scale_pos_weight': round(scale_pos_weight, 2)}
    return compute_metrics(y_te, y_pred, y_prob), best_with_spw, y_pred, y_prob, clf



# LOSO SETUP
logo      = LeaveOneGroupOut()
MODEL_COL = {'LR': 'lr', 'RF': 'rf', 'XGBoost': 'xgb'}

# Per-window predictions DataFrame — populated during the LOSO loop.
# Saved to window_predictions.csv for motion_stratification.py.
pred_df = df[['subject_id', 'label', 'acc_magnitude_mean']].copy()
for pfx in ['lr', 'rf', 'xgb']:
    pred_df[f'{pfx}_pred'] = np.nan
    pred_df[f'{pfx}_prob'] = np.nan

# SHAP accumulators: one (n_test × n_feat) array per fold
shap_values_list = []   # SHAP values computed on scaled test features
shap_X_list      = []   # original unscaled test features (for color axis)

records = []
wall_t0 = time.time()


# LOSO LOOP
for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
    sid     = groups[test_idx[0]]
    fold_t0 = time.time()
    print(f'Fold {fold_idx:>2}/{n_subjects}  |  test = {sid}')

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    inner_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                               random_state=RANDOM_SEED)

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    spw   = n_neg / max(n_pos, 1)

    trainers = [
        ('LR',      train_lr,  {}),
        ('RF',      train_rf,  {}),
        ('XGBoost', train_xgb, {'scale_pos_weight': spw}),
    ]

    for model_name, train_fn, kwargs in trainers:
        m_t0 = time.time()

        if model_name == 'XGBoost':
            m, bp, y_pred_fold, y_prob_fold, xgb_clf = train_fn(
                X_tr_sc, y_tr, X_te_sc, y_te, inner_cv, **kwargs)
            # SHAP values for positive class on this fold's test set
            explainer = shap.TreeExplainer(xgb_clf)
            sv = explainer.shap_values(X_te_sc)
            # Newer shap returns list [neg_class, pos_class] for binary clf
            if isinstance(sv, list):
                sv = sv[1]
            shap_values_list.append(sv)
            shap_X_list.append(X[test_idx])  
        else:
            m, bp, y_pred_fold, y_prob_fold = train_fn(
                X_tr_sc, y_tr, X_te_sc, y_te, inner_cv, **kwargs)

        elapsed = time.time() - m_t0
        print(f'  {model_name:<10} '
              f'F1={m["f1"]:.3f}  Recall={m["recall"]:.3f}  '
              f'Precision={m["precision"]:.3f}  PR-AUC={m["pr_auc"]:.3f}  '
              f'({elapsed:.0f}s)')
        records.append({'model': model_name, 'subject_id': sid,
                        **m, 'best_params': str(bp)})

        # Store per-window predictions (indices map directly to df rows)
        col = MODEL_COL[model_name]
        pred_df.iloc[test_idx, pred_df.columns.get_loc(f'{col}_pred')] = y_pred_fold
        pred_df.iloc[test_idx, pred_df.columns.get_loc(f'{col}_prob')] = y_prob_fold

    print(f'  Fold done in {time.time()-fold_t0:.0f}s\n')

print(f'Total elapsed: {(time.time()-wall_t0)/60:.1f} min')


# SAVE PER-WINDOW PREDICTIONS
pred_csv = os.path.join(DATA_ROOT, 'window_predictions.csv')
pred_df.to_csv(pred_csv, index=False)
print(f'Per-window predictions saved -> {pred_csv}')



# COMBINED RESULTS TABLE
results_df  = pd.DataFrame(records)
metric_cols = ['f1', 'recall', 'precision', 'pr_auc', 'accuracy']
models      = ['LR', 'RF', 'XGBoost']

print('\n' + '=' * 70)
print('  MEAN +/- STD ACROSS SUBJECTS')
print('=' * 70)
col_w = 18
print(f'{"Model":<12}' + ''.join(f'{m.upper():<{col_w}}' for m in metric_cols))
print('-' * (12 + col_w * len(metric_cols)))
for mname in models:
    sub = results_df[results_df['model'] == mname]
    row = f'{mname:<12}'
    for col in metric_cols:
        row += f'{sub[col].mean():.3f} +/- {sub[col].std():.3f}  '
    print(row)

print('\n' + '=' * 70)
print('  PER-SUBJECT F1')
print('=' * 70)
pivot   = results_df.pivot(index='subject_id', columns='model', values='f1')[models]
display = pd.concat([pivot,
                     pivot.mean().rename('Mean').to_frame().T,
                     pivot.std().rename('Std').to_frame().T])
fw = 12
print(f'{"Subject":<12}' + ''.join(f'{m:>{fw}}' for m in models))
print('-' * (12 + fw * len(models)))
for idx, row_vals in display.iterrows():
    print(f'{str(idx):<12}' + ''.join(f'{v:>{fw}.3f}' for v in row_vals))



# SHAP ANALYSIS — XGBoost
shap_matrix   = np.vstack(shap_values_list)   
shap_X_matrix = np.vstack(shap_X_list)         

n_shap_wins = shap_matrix.shape[0]
print(f'\nGenerating SHAP plots ({n_shap_wins} test windows across all folds)...')

# -- Beeswarm (summary) plot: direction and magnitude of each feature's effect --
shap.summary_plot(shap_matrix, shap_X_matrix,
                  feature_names=feat_cols,
                  max_display=len(feat_cols),
                  show=False)
plt.title('XGBoost SHAP — Feature Effects (all LOSO folds)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
beeswarm_path = os.path.join(DATA_ROOT, 'shap_beeswarm.png')
plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'SHAP beeswarm plot saved -> {beeswarm_path}')

# -- Bar chart: mean |SHAP| per feature (global importance ranking) --
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)   
sorted_idx    = np.argsort(mean_abs_shap)           

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(range(len(feat_cols)),
        mean_abs_shap[sorted_idx],
        color='steelblue', alpha=0.85)
ax.set_yticks(range(len(feat_cols)))
ax.set_yticklabels([feat_cols[i] for i in sorted_idx], fontsize=9)
ax.set_xlabel('Mean |SHAP value|  (impact on stress prediction)', fontsize=10)
ax.set_title('XGBoost Global Feature Importance (SHAP)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
bar_path = os.path.join(DATA_ROOT, 'shap_bar.png')
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'SHAP bar chart saved -> {bar_path}')



# CONFUSION MATRICES — aggregated across all LOSO folds
y_true_all = pred_df['label'].astype(int).values
cm_models  = [('LR', 'lr'), ('Random Forest', 'rf'), ('XGBoost', 'xgb')]

print('\n' + '=' * 70)
print('  AGGREGATED CONFUSION MATRICES (all LOSO folds)')
print('=' * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Aggregated Confusion Matrices — All LOSO Folds',
             fontsize=13, fontweight='bold')

for ax, (label, col) in zip(axes, cm_models):
    y_pred_all = pred_df[f'{col}_pred'].astype(int).values
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f'\n{label}:')
    print(f'  {"":>20} Pred non-stress   Pred stress')
    print(f'  True non-stress     {tn:>10d}      {fp:>8d}')
    print(f'  True stress         {fn:>10d}      {tp:>8d}')
    total = tn + fp + fn + tp
    print(f'  TP={tp} ({100*tp/total:.1f}%)  FP={fp} ({100*fp/total:.1f}%)  '
          f'TN={tn} ({100*tn/total:.1f}%)  FN={fn} ({100*fn/total:.1f}%)')

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['Non-Stress', 'Stress'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)

plt.tight_layout()
cm_path = os.path.join(DATA_ROOT, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nConfusion matrix figure saved -> {cm_path}')



# SAVE MODEL COMPARISON RESULTS
out_csv = os.path.join(DATA_ROOT, 'model_comparison_results.csv')
results_df.to_csv(out_csv, index=False)
print(f'\nModel comparison results saved -> {out_csv}')
