#!/usr/bin/env python3
"""
lr_pipeline.py
──────────────
Logistic Regression stress classifier with Leave-One-Subject-Out (LOSO)
cross-validation.  Loads wesad_features.csv produced by wesad_pipeline.py.

Design choices:
  - StandardScaler fit on the LOSO training set only (no data leakage)
  - Inner 3-fold stratified grid search selects regularisation strength C
  - class_weight='balanced' compensates for the ~88/12 class imbalance
  - Metrics reported for the stress class (label = 1) only
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# CONFIGURATION
DATA_ROOT      = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED    = 42
INNER_CV_FOLDS = 3
C_GRID         = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


# LOAD FEATURE MATRIX
csv_path = os.path.join(DATA_ROOT, 'wesad_features.csv')
df = pd.read_csv(csv_path)

meta_cols = ['label', 'subject_id']
feat_cols = [c for c in df.columns if c not in meta_cols]

X      = df[feat_cols].values.astype(np.float64)
y      = df['label'].values.astype(np.int32)
groups = df['subject_id'].values

unique_subjects = np.unique(groups)
label_counts    = dict(zip(*np.unique(y, return_counts=True)))

print('=' * 65)
print('  Logistic Regression — LOSO Cross-Validation')
print('=' * 65)
print(f'  CSV          : {csv_path}')
print(f'  Windows      : {len(df)}')
print(f'  Features     : {len(feat_cols)}')
print(f'  Subjects     : {len(unique_subjects)}')
print(f'  Class counts : {label_counts}  (0=non-stress, 1=stress)')
print(f'  C grid       : {C_GRID}')
print(f'  Inner folds  : {INNER_CV_FOLDS}')


# LOSO CROSS-VALIDATION
logo     = LeaveOneGroupOut()
inner_cv = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True,
                           random_state=RANDOM_SEED)

fold_results = []   # one dict per subject
all_y_true   = []   # accumulated across folds for confusion matrix
all_y_pred   = []
all_coefs    = []   # LR coefficients per fold (shape: n_folds × n_features)

header = (f'\n{"Subject":<10} {"F1":>7} {"Recall":>7} '
          f'{"Precision":>10} {"PR-AUC":>8} {"Accuracy":>10} {"Best C":>8}')
print(header)
print('-' * 65)

for train_idx, test_idx in logo.split(X, y, groups):
    test_subject = groups[test_idx[0]]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit scaler on training data only — prevents leakage into the test fold
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    # Inner grid search: tune C on the training subjects
    base_lr = LogisticRegression(
        class_weight='balanced',   # compensates for class imbalance
        max_iter=2000,
        solver='lbfgs',
        random_state=RANDOM_SEED,
    )
    gs = GridSearchCV(
        estimator=base_lr,
        param_grid={'C': C_GRID},
        cv=inner_cv,
        scoring='f1',              # optimise for f1
        n_jobs=-1,
        refit=True,                # refit best model on full training set
    )
    gs.fit(X_train_sc, y_train)

    best_C  = gs.best_params_['C']
    y_pred  = gs.predict(X_test_sc)
    y_prob  = gs.predict_proba(X_test_sc)[:, 1]   # probability of stress
    all_coefs.append(gs.best_estimator_.coef_[0])

    # Metrics for the stress class (pos_label=1)
    f1        = f1_score(y_test, y_pred,        pos_label=1, zero_division=0)
    recall    = recall_score(y_test, y_pred,    pos_label=1, zero_division=0)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    pr_auc    = average_precision_score(y_test, y_prob)
    accuracy  = accuracy_score(y_test, y_pred)

    print(f'{test_subject:<10} {f1:>7.3f} {recall:>7.3f} '
          f'{precision:>10.3f} {pr_auc:>8.3f} {accuracy:>10.3f} {best_C:>8}')

    fold_results.append({
        'subject_id': test_subject,
        'f1':         round(f1,        4),
        'recall':     round(recall,    4),
        'precision':  round(precision, 4),
        'pr_auc':     round(pr_auc,    4),
        'accuracy':   round(accuracy,  4),
        'best_C':     best_C,
        'n_test_windows': len(y_test),
        'n_stress_windows': int(y_test.sum()),
    })

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(y_pred.tolist())


# AGGREGATED RESULTS
results_df = pd.DataFrame(fold_results)

print('\n' + '=' * 65)
print('  MEAN +/- STD ACROSS SUBJECTS')
print('=' * 65)
metric_cols = ['f1', 'recall', 'precision', 'pr_auc', 'accuracy']
for col in metric_cols:
    m = results_df[col].mean()
    s = results_df[col].std()
    print(f'  {col:<12}: {m:.3f}  +/-  {s:.3f}')

# Aggregated confusion matrix (sum over all LOSO folds)
cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print('\nAggregated Confusion Matrix (all folds):')
print(f'                   Pred non-stress   Pred stress')
print(f'  True non-stress       {tn:>8d}      {fp:>8d}')
print(f'  True stress           {fn:>8d}      {tp:>8d}')

total   = tn + fp + fn + tp
print(f'\n  True Positives  (TP): {tp}   ({100*tp/total:.1f}%)')
print(f'  False Positives (FP): {fp}   ({100*fp/total:.1f}%)')
print(f'  True Negatives  (TN): {tn}   ({100*tn/total:.1f}%)')
print(f'  False Negatives (FN): {fn}   ({100*fn/total:.1f}%)')

# SAVE
out_csv = os.path.join(DATA_ROOT, 'logistic_regression_results.csv')
results_df.to_csv(out_csv, index=False)
print(f'\nResults saved -> {out_csv}')

# LR COEFFICIENT PLOT — mean +/- std across LOSO folds
coef_matrix = np.array(all_coefs)           # (n_subjects, n_features)
coef_mean   = coef_matrix.mean(axis=0)
coef_std    = coef_matrix.std(axis=0)

# Sort by absolute mean coefficient (most influential at top of chart)
sort_idx = np.argsort(np.abs(coef_mean))

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = np.arange(len(feat_cols))
ax.barh(y_pos, coef_mean[sort_idx], xerr=coef_std[sort_idx],
        color=['tomato' if v > 0 else 'steelblue' for v in coef_mean[sort_idx]],
        alpha=0.85, capsize=3, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_cols[i] for i in sort_idx], fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('LR Coefficient  (red = promotes stress, blue = suppresses)', fontsize=10)
ax.set_title('Logistic Regression Feature Coefficients\n(mean ± std across 15 LOSO folds)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
coef_path = os.path.join(DATA_ROOT, 'lr_coefficients.png')
plt.savefig(coef_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'LR coefficient plot saved -> {coef_path}')
