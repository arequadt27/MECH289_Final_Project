#!/usr/bin/env python3
"""
motion_stratification.py
─────────────────────────
Loads window_predictions.csv (produced by model_comparison.py) and computes
F1, recall, and precision for each model separately in low-motion and
high-motion windows.

Threshold: median of acc_magnitude_mean across all windows (data-driven,
as specified in the project proposal §5).

Outputs:
  motion_stratification.png  —  grouped bar chart per metric
  motion_stratification_results.csv
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


# CONFIGURATION
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))  # project folder


# LOAD PREDICTIONS
pred_path = os.path.join(DATA_ROOT, 'window_predictions.csv')
pred_df   = pd.read_csv(pred_path)

# Drop any windows with missing predictions (shouldn't occur after a full run)
pred_df = pred_df.dropna(subset=['lr_pred', 'rf_pred', 'xgb_pred'])


# MOTION STRATIFICATION
# Per-subject median: each subject's windows split 50/50 around their own baseline,
# preventing subjects with high baseline acceleration from being entirely classified
# as high-motion relative to the group.
pred_df['motion_threshold'] = pred_df.groupby('subject_id')['acc_magnitude_mean'].transform('median')
pred_df['motion_stratum'] = np.where(
    pred_df['acc_magnitude_mean'] > pred_df['motion_threshold'], 'High Motion', 'Low Motion'
)

per_subj_thresholds = pred_df.groupby('subject_id')['motion_threshold'].first()

print('=' * 65)
print('  Motion-Artifact Stratification Analysis')
print('=' * 65)
print(f'  Total windows        : {len(pred_df)}')
print(f'  Threshold            : per-subject median of acc_magnitude_mean')
print(f'  Per-subject thresholds (g):')
for sid, thr in per_subj_thresholds.items():
    print(f'    {sid}: {thr:.4f}')
n_low  = (pred_df['motion_stratum'] == 'Low Motion').sum()
n_high = (pred_df['motion_stratum'] == 'High Motion').sum()
print(f'  Low-motion windows   : {n_low}')
print(f'  High-motion windows  : {n_high}')


# COMPUTE METRICS PER STRATUM PER MODEL
MODEL_COLS   = {'LR': 'lr', 'RF': 'rf', 'XGBoost': 'xgb'}
STRATA       = ['Low Motion', 'High Motion']
METRIC_NAMES = ['f1', 'recall', 'precision']

records = []

for stratum in STRATA:
    subset  = pred_df[pred_df['motion_stratum'] == stratum]
    y_true  = subset['label'].astype(int)
    n_total  = len(y_true)
    n_stress = int(y_true.sum())

    print(f'\n--- {stratum} (n={n_total}, stress={n_stress} / {n_stress/n_total*100:.1f}%) ---')
    print(f'  {"Model":<12} {"F1":>7} {"Recall":>8} {"Precision":>11}')
    print(f'  {"-"*40}')

    for model_label, col_pfx in MODEL_COLS.items():
        y_pred = subset[f'{col_pfx}_pred'].astype(int)
        f1     = f1_score(y_true, y_pred,        pos_label=1, zero_division=0)
        rec    = recall_score(y_true, y_pred,    pos_label=1, zero_division=0)
        prec   = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        print(f'  {model_label:<12} {f1:>7.3f} {rec:>8.3f} {prec:>11.3f}')
        records.append({
            'stratum':   stratum,
            'model':     model_label,
            'n_windows': n_total,
            'n_stress':  n_stress,
            'f1':        round(f1,   4),
            'recall':    round(rec,  4),
            'precision': round(prec, 4),
        })

results_df = pd.DataFrame(records)

# PLOT — grouped bar chart per metric
models = list(MODEL_COLS.keys())
x      = np.arange(len(models))
bar_w  = 0.35
colors = {'Low Motion': 'steelblue', 'High Motion': 'tomato'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle(
    'Model Performance by Motion Stratum\n'
    '(threshold = per-subject median of acc_magnitude_mean)',
    fontsize=12, fontweight='bold',
)

for ax, metric in zip(axes, METRIC_NAMES):
    for i, (stratum, color) in enumerate(colors.items()):
        vals = [
            results_df.loc[
                (results_df['model'] == m) & (results_df['stratum'] == stratum),
                metric
            ].values[0]
            for m in models
        ]
        bars = ax.bar(x + i * bar_w, vals, bar_w,
                      label=stratum, color=color, alpha=0.85, edgecolor='white')
        # Label each bar with its value
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)

    # Proposal target line
    ax.axhline(0.80, color='gray', linestyle='--', linewidth=0.9, alpha=0.6,
               label='Target (0.80)')
    ax.set_title(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel(metric.capitalize(), fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(DATA_ROOT, 'motion_stratification.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nStratification plot saved -> {out_path}')


# SAVE RESULTS
out_csv = os.path.join(DATA_ROOT, 'motion_stratification_results.csv')
results_df.to_csv(out_csv, index=False)
print(f'Results saved -> {out_csv}')
