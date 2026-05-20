#!/usr/bin/env python3
"""
wesad_pipeline.py
─────────────────
WESAD stress detection pipeline — Empatica E4 wrist data only.
Signals: ACC (32 Hz) · BVP (64 Hz) · EDA (4 Hz) · TEMP (4 Hz)
Chest sensor data is never loaded or accessed at any point.

Usage:
    Set DATA_ROOT and PLOT_SUBJECT below, then:
        python wesad_pipeline.py
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')          # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from scipy.stats import linregress

warnings.filterwarnings('ignore')


# CONFIGURATION  ←  only these two lines need to change between runs
DATA_ROOT    = os.path.dirname(os.path.abspath(__file__))  # project folder (contains WESAD/)

# Subjects S2–S17; S12 is absent from the dataset
SUBJECTS = [f'S{i}' for i in range(2, 18) if i != 12]

# Nominal sampling rates (Hz)
FS = {
    'ACC':    32,
    'BVP':    64,
    'EDA':     4,
    'TEMP':    4,
    'labels': 700,
}

WINDOW_SEC   = 60     # window length in seconds
OVERLAP      = 0.50  # 50 % overlap → effective step = 30 s
VALID_LABELS = frozenset({1, 2, 3})   # TSST-relevant labels; exclude 0 (undefined/transition) and 5–7



# 1. DATA LOADING

def load_subject(subject_id: str) -> dict:
    """
    Load wrist-only signals and ground-truth labels for one WESAD subject.
    raw['signal']['chest'] is deliberately never referenced.
    """
    pkl_path = os.path.join(DATA_ROOT, 'WESAD', subject_id, f'{subject_id}.pkl')
    with open(pkl_path, 'rb') as fh:
        raw = pickle.load(fh, encoding='latin1')

    wrist  = raw['signal']['wrist']                        # ← wrist only
    labels = raw['label'].flatten().astype(np.int32)       # (N,) at 700 Hz

    return {
        'ACC':    wrist['ACC'].astype(np.float64),             # (N, 3) @ 32 Hz
        'BVP':    wrist['BVP'].flatten().astype(np.float64),   # (N,)   @ 64 Hz
        'EDA':    wrist['EDA'].flatten().astype(np.float64),   # (N,)   @  4 Hz
        'TEMP':   wrist['TEMP'].flatten().astype(np.float64),  # (N,)   @  4 Hz
        'labels': labels,
    }


def print_subject_summary(subject_id: str, data: dict) -> None:
    """Sanity-check printout: shapes and per-label sample counts."""
    counts  = {int(v): int(np.sum(data['labels'] == v))
               for v in np.unique(data['labels'])}
    dur_min = len(data['labels']) / FS['labels'] / 60
    print(f"\n{'-'*62}")
    print(f"  {subject_id}   (~{dur_min:.1f} min)")
    for sig in ('ACC', 'BVP', 'EDA', 'TEMP'):
        print(f"  {sig:<5}: shape {str(data[sig].shape):<14}  {FS[sig]} Hz")
    print(f"  labels: shape {str(data['labels'].shape):<12}  {FS['labels']} Hz")
    print(f"  label counts (raw 700 Hz): {counts}")



# 2. PREPROCESSING / FILTERING

def _lp_coeffs(cutoff: float, fs: float, order: int = 4):
    return butter(order, cutoff / (fs / 2), btype='low')


def _bp_coeffs(low: float, high: float, fs: float, order: int = 4):
    nyq = fs / 2
    return butter(order, [low / nyq, high / nyq], btype='band')


def filter_eda(eda: np.ndarray, fs: float = 4.0) -> np.ndarray:
    """4th-order Butterworth LP at 1 Hz, zero-phase via filtfilt."""
    b, a = _lp_coeffs(1.0, fs, order=4)
    return filtfilt(b, a, eda)


def filter_bvp(bvp: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """
    4th-order Butterworth BP 0.5–5 Hz (zero-phase).
    60 Hz power-line notch is applied only when Nyquist > 60 Hz.
    For BVP at 64 Hz the Nyquist is 32 Hz < 60 Hz, so the notch
    is intentionally skipped — 60 Hz is not representable at this rate.
    """
    b, a = _bp_coeffs(0.5, 5.0, fs, order=4)
    out  = filtfilt(b, a, bvp)

    if (fs / 2) > 60.0:
        w0       = 60.0 / (fs / 2)
        b_n, a_n = iirnotch(w0, Q=30)
        out      = filtfilt(b_n, a_n, out)
    # else: notch skipped — BVP Nyquist (32 Hz) < 60 Hz

    return out


def smooth_rolling(sig: np.ndarray, window: int = 5) -> np.ndarray:
    """Centre-aligned rolling mean; min_periods=1 avoids NaN at edges."""
    return (pd.Series(sig)
              .rolling(window, center=True, min_periods=1)
              .mean()
              .to_numpy())


def preprocess_subject(data: dict) -> dict:
    """Compute ACC magnitude and apply all modality-specific filters."""
    acc_raw = np.sqrt((data['ACC'] ** 2).sum(axis=1))   # Euclidean magnitude

    return {
        # Raw signals kept for diagnostic plotting only
        'EDA_raw':  data['EDA'],
        'BVP_raw':  data['BVP'],
        'TEMP_raw': data['TEMP'],
        'ACC_raw':  acc_raw,
        # Filtered / smoothed signals used for feature extraction
        'EDA':      filter_eda(data['EDA'],    fs=FS['EDA']),
        'BVP':      filter_bvp(data['BVP'],    fs=FS['BVP']),
        'TEMP':     smooth_rolling(data['TEMP'], window=5),
        'ACC':      smooth_rolling(acc_raw,      window=5),
        'labels':   data['labels'],
    }



# 3. LABEL ALIGNMENT & BINARY REMAPPING

def downsample_labels_majority(labels: np.ndarray,
                                src_fs: int, tgt_fs: int) -> np.ndarray:
    """
    Downsample label signal from src_fs → tgt_fs via majority vote.
    Works for non-integer decimation ratios (e.g. 700 Hz → 64 Hz).

    Strategy: assign every source sample to its target time bin with
    integer arithmetic, then iterate only over the (small) number of
    unique bins — O(n_src + n_tgt) rather than O(n_src * n_tgt).
    """
    n_src    = len(labels)
    n_tgt    = int(n_src * tgt_fs / src_fs)

    # Integer bin assignment — avoids floating-point drift
    src_idx  = np.arange(n_src, dtype=np.int64)
    tgt_bins = (src_idx * tgt_fs // src_fs).astype(np.int32).clip(0, n_tgt - 1)

    max_lbl  = max(1, int(labels.clip(0).max()) + 1)
    out      = np.zeros(n_tgt, dtype=np.int32)

    # Change-point iteration: group consecutive elements with the same bin
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(tgt_bins)) + 1, [n_src]))
    for j in range(len(bounds) - 1):
        s, e        = bounds[j], bounds[j + 1]
        bin_id      = int(tgt_bins[s])
        chunk       = labels[s:e].clip(0).astype(np.intp)
        out[bin_id] = int(np.bincount(chunk, minlength=max_lbl).argmax())

    return out


def binarize(labels: np.ndarray) -> np.ndarray:
    """
    Stress (label 2) → 1; all others → 0.
    Others: 0 (undefined/transition), 1 (baseline), 3 (amusement), 4 (meditation).
    NOTE: label 4 (meditation) is kept as non-stress for now;
          its boundary with the stress condition should be evaluated
          experimentally in later analysis.
    """
    return (labels == 2).astype(np.int32)


def align_labels(proc: dict) -> tuple:
    """
    Return (raw_aligned, bin_aligned) at each modality's sampling rate.
    raw_aligned : pre-binarization labels — used to filter non-TSST windows.
    bin_aligned : binary stress labels (0/1) — used for training.
    """
    raw_lbl = proc['labels']
    raw_aligned, bin_aligned = {}, {}
    for mod in ('EDA', 'BVP', 'TEMP', 'ACC'):
        ds               = downsample_labels_majority(raw_lbl, FS['labels'], FS[mod])
        raw_aligned[mod] = ds
        bin_aligned[mod] = binarize(ds)
    return raw_aligned, bin_aligned



# 4. SIGNAL VISUALISATION

def _shade_stress(ax, lbl: np.ndarray, fs: float) -> None:
    """Add red transparent shading wherever binary label == 1 (stress)."""
    t = np.arange(len(lbl)) / fs
    in_stress, t0 = False, 0.0
    for ti, lb in zip(t, lbl):
        if lb == 1 and not in_stress:
            t0 = ti
            in_stress = True
        elif lb == 0 and in_stress:
            ax.axvspan(t0, ti, alpha=0.15, color='red', zorder=0)
            in_stress = False
    if in_stress:
        ax.axvspan(t0, t[-1], alpha=0.15, color='red', zorder=0)


def plot_subject_signals(subject_id: str, proc: dict,
                         aligned: dict, save_dir: str) -> None:
    """
    2×2 figure: raw (light gray) overlaid with filtered (colour) per modality.
    Red shading marks stress windows (binary label == 1).
    Saved as signal_comparison_<subject_id>.png in save_dir.
    """
    panels = [
        ('EDA',  'EDA_raw',  'EDA',  'cornflowerblue', 'EDA (µS)',          FS['EDA']),
        ('BVP',  'BVP_raw',  'BVP',  'tomato',          'BVP (a.u.)',         FS['BVP']),
        ('TEMP', 'TEMP_raw', 'TEMP', 'mediumseagreen',  'Temperature (°C)',   FS['TEMP']),
        ('ACC',  'ACC_raw',  'ACC',  'darkorange',      'ACC Magnitude (g)',  FS['ACC']),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(
        f'WESAD Wrist Signals — {subject_id}  |  '
        f'raw = light gray  ·  filtered = colour  ·  red = stress (TSST)',
        fontsize=13, fontweight='bold',
    )

    for ax, (mod, raw_key, filt_key, color, ylabel, fs) in zip(axes.flat, panels):
        raw_sig  = proc[raw_key]
        filt_sig = proc[filt_key]
        t        = np.arange(len(raw_sig)) / fs

        _shade_stress(ax, aligned[mod], fs)

        ax.plot(t, raw_sig,  color='lightgray', lw=0.5, label='Raw',      zorder=1)
        ax.plot(t, filt_sig, color=color,        lw=0.9, label='Filtered', zorder=2)

        ax.set_title(mod, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(t[0], t[-1])

    stress_patch = mpatches.Patch(color='red', alpha=0.3, label='Stress (TSST — label 2)')
    fig.legend(handles=[stress_patch], loc='lower center',
               ncol=1, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(save_dir, f'signal_comparison_{subject_id}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Diagnostic plot saved -> {out_path}')



# 5. WINDOWING

def make_windows(sig: np.ndarray, labels: np.ndarray,
                 fs: int, window_sec: int = 60,
                 overlap: float = 0.5,
                 raw_labels: np.ndarray = None):
    """
    Slide a fixed-length window over sig and its aligned binary labels.

    Only complete windows are kept (no zero-padding).
    All windows belong to a single subject — cross-subject windows
    are prevented by calling this function per-subject.
    Window label = majority vote of per-sample labels within the window.

    If raw_labels is provided, windows whose majority raw label is not in
    VALID_LABELS are discarded (removes undefined/transition label-0 windows
    and any non-TSST labels 5-7, aligning with the proposal's truncation intent).

    Returns
    -------
    windows    : list[np.ndarray]  each of length window_sec * fs
    win_labels : list[int]         majority-vote binary label per window
    """
    n_win  = int(window_sec * fs)
    n_step = int(n_win * (1.0 - overlap))
    windows, win_labels = [], []

    for start in range(0, len(sig) - n_win + 1, n_step):
        if raw_labels is not None:
            raw_chunk    = raw_labels[start : start + n_win]
            raw_majority = int(np.bincount(raw_chunk.clip(0).astype(np.intp)).argmax())
            if raw_majority not in VALID_LABELS:
                continue

        lbl_chunk = labels[start : start + n_win]
        windows.append(sig[start : start + n_win])
        win_labels.append(int(np.bincount(lbl_chunk.astype(np.intp)).argmax()))

    return windows, win_labels



# 6. FEATURE EXTRACTION

def feat_eda(win: np.ndarray, fs: float) -> dict:
    """
    EDA features: mean, std, min, max, linear slope,
    SCR peak count, mean SCR peak amplitude.
    """
    t         = np.arange(len(win)) / fs
    slope, *_ = linregress(t, win)
    peaks, _  = find_peaks(win, prominence=0.01)
    mean_amp  = float(np.mean(win[peaks])) if len(peaks) > 0 else 0.0

    return {
        'eda_mean':          float(np.mean(win)),
        'eda_std':           float(np.std(win)),
        'eda_min':           float(np.min(win)),
        'eda_max':           float(np.max(win)),
        'eda_slope':         float(slope),
        'eda_peak_count':    int(len(peaks)),
        'eda_mean_peak_amp': mean_amp,
    }


def feat_bvp(win: np.ndarray, fs: float) -> dict:
    """
    BVP features after ectopic-beat rejection.

    Steps:
      1. Detect systolic peaks (minimum 0.3 s apart, ≤ ~200 bpm).
      2. Compute RR intervals (s).
      3. Reject ectopic beats: remove RR intervals that deviate > 20 %
         from the local (window-level) median.
      4. Return signal mean/std and cleaned RR statistics.
    """
    min_dist = max(1, int(0.3 * fs))
    peaks, _ = find_peaks(win, distance=min_dist, prominence=0.01)

    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])

    if len(rr) > 1:
        med      = np.median(rr)
        valid    = np.abs(rr - med) / (med + 1e-9) < 0.20
        rr_clean = rr[valid]
    else:
        rr_clean = rr

    mean_rr = float(np.mean(rr_clean)) if len(rr_clean) > 0 else 0.0
    std_rr  = float(np.std(rr_clean))  if len(rr_clean) > 0 else 0.0

    return {
        'bvp_mean':             float(np.mean(win)),
        'bvp_std':              float(np.std(win)),
        'bvp_peak_count':       int(len(peaks)),
        'bvp_mean_rr_interval': mean_rr,
        'bvp_std_rr_interval':  std_rr,
    }


def feat_temp(win: np.ndarray, fs: float) -> dict:
    """TEMP features: mean, std, linear slope."""
    t         = np.arange(len(win)) / fs
    slope, *_ = linregress(t, win)
    return {
        'temp_mean':  float(np.mean(win)),
        'temp_std':   float(np.std(win)),
        'temp_slope': float(slope),
    }


def feat_acc(win: np.ndarray, fs: float) -> dict:
    """
    ACC magnitude features: mean, std, max, signal energy (sum of squares).
    acc_magnitude_mean is preserved explicitly for motion-artifact
    stratification analysis in downstream experiments.
    """
    return {
        'acc_magnitude_mean':   float(np.mean(win)),    # retained for motion stratification
        'acc_magnitude_std':    float(np.std(win)),
        'acc_magnitude_max':    float(np.max(win)),
        'acc_magnitude_energy': float(np.sum(win ** 2)),
    }


_FEAT_FN = {
    'EDA':  feat_eda,
    'BVP':  feat_bvp,
    'TEMP': feat_temp,
    'ACC':  feat_acc,
}


def extract_features(proc: dict, raw_aligned: dict,
                     bin_aligned: dict, subject_id: str) -> pd.DataFrame:
    """
    Window each modality independently at its own sampling rate, extract
    features, and assemble one feature row per window.

    raw_aligned is used to filter windows dominated by non-TSST labels
    (label 0 transitions, labels 5-7). bin_aligned provides the binary
    stress label for each kept window.

    Window counts may differ by +-1 across modalities due to rounding;
    the minimum is used to keep all modalities time-aligned.
    """
    MODS = [
        ('EDA',  FS['EDA']),
        ('BVP',  FS['BVP']),
        ('TEMP', FS['TEMP']),
        ('ACC',  FS['ACC']),
    ]

    all_wins, all_lbls = {}, {}
    for mod, fs in MODS:
        wins, lbls    = make_windows(proc[mod], bin_aligned[mod], fs,
                                     WINDOW_SEC, OVERLAP,
                                     raw_labels=raw_aligned[mod])
        all_wins[mod] = wins
        all_lbls[mod] = lbls

    n = min(len(v) for v in all_wins.values())
    print(f'    windows per modality: EDA={len(all_wins["EDA"])} '
          f'BVP={len(all_wins["BVP"])} '
          f'TEMP={len(all_wins["TEMP"])} '
          f'ACC={len(all_wins["ACC"])}  ->  using {n}')

    rows = []
    for i in range(n):
        row        = {}
        lbl_votes  = []
        for mod, fs in MODS:
            row.update(_FEAT_FN[mod](all_wins[mod][i], fs))
            lbl_votes.append(all_lbls[mod][i])
        row['label']      = int(np.bincount(np.array(lbl_votes, dtype=np.intp)).argmax())
        row['subject_id'] = subject_id
        rows.append(row)

    return pd.DataFrame(rows)



# 7. MAIN PIPELINE

def main() -> None:
    print('=' * 65)
    print('  WESAD Wrist-only Stress Detection Pipeline')
    print('=' * 65)
    print(f'  DATA_ROOT    : {DATA_ROOT}')
    print(f'  Subjects     : {SUBJECTS}')

    frames = []

    for sid in SUBJECTS:
        print(f'\n[{sid}] Loading ...')
        try:
            data = load_subject(sid)
        except FileNotFoundError as exc:
            print(f'  WARNING: skipping {sid} — {exc}')
            continue

        print_subject_summary(sid, data)

        print(f'[{sid}] Preprocessing ...')
        proc                    = preprocess_subject(data)
        raw_aligned, bin_aligned = align_labels(proc)

        print(f'[{sid}] Generating diagnostic plot ...')
        signals_dir = os.path.join(DATA_ROOT, 'Subject Signals')
        os.makedirs(signals_dir, exist_ok=True)
        plot_subject_signals(sid, proc, bin_aligned, save_dir=signals_dir)

        print(f'[{sid}] Extracting features ...')
        df = extract_features(proc, raw_aligned, bin_aligned, sid)
        frames.append(df)

        counts = df['label'].value_counts().sort_index().to_dict()
        print(f'    {sid}: {len(df)} windows  ->  class counts {counts}')

    if not frames:
        print('\nERROR: No subjects loaded successfully. Check DATA_ROOT path.')
        return

    # Concatenate all subjects 
    full = pd.concat(frames, ignore_index=True)

    meta_cols = ['label', 'subject_id']
    feat_cols = [c for c in full.columns if c not in meta_cols]

    # Save feature matrix
    out_csv = os.path.join(DATA_ROOT, 'wesad_features.csv')
    full[feat_cols + meta_cols].to_csv(out_csv, index=False)
    print(f'\nFeature matrix saved -> {out_csv}')

    # Final summary 
    print('\n' + '=' * 65)
    print('  FINAL SUMMARY')
    print('=' * 65)
    print(f'Total windows      : {len(full)}')
    print(f'Features per window: {len(feat_cols)}')
    print(f'Feature names      :\n  {feat_cols}')

    print('\nGlobal class distribution:')
    vc = full['label'].value_counts().sort_index()
    for lbl, cnt in vc.items():
        pct = 100 * cnt / len(full)
        tag = 'stress' if lbl == 1 else 'non-stress'
        print(f'  label {lbl} ({tag:>10s}): {cnt:>6d} windows  ({pct:.1f} %)')

    print('\nPer-subject window counts (class 0 / class 1):')
    per_subj = (
        full.groupby('subject_id')['label']
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={0: 'non-stress', 1: 'stress'})
    )
    print(per_subj.to_string())

    # Flag subjects with unusually few windows (< 50 % of median)
    win_counts = full.groupby('subject_id').size()
    threshold  = win_counts.median() * 0.50
    low        = win_counts[win_counts < threshold]
    if not low.empty:
        print(f'\nWARNING — subjects with < 50 % of median window count:')
        print(low.to_string())
    else:
        print('\nNo subjects with unusually low window counts.')


if __name__ == '__main__':
    main()
