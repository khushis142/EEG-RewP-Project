"""
ULTIMATE ERP ANALYSIS PIPELINE – v3
Team NeuroSimha – EEG Casinos Task

Merged v3: Combines v1 (TFA / Time-Frequency Analysis, GA TFR heatmaps,
methodology CSV) with v2 (ICA caching, ML single-trial decoding, robust
statistics, memory-efficient gc.collect, cohort split, SEM shading, ICA
audit report, overlay sanity check).

Architecture overview
---------------------
  preliminary-analysis-script_v3.py  →  good_subjects.json
                                              ↓
               complete-analysis-script_v3.py   (this file)
                  ├─ preprocess_subject()
                  ├─ parse_events_by_condition()
                  ├─ generate_individual_plots()   ← +joint plot (new v3)
                  ├─ plot_epoch_drop_log()          ← new in v3
                  ├─ run_time_frequency_analysis()      [from v1]
                  ├─ run_single_trial_ml_decoding()     [from v2]
                  ├─ generate_grand_average_visuals()   ← +RewP modulation curve (new v3)
                  ├─ run_robust_statistics_and_ml()     [from v2]
                  └─ generate_methodology_csv()

Key parameter decisions
-----------------------
  Rejection threshold : 100 µV (selected after running the check_epoch_rejection.py script)
  ICA random_state    : 97  (v2 value; keeps results reproducible across runs)
  ICA n_components    : 20
  ICA method          : infomax
  Reference           : mastoid (TP9+TP10), applied AFTER ICA
  Low-pass for epochs : 30 Hz (FIR, applied after ICA)
  BIDS_ROOT           : './data/'  (v2 path, matches preliminary script)
"""

import gc
import json
import warnings

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & PATH HANDLING
# ============================================================================

# Auto-detect project root (script is in scripts/ folder)
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent

# Check for BIDS data in expected locations
if (project_root / 'data' / 'ds004147').exists():
    BIDS_ROOT = project_root / 'data' / 'ds004147'
elif (project_root / 'data').exists():
    BIDS_ROOT = project_root / 'data'
else:
    # Fallback to CWD relative (legacy)
    BIDS_ROOT = Path('./data/ds004147') if Path('./data/ds004147').exists() else Path('./data')

OUTPUT_DIR = project_root / 'results' / 'complete_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subject list is overridden at runtime by good_subjects.json if available
SUBJECT_IDS = [str(i) for i in range(27, 39)]

REJECT_THRESHOLD = 100e-6   # V; peak-to-peak epoch rejection (100 µV)
PRIMARY_CHANNEL  = 'FCz'
REWP_WINDOW      = (0.24, 0.32)   # s
BASELINE         = (-0.2, 0.0)    # s

# All eight feedback-outcome × task-context conditions
ALL_CONDITIONS = [
    'low_low_win', 'low_low_loss',
    'mid_low_win', 'mid_low_loss',
    'mid_high_win', 'mid_high_loss',
    'high_high_win', 'high_high_loss',
]

# TFA parameters
TFA_FREQS   = np.arange(1, 21, 1)  # Hz; 1–20 Hz covers delta & theta
TFA_CYCLES  = TFA_FREQS / 2.0      # Morlet cycles (half the frequency)
THETA_BAND  = (4, 8)               # Hz
TFA_WINDOW  = (0.2, 0.5)           # s; post-stimulus window for theta stats

# ============================================================================
# DATA HANDLING
# ============================================================================

def parse_events_by_condition(subject_id, raw):
    """
    Build a per-condition MNE events array from the BIDS events.tsv.

    Each condition receives a unique integer code (1–8) so that
    mne.Epochs can subset by condition name. The TSV is preferred over
    raw EEG annotations because annotations may contain duplicate
    timestamps that confuse MNE's epoch-creation routine.

    Also returns a performance dict (choice accuracy, win rates) that
    is used to decide cohort membership when good_subjects.json is absent.

    Returns
    -------
    events     : np.ndarray  shape (n_events, 3)
    event_id   : dict        {condition_name: int_code}
    performance: dict        {choice_accuracy, mh_win_rate, h_win_rate}
    """
    ev_file = (Path(BIDS_ROOT) / f'sub-{subject_id}' / 'eeg'
               / f'sub-{subject_id}_task-casinos_events.tsv')
    df   = pd.read_csv(ev_file, sep='\t')
    sfreq = raw.info['sfreq']
    stim  = df[df['trial_type'].str.contains('Stimulus', na=False)].copy()

    # Trigger-code → condition mapping
    event_codes = {
        'low_low_win':  (r'S\s+6$',  1),
        'low_low_loss': (r'S\s+7$',  2),
        'mid_low_win':  (r'S\s16$',  3),
        'mid_low_loss': (r'S\s17$',  4),
        'mid_high_win': (r'S\s26$',  5),
        'mid_high_loss':(r'S\s27$',  6),
        'high_high_win':(r'S\s36$',  7),
        'high_high_loss':(r'S\s37$', 8),
    }

    all_events, event_id = [], {}
    for cond, (pattern, code) in event_codes.items():
        matches = stim[stim['value'].str.match(pattern, na=False)]
        if len(matches) > 0:
            samples = (matches['onset'].values * sfreq).astype(int)
            evs = np.column_stack([
                samples,
                np.zeros(len(samples), int),
                np.full(len(samples), code, int),
            ])
            all_events.append(evs)
            event_id[cond] = code

    if not all_events:
        raise ValueError(f"No events found for sub-{subject_id}!")

    events = np.vstack(all_events)
    events = events[np.argsort(events[:, 0])]

    # Derive performance metrics for cohort flagging
    mh_trials = stim[stim['value'].str.match(r'S\s(26|27)$', na=False)]
    ml_trials = stim[stim['value'].str.match(r'S\s(16|17)$', na=False)]
    hi_trials = stim[stim['value'].str.match(r'S\s(36|37)$', na=False)]

    total_high_cue = len(mh_trials) + len(ml_trials)
    choice_accuracy = (
        len(mh_trials) / total_high_cue * 100 if total_high_cue > 0 else 0.0
    )
    mh_win_rate = (
        len(mh_trials[mh_trials['value'].str.match(r'S\s26$', na=False)])
        / len(mh_trials) * 100 if len(mh_trials) > 0 else 0.0
    )
    h_win_rate = (
        len(hi_trials[hi_trials['value'].str.match(r'S\s36$', na=False)])
        / len(hi_trials) * 100 if len(hi_trials) > 0 else 0.0
    )

    performance = {
        'choice_accuracy': choice_accuracy,
        'mh_win_rate':     mh_win_rate,
        'h_win_rate':      h_win_rate,
    }

    return events, event_id, performance

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_subject(subject_id, out_dir):
    """
    Full preprocessing pipeline for one subject.

    Order of operations
    -------------------
    1.  Load raw BIDS file; set EasyCap-M1 montage.
    2.  Resample to 250 Hz if needed (speeds up ICA considerably).
    3.  Band-pass filter 0.1–100 Hz + notch at 50 Hz (FIR).
        Rationale: broad band before ICA preserves muscle artefact
        components; we low-pass to 30 Hz after ICA removal.
    4.  Save pre-ICA power spectrum as a sanity check plot.
    5.  Run Infomax ICA (20 components, random_state=97).
        ICA solution is cached to disk so reruns are instant.
    6.  ICLabel auto-classification; exclude everything except 'brain'
        and 'other'.
    7.  Save all-component topographies (audit) + ICA variance report.
    8.  Save ICA overlay (before vs after) to verify reconstruction.
    9.  Apply ICA; re-filter to 0.1–30 Hz for ERP analysis.
    10. Re-reference to average of TP9 + TP10 (linked mastoids).
        Mastoid reference is applied after ICA so artefact removal is
        performed on average-referenced data (better ICA separation).

    Returns
    -------
    raw_clean : mne.io.Raw   (preprocessed, for epoching)
    labels    : list[str]    (ICLabel classifications, 20 entries)
    excluded  : list[int]    (component indices removed)
    """
    print(f"  [1/3] Loading & filtering (0.1–100 Hz + 50 Hz notch)...")
    b_path = BIDSPath(
        subject=subject_id, task='casinos',
        datatype='eeg', root=BIDS_ROOT,
    )
    raw = read_raw_bids(b_path, verbose=False).load_data()
    
    raw.set_montage('easycap-M1', verbose=False)

    if raw.info['sfreq'] != 250:
        raw.resample(250, verbose=False)

    viz_data = {}
    if PRIMARY_CHANNEL in raw.ch_names:
        viz_data['raw'] = raw.copy().pick([PRIMARY_CHANNEL]).crop(tmin=0, tmax=10).get_data()[0] * 1e6
        psd_b = raw.compute_psd(picks=[PRIMARY_CHANNEL], fmax=70, verbose=False)
        viz_data['freqs'] = psd_b.freqs
        viz_data['psd_raw'] = 10 * np.log10(np.maximum(psd_b.get_data()[0] * 1e12, 1e-10))
        viz_data['times'] = np.linspace(0, 10, len(viz_data['raw']))

    raw.filter(l_freq=0.1, h_freq=100, method='fir', verbose=False)
    raw.notch_filter(freqs=[50], method='fir', verbose=False)

    if PRIMARY_CHANNEL in raw.ch_names:
        viz_data['filt'] = raw.copy().pick([PRIMARY_CHANNEL]).crop(tmin=0, tmax=10).get_data()[0] * 1e6

    # Pre-ICA power spectrum (sanity check)
    try:
        fig_psd = raw.compute_psd().plot(show=False)
        fig_psd.savefig(out_dir / 'spectrum_pre_ica.png', dpi=100)
        plt.close(fig_psd)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # ICA with caching
    # ------------------------------------------------------------------
    print(f"  [2/3] ICA (Infomax, 20 comps, random_state=97)...")
    ica_path = out_dir / 'ica_solution.fif'
    if ica_path.exists():
        print(f"    → Using cached solution")
        ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    else:
        ica = ICA(n_components=20, method='infomax',
                  random_state=97, max_iter='auto')
        ica.fit(raw, verbose=False)
        ica.save(ica_path, overwrite=True, verbose=False)

    ic_labels = label_components(raw, ica, method='iclabel')
    labels    = ic_labels['labels']
    ica.exclude = [i for i, lbl in enumerate(labels)
                   if lbl not in ['brain', 'other']]

    # All-component topography map (audit)
    fig_comp = ica.plot_components(show=False)
    fig_comp.suptitle(
        f"sub-{subject_id} ICA Component Topographies\nLabels: {labels}"
    )
    fig_comp.savefig(out_dir / 'ica_all_components.png', dpi=120)
    plt.close(fig_comp)

    # Variance audit report
    sources   = ica.get_sources(raw).get_data()
    total_var = np.sum(np.var(sources, axis=1))
    report_lines = [
        f"ICA AUDIT REPORT – sub-{subject_id}",
        f"Excluded: {len(ica.exclude)}/20 components", "",
    ]
    for i in range(20):
        v = (np.var(sources[i]) / total_var) * 100
        tag = '[X]' if i in ica.exclude else '[ ]'
        report_lines.append(
            f"IC {i:02d}: {v:4.1f}% var | {labels[i]:15} | {tag}"
        )
    with open(out_dir / 'ica_audit_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))

    # Overlay: before vs after ICA (visual sanity check)
    fig_ovl = ica.plot_overlay(
        raw, show=False,
        title=f"sub-{subject_id} ICA Reconstruction Check",
    )
    fig_ovl.savefig(out_dir / 'ica_cleaning_overlay.png', dpi=120)
    plt.close(fig_ovl)

    # Apply, low-pass, re-reference
    print(f"  Pre-ICA max amplitude: {raw.get_data().max()*1e6:.1f} µV")
    ica.apply(raw, verbose=False)
    print(f"  Post-ICA max amplitude: {raw.get_data().max()*1e6:.1f} µV")
    raw.filter(l_freq=0.1, h_freq=30, method='fir', verbose=False)

    mastoids = [ch for ch in ['TP9', 'TP10'] if ch in raw.ch_names]
    if len(mastoids) == 2:
        raw.set_eeg_reference(mastoids, verbose=False)

    if PRIMARY_CHANNEL in raw.ch_names:
        viz_data['clean'] = raw.copy().pick([PRIMARY_CHANNEL]).crop(tmin=0, tmax=10).get_data()[0] * 1e6
        psd_a = raw.compute_psd(picks=[PRIMARY_CHANNEL], fmax=70, verbose=False)
        viz_data['psd_clean'] = 10 * np.log10(np.maximum(psd_a.get_data()[0] * 1e12, 1e-10))

    print(f"    → Excluded {len(ica.exclude)} ICA components: {ica.exclude}")
    return raw, labels, ica.exclude, viz_data

# ============================================================================
# INDIVIDUAL-SUBJECT VISUALISATIONS
# ============================================================================

def generate_individual_plots(epochs, ev_dict, sub_id, out):
    """
    Produce per-subject ERP plots.

    Plots generated:
    1. Butterfly / GFP   – all channels, flags which time window is RewP
    2. RewP topography   – high-high win minus loss difference wave,
                           averaged over the RewP window; FCz marked
    3. ERP by condition  – 2×2 grid (one panel per task context); SEM shading
    4. ERP joint plot    – combined waveform + scalp topography for the
                           high-high win−loss difference wave; produced with
                           mne.Evoked.plot_joint so the topomap time point is
                           pinned to the centre of the RewP window. This gives
                           a compact single-figure summary of both the temporal
                           and spatial profile of the RewP, mirroring the
                           figure style used in the original paper.
    """
    # 1. Butterfly / GFP
    try:
        fig = epochs.average().plot(
            show=False,
            titles=f"sub-{sub_id} – Butterfly / GFP",
        )
        fig.savefig(out / 'butterfly_individual.png', dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: butterfly plot failed for sub-{sub_id}: {e}")

    # 2. RewP topography (win − loss difference, high-high context)
    if 'high_high_win' in ev_dict and 'high_high_loss' in ev_dict:
        try:
            diff = mne.combine_evoked(
                [ev_dict['high_high_win'], ev_dict['high_high_loss']],
                weights=[1, -1],
            )
            mask = np.zeros(diff.data.shape, dtype=bool)
            if PRIMARY_CHANNEL in diff.ch_names:
                mask[diff.ch_names.index(PRIMARY_CHANNEL), :] = True

            fig = diff.plot_topomap(
                times=np.mean(REWP_WINDOW), average=0.08,
                show=False, mask=mask,
                mask_params=dict(marker='o', markerfacecolor='white',
                                 markersize=10),
                colorbar=True, size=3,
            )
            fig.suptitle(f"sub-{sub_id} RewP Topography")
            fig.savefig(out / 'topography_rewp.png', dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"    Warning: topography failed for sub-{sub_id}: {e}")

    # 3. ERP by condition (SEM shading)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    groups = [
        ('Low-Low',   ['low_low_win',  'low_low_loss']),
        ('Mid-Low',   ['mid_low_win',  'mid_low_loss']),
        ('Mid-High',  ['mid_high_win', 'mid_high_loss']),
        ('High-High', ['high_high_win','high_high_loss']),
    ]
    for ax, (name, conds) in zip(axes.flatten(), groups):
        for c in conds:
            if c in ev_dict:
                d = epochs[c].get_data(picks=PRIMARY_CHANNEL).squeeze() * 1e6
                if d.ndim == 1:
                    d = d[np.newaxis, :]
                m = np.mean(d, axis=0)
                s = stats.sem(d, axis=0)
                ax.plot(epochs.times, m, label=c)
                ax.fill_between(epochs.times, m - s, m + s, alpha=0.15)
        ax.axvspan(*REWP_WINDOW, alpha=0.1, color='red', label='RewP window')
        ax.axvline(0, color='gray', linestyle='--', label='Stimulus onset')
        ax.set_title(f"{name} ERP ({PRIMARY_CHANNEL})")
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
    plt.tight_layout()
    fig.savefig(out / 'erp_subject_view.png', dpi=120)
    plt.close(fig)

    # 4. ERP joint plot (waveform + topomap) for the high-high difference wave
    if 'high_high_win' in ev_dict and 'high_high_loss' in ev_dict:
        try:
            diff = mne.combine_evoked(
                [ev_dict['high_high_win'], ev_dict['high_high_loss']],
                weights=[1, -1],
            )
            # plot_joint shows a waveform at PRIMARY_CHANNEL with topomap
            # insets pinned to the RewP peak. We fix the topomap time to the
            # centre of the RewP window so the spatial distribution is always
            # shown at the theoretically motivated latency.
            fig = diff.plot_joint(
                times=[np.mean(REWP_WINDOW)],
                ts_args=dict(picks=PRIMARY_CHANNEL),
                title=f"sub-{sub_id} RewP Joint Plot (High-High Win − Loss)",
                show=False,
            )
            fig.savefig(out / 'erp_joint_rewp.png', dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"    Warning: joint plot failed for sub-{sub_id}: {e}")

# ============================================================================
# TIME-FREQUENCY ANALYSIS  [from v1]
# ============================================================================

def run_time_frequency_analysis(epochs, subject_id, output_path):
    """
    Morlet wavelet TFR for all eight conditions (4 contexts × 2 outcomes).

    Rationale for parameters
    ------------------------
    - Frequencies 1–20 Hz: covers delta (1–4 Hz) and theta (4–8 Hz), the
      bands most implicated in reward prediction error / feedback processing.
    - n_cycles = freqs / 2: standard compromise between time and frequency
      resolution; gives ~250 ms temporal precision at 4 Hz.
    - Baseline: log-ratio relative to -200 to 0 ms pre-stimulus.
    - Peak theta power (4–8 Hz, 200–500 ms) is extracted as a scalar
      statistic for the methodology CSV.

    Returns
    -------
    tfr_objects : dict  {condition: mne.time_frequency.AverageTFR}
    tfr_stats   : list  [['TFR_Theta_<cond>', '<value>'], ...]
    """
    levels   = ['high_high', 'mid_high', 'mid_low', 'low_low']
    outcomes = ['win', 'loss']

    fig, axes = plt.subplots(4, 2, figsize=(15, 22))
    tfr_stats   = []
    tfr_objects = {}

    for row, level in enumerate(levels):
        for col, outcome in enumerate(outcomes):
            cond = f"{level}_{outcome}"
            ax   = axes[row, col]

            if cond in epochs.event_id and len(epochs[cond]) > 0:
                # power = epochs[cond].compute_tfr(
                #     method='morlet', freqs=TFA_FREQS, n_cycles=TFA_CYCLES,
                #     return_itc=False, average=True,
                #     verbose=False,
                # )
                # power.apply_baseline(
                #     baseline=BASELINE, mode='logratio', verbose=False
                #)
            
                
                # tfr_objects[cond] = power

                # ch_idx = power.ch_names.index(PRIMARY_CHANNEL)
                # power.plot(
                #     [ch_idx], baseline=None, mode='logratio',
                #     axes=ax, show=False, colorbar=False,
                # )

                #----
                power = epochs[cond].compute_tfr(
                    method='morlet',
                    freqs=TFA_FREQS,
                    n_cycles=TFA_CYCLES,
                    return_itc=False,
                    average=True,
                    verbose=False,
                )

                power.apply_baseline(baseline=BASELINE, mode='logratio', verbose=False)

                tfr_objects[cond] = power

                # SAFE channel handling
                if PRIMARY_CHANNEL in power.ch_names:
                    ch_idx = power.ch_names.index(PRIMARY_CHANNEL)

                    power.plot(
                        [ch_idx],
                        baseline=None,
                        mode=None,
                        axes=ax,
                        show=False,
                        colorbar=False,
                    )
                else:
                    ax.set_title(f"{cond} ({PRIMARY_CHANNEL} not found)")
                    ax.axis('off')
                    continue
                #---

                # Frequency band markers
                ax.axhline(4, color='white', linestyle='--', alpha=0.5)
                ax.axhline(8, color='white', linestyle='--', alpha=0.5)
                ax.text(-0.15, 2.5, 'DELTA', color='black',
                        fontweight='bold', fontsize=8, va='center')
                ax.text(-0.15, 6,   'THETA', color='black',
                        fontweight='bold', fontsize=8, va='center')

                # Peak theta power statistic
                theta_crop = power.copy().crop(
                    tmin=TFA_WINDOW[0], tmax=TFA_WINDOW[1],
                    fmin=THETA_BAND[0], fmax=THETA_BAND[1],
                )
                peak_theta = float(np.max(theta_crop.data))
                tfr_stats.append([f"TFR_Theta_{cond}", f"{peak_theta:.4f}"])
                ax.set_title(
                    f"{level.upper()} {outcome.upper()}\n"
                    f"Peak Theta: {peak_theta:.2f}"
                )
            else:
                ax.set_title(f"{cond} (missing / no epochs)")
                ax.axis('off')

    fig.suptitle(
        f"sub-{subject_id}: Time-Frequency Oscillations at {PRIMARY_CHANNEL}",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(output_path / f"sub-{subject_id}_TFR_grid.png", dpi=150)
    plt.close(fig)

    return tfr_objects, tfr_stats

# ============================================================================
# METHODOLOGY CSV  [from v1]
# ============================================================================

def generate_methodology_csv(sub_id, labels, excluded, epochs, ev_dict,
                              diffs, tfr_stats, out):
    """
    Write a structured CSV documenting all key processing choices and
    resulting statistics for one subject.

    Each row follows the schema: [Step, Parameter, Value, Notes].
    This makes the CSV easy to review as a table and easy to aggregate
    across subjects.
    """
    total    = len(epochs.selection) + len([x for x in epochs.drop_log if x])
    rej_pct  = ((total - len(epochs)) / total * 100) if total > 0 else 0.0

    rows = [
        ['Preprocessing', 'ICA method',           'Infomax',                   'random_state=97'],
        ['Preprocessing', 'ICA components',        20,                          ''],
        ['Preprocessing', 'ICA excluded',           len(excluded),               str(excluded)],
        ['Preprocessing', 'Reference',             'Mastoid (TP9+TP10)',         'Post-ICA'],
        ['Preprocessing', 'Low-pass (post-ICA)',   '30 Hz',                     'FIR'],
        ['Epoching',      'Rejection threshold',   f"{REJECT_THRESHOLD*1e6} µV", 'Peak-to-peak'],
        ['Epoching',      'Total trials',           total,                       ''],
        ['Epoching',      'Kept trials',
         f"{len(epochs)} ({100 - rej_pct:.1f}%)",
         f"Dropped: {rej_pct:.1f}%"],
    ]

    for c in ALL_CONDITIONS:
        if c in ev_dict:
            rows.append(['Counts', c, len(epochs[c]), 'Valid epochs'])
            peak = np.max(
                ev_dict[c].copy().crop(*REWP_WINDOW)
                    .get_data(picks=PRIMARY_CHANNEL)
            ) * 1e6
            rows.append(['Statistics', f"{c}_peak_amp",
                         f"{peak:.2f}", f"µV at {PRIMARY_CHANNEL}"])

    for name, diff in diffs.items():
        peak = np.max(
            diff.copy().crop(*REWP_WINDOW).get_data(picks=PRIMARY_CHANNEL)
        ) * 1e6
        rows.append(['Statistics', f"{name}_diff_peak_amp",
                     f"{peak:.2f}", 'µV (Reward Positivity)'])

    for stat_name, stat_val in tfr_stats:
        rows.append(['Oscillations', stat_name, stat_val,
                     'Peak log-ratio power (4–8 Hz, 200–500 ms)'])

    pd.DataFrame(rows, columns=['Step', 'Param', 'Value', 'Notes']).to_csv(
        out / 'methodology.csv', index=False
    )

# ============================================================================
# EPOCH DROP LOG  [new in v3]
# ============================================================================

def plot_epoch_drop_log(epochs, sub_id, out, events, event_id):
    """
    Bar chart of retained vs rejected epochs, broken down per condition.

    Rationale: Knowing how many trials survive rejection per condition is
    an important sanity check. Heavily imbalanced trial counts between win
    and loss conditions within the same task context could bias amplitude
    estimates or make the ML decoding task trivial. This plot makes that
    immediately visible without needing to inspect the methodology CSV.

    Two panels are produced:
      Left  – absolute counts (retained / rejected) per condition.
      Right – rejection percentage per condition, with a dashed reference
              line at the mean rejection rate across all conditions.

    Parameters
    ----------
    epochs : mne.Epochs  (after rejection has been applied)
    sub_id : str
    out    : Path
    events : np.ndarray  (the original events array passed to Epochs)
    event_id: dict       (event ID mapping)
    """
    if not epochs.drop_log:
        return

    conditions = [c for c in ALL_CONDITIONS if c in event_id]
    if not conditions:
        return

    retained  = []
    rejected  = []
    for c in conditions:
        code = event_id[c]
        # Indices in the original event array that belong to this condition
        cond_indices = np.where(events[:, 2] == code)[0]
        # Which of these were kept? (MNE stores kept indices in .selection)
        n_kept = int(np.sum(np.isin(cond_indices, epochs.selection)))
        n_total = len(cond_indices)
        n_rej = n_total - n_kept
        
        retained.append(n_kept)
        rejected.append(n_rej)

    rej_pct = [
        (r / (k + r) * 100) if (k + r) > 0 else 0.0
        for k, r in zip(retained, rejected)
    ]
    mean_rej = float(np.mean(rej_pct))

    x     = np.arange(len(conditions))
    width = 0.4
    short_labels = [c.replace('_', '\n') for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: stacked bar (retained + rejected)
    ax1.bar(x, retained, width, label='Retained', color='#4878CF', alpha=0.85)
    ax1.bar(x, rejected, width, bottom=retained,
            label='Rejected', color='#D65F5F', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=8)
    ax1.set_ylabel('Trial count')
    ax1.set_title(f"sub-{sub_id}: Epoch counts per condition")
    ax1.legend()

    # Right panel: rejection percentage
    bars = ax2.bar(x, rej_pct, width, color='#D65F5F', alpha=0.75)
    ax2.axhline(mean_rej, color='black', linestyle='--', linewidth=1.2,
                label=f"Mean: {mean_rej:.1f}%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=8)
    ax2.set_ylabel('Rejection rate (%)')
    ax2.set_title(f"sub-{sub_id}: Epoch rejection rate per condition")
    ax2.set_ylim(0, max(max(rej_pct) * 1.2, 10))
    ax2.legend()

    # Annotate bars with the absolute percentage
    for bar, pct in zip(bars, rej_pct):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha='center', va='bottom', fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(out / 'epoch_drop_log.png', dpi=150)
    plt.close(fig)

# ============================================================================
# ML SINGLE-TRIAL DECODING  [from v2]
# ============================================================================

def run_single_trial_ml_decoding(epochs, cond_win, cond_loss):
    """
    Within-subject single-trial Win vs Loss decoding at PRIMARY_CHANNEL.

    Approach
    --------
    Features : mean amplitude in REWP_WINDOW at FCz (one scalar per trial)
    Classifier: Logistic Regression with standard scaling
    Validation: 5-fold stratified cross-validation (AUC)

    Rationale: A simple, interpretable test of whether the RewP window
    amplitude is sufficient on its own to distinguish win from loss
    outcomes single-trial. This parallels the multivariate decoding
    literature while remaining tied to the ERP of interest.

    Returns AUC (float) or np.nan if data are insufficient.
    """
    if (cond_win  not in epochs.event_id or
            cond_loss not in epochs.event_id):
        return np.nan

    ep_win  = epochs[cond_win]
    ep_loss = epochs[cond_loss]
    if len(ep_win) < 5 or len(ep_loss) < 5:
        return np.nan

    time_mask = (
        (epochs.times >= REWP_WINDOW[0]) &
        (epochs.times <= REWP_WINDOW[1])
    )
    X_win  = ep_win.get_data(picks=PRIMARY_CHANNEL)[:, 0, time_mask].mean(axis=1)
    X_loss = ep_loss.get_data(picks=PRIMARY_CHANNEL)[:, 0, time_mask].mean(axis=1)

    X = np.concatenate([X_win, X_loss]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(X_win)), np.zeros(len(X_loss))])

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced'),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')))

# ============================================================================
# GRAND-AVERAGE VISUALISATIONS  [from v2 + v1 TFR extension]
# ============================================================================

def plot_rewp_waveforms(ga_dict, cohort_name, ga_out):
    """
    Classic RewP figure: win vs loss ERP waveforms at FCz for each
    task context, with the RewP window shaded.

    This is the core result figure for any RewP study. Each panel shows
    the grand-average win and loss waveforms for one task context,
    making the win-loss separation (the RewP) directly visible as the
    gap between the two lines around 240-320 ms.
    """
    contexts = [
        ('Low (50%)',      'low_low_win',   'low_low_loss'),
        ('Mid-Low (50%)',  'mid_low_win',   'mid_low_loss'),
        ('Mid-High (80%)', 'mid_high_win',  'mid_high_loss'),
        ('High (80%)',     'high_high_win', 'high_high_loss'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Grand Average RewP: Win vs Loss at {PRIMARY_CHANNEL} ({cohort_name})",
        fontsize=14, fontweight='bold',
    )

    for ax, (label, win_c, loss_c) in zip(axes.flatten(), contexts):
        if win_c not in ga_dict or loss_c not in ga_dict:
            ax.text(0.5, 0.5, f'{label}\nNo data', ha='center',
                    va='center', transform=ax.transAxes)
            continue

        times    = ga_dict[win_c].times
        ch_idx   = ga_dict[win_c].ch_names.index(PRIMARY_CHANNEL)
        win_amp  = ga_dict[win_c].data[ch_idx, :]  * 1e6
        loss_amp = ga_dict[loss_c].data[ch_idx, :] * 1e6
        diff_amp = win_amp - loss_amp

        # Win and loss waveforms
        ax.plot(times, win_amp,  color='#2196F3', linewidth=2,
                label='Win')
        ax.plot(times, loss_amp, color='#F44336', linewidth=2,
                label='Loss')

        # Difference wave
        ax.plot(times, diff_amp, color='#4CAF50', linewidth=1.5,
                linestyle='--', label='Win − Loss (RewP)')

        # RewP window shading
        ax.axvspan(*REWP_WINDOW, alpha=0.15, color='gold',
                   label='RewP window (240–320 ms)')

        # Reference lines
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--',
                   alpha=0.5, label='Feedback onset')
        ax.axhline(0, color='black', linewidth=0.6, alpha=0.3)

        # Peak RewP amplitude annotation
        t_mask   = (times >= REWP_WINDOW[0]) & (times <= REWP_WINDOW[1])
        peak_val = float(np.max(diff_amp[t_mask]))
        peak_t   = float(times[t_mask][np.argmax(diff_amp[t_mask])])
        ax.annotate(
            f'Peak: {peak_val:.2f} µV\n@ {peak_t*1000:.0f} ms',
            xy=(peak_t, peak_val),
            xytext=(peak_t + 0.05, peak_val + 0.5),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=8,
        )

        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_xlim(-0.2, 0.6)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(ga_out / 'ga_rewp_waveforms.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

def generate_grand_average_visuals(cohort_name, avgs, diffs, ga_tfr_dict, ga_out):
    """
    Grand-average ERP and TFR figures for a given cohort.

    Figures
    -------
    1. ga_butterfly.png           – all-channel GA butterfly + GFP
    2. ga_erp_modulation.png      – win/loss ERPs collapsed by task context
    3. ga_topography.png          – RewP difference topography (high-high)
    4. ga_rewp_modulation.png     – RewP amplitude as a function of task
                                    context (modulation curve); replicates
                                    the key result figure from the paper
    5. ga_tfr_<cond>.png          – per-condition GA TFR heatmaps  [v1]

    Parameters
    ----------
    avgs       : dict  {condition: list[Evoked]}
    diffs      : dict  {diff_name: list[Evoked]}
    ga_tfr_dict: dict  {condition: AverageTFR}  (may be empty)
    """
    print(f"  Generating GA figures ({cohort_name})...")
    try:
        # Butterfly
        all_evokeds = [ev for evs in avgs.values() for ev in evs if evs]
        if all_evokeds:
            full_ga = mne.grand_average(all_evokeds)
            fig = full_ga.plot(
                show=False,
                titles=f"Butterfly GA ({cohort_name})",
            )
            fig.savefig(ga_out / 'ga_butterfly.png', dpi=150)
            plt.close(fig)

        # ERP modulation by task context
        fig, ax = plt.subplots(figsize=(10, 6))
        task_map = {
            'Low':      ['low_low_win',  'low_low_loss'],
            'Mid-Low':  ['mid_low_win',  'mid_low_loss'],
            'Mid-High': ['mid_high_win', 'mid_high_loss'],
            'High':     ['high_high_win','high_high_loss'],
        }
        colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
        for (label, conds), color in zip(task_map.items(), colors):
            group = [ev for c in conds if c in avgs for ev in avgs[c]]
            if group:
                data = np.array([
                    ev.get_data(picks=PRIMARY_CHANNEL).flatten()
                    for ev in group
                ])
                ax.plot(
                    group[0].times,
                    np.mean(data, axis=0) * 1e6,
                    label=label, color=color,
                )
        ax.axvspan(*REWP_WINDOW, alpha=0.08, color='red')
        ax.set_title(f"ERPs by Task Context ({cohort_name})")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend()
        fig.savefig(ga_out / 'ga_erp_modulation.png', dpi=150)
        plt.close(fig)

        # RewP topography
        if 'High-High' in diffs and diffs['High-High']:
            diff_ga = mne.grand_average(diffs['High-High'])
            mask = np.zeros(diff_ga.data.shape, dtype=bool)
            if PRIMARY_CHANNEL in diff_ga.ch_names:
                mask[diff_ga.ch_names.index(PRIMARY_CHANNEL), :] = True
            fig = diff_ga.plot_topomap(
                times=np.mean(REWP_WINDOW), average=0.08,
                show=False, mask=mask,
                mask_params=dict(marker='o', markerfacecolor='white',
                                 markersize=10),
                colorbar=True, size=3,
            )
            fig.suptitle(f"GA RewP Topography ({cohort_name})")
            fig.savefig(ga_out / 'ga_topography.png', dpi=150)
            plt.close(fig)

        # ── RewP Amplitude Modulation Curve ──────────────────────────────
        # This figure directly replicates the key result from the paper:
        # RewP amplitude (win − loss difference at FCz, 240–320 ms) plotted
        # as a function of task context, ordered by increasing reward
        # probability (Low → Mid-Low → Mid-High → High).
        #
        # Rationale: The original paper's central claim is that the RewP
        # scales with the predictability / value of the rewarding context.
        # Plotting the difference amplitude across contexts on a single axis
        # makes this gradient immediately visible. Separate win and loss
        # lines are also shown to reveal which component of the difference
        # is driving any modulation.
        context_order = ['low_low', 'mid_low', 'mid_high', 'high_high']
        context_labels = ['Low\n(50%)', 'Mid-Low\n(50%)',
                          'Mid-High\n(80%)', 'High\n(80%)']

        win_amps, loss_amps, diff_amps = [], [], []
        for ctx in context_order:
            win_c, loss_c = f"{ctx}_win", f"{ctx}_loss"
            ch_idx_ga = None

            if win_c in ga_dict and loss_c in ga_dict:
                ev_win  = ga_dict[win_c]
                ev_loss = ga_dict[loss_c]
                if PRIMARY_CHANNEL in ev_win.ch_names:
                    ch_idx_ga = ev_win.ch_names.index(PRIMARY_CHANNEL)
                    t_mask = (
                        (ev_win.times >= REWP_WINDOW[0]) &
                        (ev_win.times <= REWP_WINDOW[1])
                    )
                    w_amp = float(np.mean(ev_win.data[ch_idx_ga, t_mask]))  * 1e6
                    l_amp = float(np.mean(ev_loss.data[ch_idx_ga, t_mask])) * 1e6
                    win_amps.append(w_amp)
                    loss_amps.append(l_amp)
                    diff_amps.append(w_amp - l_amp)
                else:
                    win_amps.append(np.nan)
                    loss_amps.append(np.nan)
                    diff_amps.append(np.nan)
            else:
                win_amps.append(np.nan)
                loss_amps.append(np.nan)
                diff_amps.append(np.nan)

        if not all(np.isnan(diff_amps)):
            x_pos = np.arange(len(context_order))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_pos, win_amps,  'o-', color='#4878CF',
                    linewidth=2, markersize=8, label='Win')
            ax.plot(x_pos, loss_amps, 's-', color='#D65F5F',
                    linewidth=2, markersize=8, label='Loss')
            ax.plot(x_pos, diff_amps, '^--', color='#2ca02c',
                    linewidth=2, markersize=8, label='Win − Loss (RewP)')
            ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(context_labels, fontsize=11)
            ax.set_xlabel('Task Context (reward probability)')
            ax.set_ylabel(f"Mean amplitude at {PRIMARY_CHANNEL} (µV)\n"
                          f"[{int(REWP_WINDOW[0]*1000)}–"
                          f"{int(REWP_WINDOW[1]*1000)} ms]")
            ax.set_title(
                f"RewP Amplitude Modulation by Task Context ({cohort_name})"
            )
            ax.legend()
            plt.tight_layout()
            fig.savefig(ga_out / 'ga_rewp_modulation.png', dpi=150)
            plt.close(fig)

    except Exception as e:
        print(f"    Warning: GA ERP stage failed for {cohort_name}: {e}")

    # GA TFR heatmaps [v1]
    n_conds = len(ga_tfr_dict)
    if n_conds > 0:
        cols = min(4, n_conds)
        rows = (n_conds + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        
        fig.suptitle(f"Grand Average TFR Heatmaps ({cohort_name})", fontsize=16, fontweight='bold', y=1.02)
        
        for idx, (cond, tfr) in enumerate(ga_tfr_dict.items()):
            ax = axes[idx]
            clean_title = cond.replace('_', ' ').title()
            try:
                tfr.plot(
                    picks=PRIMARY_CHANNEL,
                    baseline=None, mode=None,
                    vlim=(-1, 1),
                    title="",
                    axes=ax, show=False, colorbar=True,
                )
                ax.set_title(clean_title, fontsize=12, fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, f"Failed: {e}", ha='center', va='center')
                ax.set_title(clean_title, fontsize=12, fontweight='bold')
                
        for idx in range(n_conds, len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        fig.savefig(ga_out / f"ga_tfr_all_conditions.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

# ============================================================================
# ROBUST STATISTICS & ML REPORT  [from v2]
# ============================================================================

def run_robust_statistics_and_ml(stats_data, cohort_name, out_dir):
    """
    Compute and visualise amplitude statistics and ML decoding results.

    Statistical tests
    -----------------
    Paired t-test   : parametric; assumes normality.
    Wilcoxon signed-rank: non-parametric fallback; robust to outliers.
    Cohen's d       : effect size (mean difference / pooled SD).

    Two categories are tested:
      • Low / Mid-Low contexts (non-learnable, 50 % reward probability)
      • Mid-High / High contexts (learnable, 80 % reward probability)

    ML decoding
    -----------
    One-sample t-test of per-subject AUC against chance (0.50),
    one-tailed (alternative='greater').

    Outputs
    -------
    robust_stats_and_ml_report.txt  – full results text
    stats_boxplot_win_loss.png      – boxplots with individual data
    ml_decoding_bar_plot.png        – bar chart of mean AUC ± SEM
    """
    df = pd.DataFrame(stats_data)
    if 'Performers' in cohort_name:
        df = df[df['Cohort'] == True]

    n_subs = len(df)
    if n_subs < 2:
        print(f"  Skipping stats for {cohort_name}: only {n_subs} subject(s).")
        return

    report     = [f"--- Statistics & ML Decoding: {cohort_name} (n={n_subs}) ---"]
    plot_stats = []
    print(f"\n{report[0]}")

    def test_category(win_cols, loss_cols, category_name):
        wp = [c for c in win_cols  if c in df.columns]
        lp = [c for c in loss_cols if c in df.columns]
        if not wp or not lp:
            return

        win_vals  = df[wp].mean(axis=1)
        loss_vals = df[lp].mean(axis=1)
        diff_vals = win_vals - loss_vals

        t_stat, t_pval = stats.ttest_rel(win_vals, loss_vals)
        mean_diff = diff_vals.mean()
        d = (mean_diff / diff_vals.std(ddof=1)
             if diff_vals.std(ddof=1) != 0 else 0.0)
        w_stat, w_pval = stats.wilcoxon(win_vals, loss_vals)

        res = (
            f"[{category_name}] Win vs Loss (240–320 ms, {PRIMARY_CHANNEL}):\n"
            f"  Mean diff: {mean_diff:.2f} µV\n"
            f"  Parametric: t({n_subs-1}) = {t_stat:.3f}, "
            f"p = {t_pval:.4f}, Cohen's d = {d:.3f}\n"
            f"  Non-parametric: Wilcoxon W = {w_stat:.3f}, p = {w_pval:.4f}"
        )
        report.append(res)
        print(res)
        plot_stats.append({
            'Category': category_name,
            'Win':   win_vals.values,
            'Loss':  loss_vals.values,
            'P_Val': t_pval,
        })

    report.append('\n[Amplitude Tests (Win vs Loss)]')
    test_category(
        ['low_low_win', 'mid_low_win'],
        ['low_low_loss', 'mid_low_loss'],
        'Low/Mid-Low (50% reward)',
    )
    test_category(
        ['mid_high_win', 'high_high_win'],
        ['mid_high_loss', 'high_high_loss'],
        'Mid-High/High (80% reward)',
    )

    # ML decoding results
    report.append('\n[Single-Trial ML Decoding (ROC-AUC)]')
    ml_plot_data = []
    for ml_col, name in [
        ('ML_AUC_Low',  'Low-Low Context'),
        ('ML_AUC_High', 'High-High Context'),
    ]:
        if ml_col in df.columns:
            aucs = df[ml_col].dropna()
            if len(aucs) > 2:
                mean_auc = float(aucs.mean())
                sem_auc  = float(stats.sem(aucs))
                t_stat, p_val = stats.ttest_1samp(
                    aucs, 0.5, alternative='greater'
                )
                res = (f"  {name}: Mean AUC = {mean_auc:.3f} ± {sem_auc:.3f} "
                       f"(p = {p_val:.4f} vs chance = 0.50)")
                report.append(res)
                print(res)
                ml_plot_data.append((name, mean_auc, sem_auc, p_val))

    with open(out_dir / 'robust_stats_and_ml_report.txt', 'w') as f:
        f.write('\n'.join(report))

    # -- Boxplots --
    if plot_stats:
        fig, axes = plt.subplots(
            1, len(plot_stats),
            figsize=(5 * len(plot_stats), 6),
        )
        if len(plot_stats) == 1:
            axes = [axes]
        for ax, p in zip(axes, plot_stats):
            ax.boxplot(
                [p['Win'], p['Loss']], widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
            )
            for i in range(len(p['Win'])):
                ax.plot([1, 2], [p['Win'][i], p['Loss'][i]], 'k-', alpha=0.3, zorder=2)
                ax.plot(1, p['Win'][i], 'ko', alpha=0.5, markersize=5, zorder=3)
                ax.plot(2, p['Loss'][i], 'ko', alpha=0.5, markersize=5, zorder=3)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Win', 'Loss'])
            ax.set_title(f"{p['Category']} (p = {p['P_Val']:.3f})")
            ax.set_ylabel(f"Mean amplitude (µV) at {PRIMARY_CHANNEL}")
        plt.tight_layout()
        fig.savefig(out_dir / 'stats_boxplot_win_loss.png', dpi=150)
        plt.close(fig)

    # -- ML bar plot --
    if ml_plot_data:
        fig, ax = plt.subplots(figsize=(6, 6))
        names = [d[0] for d in ml_plot_data]
        means = [d[1] for d in ml_plot_data]
        errs  = [d[2] for d in ml_plot_data]
        pvals = [d[3] for d in ml_plot_data]
        bars  = ax.bar(names, means, yerr=errs, capsize=5,
                       color=['#ff9999', '#66b3ff'], alpha=0.85)
        ax.axhline(0.5, color='r', linestyle='--', label='Chance (0.50)')
        ax.set_ylim(0.3, 1.0)
        ax.set_ylabel('ROC-AUC Score')
        ax.set_title(
            f"Single-Trial Decoding at {PRIMARY_CHANNEL}\n{cohort_name}"
        )
        for bar, p in zip(bars, pvals):
            sig = ('***' if p < 0.001 else '**' if p < 0.01
                   else '*' if p < 0.05 else 'ns')
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                sig, ha='center', va='bottom', fontweight='bold',
            )
        ax.legend(loc='lower left')
        plt.tight_layout()
        fig.savefig(out_dir / 'ml_decoding_bar_plot.png', dpi=150)
        plt.close(fig)

# ============================================================================
# GRAND-AVERAGE REPORT CSV
# ============================================================================

def save_ga_statistics_csv(ga_dict, ga_diffs, ga_tfr_dict, ga_out):
    """
    Write a summary CSV of grand-average peak amplitudes and theta power.

    Mirrors the per-subject methodology.csv but at the group level.
    """
    rows = []
    for c, ev in ga_dict.items():
        peak = np.max(
            ev.copy().crop(*REWP_WINDOW).get_data(picks=PRIMARY_CHANNEL)
        ) * 1e6
        rows.append(['Grand Average', f"{c}_peak_amp",
                     f"{peak:.2f}", 'µV'])

    for c, tfr in ga_tfr_dict.items():
        crop = tfr.copy().crop(
            tmin=TFA_WINDOW[0], tmax=TFA_WINDOW[1],
            fmin=THETA_BAND[0], fmax=THETA_BAND[1],
        )
        rows.append(['Grand Average', f"{c}_theta_peak_power",
                     f"{float(np.max(crop.data)):.4e}", 'Power (log-ratio)'])

    for name, diff in ga_diffs.items():
        peak = np.max(
            diff.copy().crop(*REWP_WINDOW).get_data(picks=PRIMARY_CHANNEL)
        ) * 1e6
        rows.append(['Grand Average', f"{name}_diff_peak_amp",
                     f"{peak:.2f}", 'µV (Reward Positivity)'])

    pd.DataFrame(rows, columns=['Type', 'Param', 'Value', 'Notes']).to_csv(
        ga_out / 'ga_statistics.csv', index=False
    )

def plot_preprocessing_dashboard(sub_id, viz_data, epochs, out_dir):
    """Generate a comprehensive visual dashboard of the preprocessing steps."""
    import matplotlib.gridspec as gridspec
    
    if PRIMARY_CHANNEL in epochs.ch_names and len(epochs) > 0:
        epoch_data = epochs.get_data(picks=[PRIMARY_CHANNEL])[0, 0, :] * 1e6
        epoch_times = epochs.times
    else:
        epoch_data, epoch_times = None, None

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 0.4], hspace=0.6, wspace=0.3)
    
    fig.suptitle(f"Subject {sub_id} - Complete Preprocessing Pipeline Visualization", 
                 fontsize=18, fontweight='bold', y=0.96)
                 
    ax_raw = fig.add_subplot(gs[0, :])
    if 'raw' in viz_data:
        ax_raw.plot(viz_data['times'], viz_data['raw'], color='#4a4a4a', linewidth=0.8)
    ax_raw.set_title("Step 1: Raw Data (Before Processing)", fontweight='bold')
    ax_raw.set_ylabel("Amplitude (µV)")
    ax_raw.set_xlim(0, 10)
    ax_raw.grid(True, alpha=0.3)
    
    ax_filt = fig.add_subplot(gs[1, :])
    if 'filt' in viz_data:
        ax_filt.plot(viz_data['times'], viz_data['filt'], color='#5c6bc0', linewidth=0.8)
    ax_filt.set_title("Step 2: After Filtering (0.1-100 Hz + 50 Hz Notch)", fontweight='bold')
    ax_filt.set_ylabel("Amplitude (µV)")
    ax_filt.set_xlim(0, 10)
    ax_filt.grid(True, alpha=0.3)
    
    ax_ica = fig.add_subplot(gs[2, :])
    if 'clean' in viz_data:
        ax_ica.plot(viz_data['times'], viz_data['clean'], color='#66bb6a', linewidth=0.8)
    ax_ica.set_title("Step 3: After ICA (Artifact Removal)", fontweight='bold')
    ax_ica.set_ylabel("Amplitude (µV)")
    ax_ica.set_xlabel("Time (s)")
    ax_ica.set_xlim(0, 10)
    ax_ica.grid(True, alpha=0.3)
    
    ax_psd_raw = fig.add_subplot(gs[3, 0])
    if 'psd_raw' in viz_data:
        ax_psd_raw.plot(viz_data['freqs'], viz_data['psd_raw'], color='#555555', linewidth=1)
    ax_psd_raw.set_title("PSD: Raw Data", fontweight='bold', fontsize=11)
    ax_psd_raw.set_ylabel("Power (dB/Hz re 1 µV²)")
    ax_psd_raw.set_xlabel("Frequency (Hz)")
    ax_psd_raw.set_xlim(0, 70)
    ax_psd_raw.grid(True, alpha=0.3)
    
    ax_psd_ica = fig.add_subplot(gs[3, 1])
    if 'psd_clean' in viz_data:
        ax_psd_ica.plot(viz_data['freqs'], viz_data['psd_clean'], color='#555555', linewidth=1)
    ax_psd_ica.set_title("PSD: After ICA", fontweight='bold', fontsize=11)
    ax_psd_ica.set_ylabel("Power (dB/Hz re 1 µV²)")
    ax_psd_ica.set_xlabel("Frequency (Hz)")
    ax_psd_ica.set_xlim(0, 70)
    ax_psd_ica.grid(True, alpha=0.3)
    
    ax_epoch = fig.add_subplot(gs[3, 2])
    if epoch_data is not None:
        ax_epoch.plot(epoch_times, epoch_data, color='#d32f2f', linewidth=1.5)
        ax_epoch.axvspan(*REWP_WINDOW, color='lightgray', alpha=0.5, label='RewP window')
        ax_epoch.axvline(0, color='gray', linestyle='--', label='Feedback onset')
        ax_epoch.legend(fontsize=8, loc='upper right')
    ax_epoch.set_title("Example Epoch", fontweight='bold', fontsize=11)
    ax_epoch.set_ylabel("Amplitude (µV)")
    ax_epoch.set_xlabel("Time (s)")
    ax_epoch.grid(True, alpha=0.3)
    
    # -----------------------------------------------------
    # Draw Flowchart Footer
    # -----------------------------------------------------
    ax_flow = fig.add_subplot(gs[4, :])
    ax_flow.axis('off')
    ax_flow.set_title("NeuroSimha Preprocessing Pipeline", fontsize=15, fontweight='bold', pad=15)
    
    stages = [
        "Raw EEG\n(BIDS)", "Set Montage\n(easycap)", "Resample\n(250 Hz)",
        "Filter\n(0.1-100 Hz)", "Notch\n(50 Hz)", "ICA\n(20 comp)",
        "Final Filter\n(0.1-30 Hz)", "Epoch\n(-200~600ms)",
        "Baseline\n(-200~0ms)", "Reject\n(120 µV)", "Average\n(ERPs)"
    ]
    
    n_stages = len(stages)
    x_centers = [(i + 0.5) / n_stages for i in range(n_stages)]
    
    # Continuous thick black line behind the boxes
    ax_flow.plot([x_centers[0], x_centers[-1]], [0.3, 0.3], color='black', linewidth=5, zorder=1)
    
    for i, text in enumerate(stages):
        ax_flow.text(x_centers[i], 0.3, text, ha='center', va='center', fontsize=8.5, fontweight='bold',
                     bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='square,pad=0.4', linewidth=2),
                     zorder=2)
    # -----------------------------------------------------
    
    fig.savefig(out_dir / 'preprocessing_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    # Load included subject list (check root first, then script dir)
    script_dir = Path(__file__).parent.absolute()
    good_subjects = []
    subjects_file = Path('good_subjects.json')
    if not subjects_file.exists():
        subjects_file = script_dir.parent / 'good_subjects.json'

    if subjects_file.exists():
        with open(subjects_file) as f:
            good_subjects = json.load(f)
        print(f"Loaded good_subjects.json: {good_subjects}")
    else:
        print("good_subjects.json not found – using full SUBJECT_IDS list.")

    # Storage containers
    storage = {
        'all_list':       {c: [] for c in ALL_CONDITIONS},
        'perf_list':      {c: [] for c in ALL_CONDITIONS},
        'all_diffs':      {'Low-Low': [], 'High-High': []},
        'perf_diffs':     {'Low-Low': [], 'High-High': []},
        'all_tfrs':       {c: [] for c in ALL_CONDITIONS},   # [v1]
        'perf_tfrs':      {c: [] for c in ALL_CONDITIONS},   # [v1]
        'subject_records': [],
        'stats_data':     [],
    }

    print(f"\n{'='*60}")
    print("COMPLETE EEG ANALYSIS PIPELINE – v3")
    print(f"{'='*60}")

    for i, sub_id in enumerate(SUBJECT_IDS):
        print(f"\n{'─'*60}")
        print(f"PROCESSING ({i+1}/{len(SUBJECT_IDS)}): sub-{sub_id}")
        print(f"{'─'*60}")
        try:
            out = Path(OUTPUT_DIR) / f"sub-{sub_id}"
            out.mkdir(parents=True, exist_ok=True)

            raw_cl, labels, excluded, viz_data = preprocess_subject(sub_id, out)
            events, ev_id, perf      = parse_events_by_condition(sub_id, raw_cl)

            epochs = mne.Epochs(
                raw_cl, events, ev_id,
                tmin=-0.2, tmax=0.6,
                baseline=BASELINE,
                reject=dict(eeg=REJECT_THRESHOLD),
                preload=True, verbose=False,
            )

            # Epoch drop log (sanity check: trial counts & rejection rate)
            plot_epoch_drop_log(epochs, sub_id, out, events, ev_id)
            
            # Diagnostic print
            total_attempted = len(epochs.selection) + len([x for x in epochs.drop_log if x])
            n_rejected = total_attempted - len(epochs)
            print(f"  sub-{sub_id}: {len(epochs)}/{total_attempted} epochs kept "
                  f"({n_rejected} rejected)")

            meets_threshold = (
                (sub_id in good_subjects) if good_subjects
                else (perf['mh_win_rate'] >= 60.0)
            )

            ev_dict = {
                c: epochs[c].average()
                for c in epochs.event_id if len(epochs[c]) > 0
            }

            # Individual ERP plots
            generate_individual_plots(epochs, ev_dict, sub_id, out)

            # Preprocessing dashboard [NEW]
            plot_preprocessing_dashboard(sub_id, viz_data, epochs, out)

            # Time-frequency analysis  [v1]
            print(f"  [3/3] Time-frequency analysis...")
            tfr_objects, tfr_stats_rows = run_time_frequency_analysis(
                epochs, sub_id, out
            )

            # Methodology CSV
            diffs_sub = {}
            for name, win_c, loss_c in [
                ('Low-Low',   'low_low_win',   'low_low_loss'),
                ('High-High', 'high_high_win', 'high_high_loss'),
            ]:
                if win_c in ev_dict and loss_c in ev_dict:
                    diffs_sub[name] = mne.combine_evoked(
                        [ev_dict[win_c], ev_dict[loss_c]], weights=[1, -1]
                    )
            generate_methodology_csv(
                sub_id, labels, excluded, epochs, ev_dict,
                diffs_sub, tfr_stats_rows, out,
            )

            # ML decoding  [v2]
            sub_stats = {
                'ID':         sub_id,
                'Cohort':     meets_threshold,
                'ML_AUC_Low': run_single_trial_ml_decoding(
                    epochs, 'low_low_win', 'low_low_loss'),
                'ML_AUC_High': run_single_trial_ml_decoding(
                    epochs, 'high_high_win', 'high_high_loss'),
            }

            # Store evokeds & TFRs
            for c, ev in ev_dict.items():
                storage['all_list'][c].append(ev)
                if meets_threshold:
                    storage['perf_list'][c].append(ev)

                ch_idx = ev.ch_names.index(PRIMARY_CHANNEL)
                mask   = (
                    (ev.times >= REWP_WINDOW[0]) &
                    (ev.times <= REWP_WINDOW[1])
                )
                sub_stats[c] = np.mean(ev.data[ch_idx, mask]) * 1e6

            for c, tfr_obj in tfr_objects.items():
                storage['all_tfrs'][c].append(tfr_obj)
                if meets_threshold:
                    storage['perf_tfrs'][c].append(tfr_obj)

            for name, win_c, loss_c in [
                ('Low-Low',   'low_low_win',   'low_low_loss'),
                ('High-High', 'high_high_win', 'high_high_loss'),
            ]:
                if win_c in ev_dict and loss_c in ev_dict:
                    diff = mne.combine_evoked(
                        [ev_dict[win_c], ev_dict[loss_c]], weights=[1, -1]
                    )
                    storage['all_diffs'][name].append(diff)
                    if meets_threshold:
                        storage['perf_diffs'][name].append(diff)

            storage['subject_records'].append([
                sub_id,
                f"{perf['choice_accuracy']:.1f}%",
                f"{perf['mh_win_rate']:.1f}%",
                meets_threshold,
            ])
            storage['stats_data'].append(sub_stats)

            del raw_cl
            gc.collect()

        except Exception as e:
            print(f"  ERROR processing sub-{sub_id}: {e}")

    # ------------------------------------------------------------------
    # Grand Averages
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("GRAND AVERAGES")
    print(f"{'='*60}")

    ga_root = Path(OUTPUT_DIR) / 'grand_average'
    ga_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        storage['subject_records'],
        columns=['ID', 'Accuracy', 'Win_Rate', 'Cohort'],
    ).to_csv(ga_root / 'cohort_table.csv', index=False)

    cohorts = [
        ('Full_Cohort_n12', storage['all_list'],  storage['all_diffs'],
         storage['all_tfrs']),
        ('Performers_n8',   storage['perf_list'], storage['perf_diffs'],
         storage['perf_tfrs']),
    ]

    for cohort_name, avgs, diffs, tfrs in cohorts:
        if not any(v for v in avgs.values()):
            continue

        ga_out = ga_root / cohort_name
        ga_out.mkdir(parents=True, exist_ok=True)

        # Compute GA evokeds
        ga_dict = {
            c: mne.grand_average(ev_list)
            for c, ev_list in avgs.items() if ev_list
        }
        ga_diffs = {
            name: mne.combine_evoked(
                [ga_dict[win_c], ga_dict[loss_c]], weights=[1, -1]
            )
            for name, win_c, loss_c in [
                ('Low-Low',   'low_low_win',   'low_low_loss'),
                ('High-High', 'high_high_win', 'high_high_loss'),
            ]
            if win_c in ga_dict and loss_c in ga_dict
        }
        # Compute GA TFRs  [v1]
        # ga_tfr_dict = {
        #     c: mne.grand_average(tfr_list)
        #     for c, tfr_list in tfrs.items() if tfr_list
        # }
        ga_tfr_dict = {}

        for c, tfr_list in tfrs.items():
            if tfr_list:
                ga_tfr = mne.grand_average(tfr_list)
                ga_tfr_dict[c] = ga_tfr

                

        generate_grand_average_visuals(
            cohort_name, avgs, diffs, ga_tfr_dict, ga_out
        )
        
        plot_rewp_waveforms(ga_dict, cohort_name, ga_out)

        save_ga_statistics_csv(ga_dict, ga_diffs, ga_tfr_dict, ga_out)

        try:
            run_robust_statistics_and_ml(
                storage['stats_data'], cohort_name, ga_out
            )
        except Exception as e:
            print(f"  Warning: stats/ML failed for {cohort_name}: {e}")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"Results: {ga_root}")
    print(f"{'='*60}")
