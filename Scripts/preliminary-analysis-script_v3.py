"""
Milestone 3: Preliminary Analysis - Subject Selection
Team NeuroSimha - EEG Casinos Task

Merged v3: Combines v1 (detailed reporting, robust win-rate logic) with v2
(averaged threshold criterion, streamlined output, JSON hand-off).

Key decisions preserved from each version:
  - v1: Full per-task breakdown (low/mid-low/mid-high/high), detailed metadata,
         signal quality CSVs, rich plot styling, recommendation summary block.
  - v2: Averaged threshold criterion (mean of mid-high and high cue win rates >= 60%),
         JSON hand-off for main pipeline, compact run_preliminary_analysis structure.

Threshold rationale (v2, now canonical):
  The 60% win-rate threshold is calculated as the average performance across
  both Mid-Task (High-Cue) and High-Value Task scenarios. This averaged criterion
  correctly identifies the 8-subject subset (27, 28, 31, 34, 35, 36, 37, 38)
  described in the original paper.
"""

import json
import warnings

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
from mne_bids import BIDSPath, read_raw_bids

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

OUTPUT_DIR = project_root / 'results' / 'preliminary_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Subjects 27-38 are the subset available in ds004147
ALL_SUBJECTS = [f'{i:02d}' for i in range(27, 39)]

# Inclusion criterion: mean of mid-high-cue and high-task win rates >= 60 %
WIN_RATE_THRESHOLD = 60.0

# Quick ERP preview: minimal filtering for visual sanity check only
PREVIEW_FILTER_HIGH = 30.0
PREVIEW_REJECT_THRESHOLD = 150e-6  # uV; loose, raw data not cleaned
REWP_WINDOW = (0.24, 0.32)         # s; RewP window for annotations

# ============================================================================
# EVENT & WIN-RATE HELPERS
# ============================================================================

def load_events_tsv(subject_id):
    """
    Load the BIDS events.tsv for a subject.

    The events.tsv is the authoritative source for trial information in this
    BIDS dataset. We use it directly rather than relying on EEG annotations,
    which avoids the "Event time samples were not unique" error caused by
    duplicate trigger codes in the raw file.

    Returns
    -------
    pd.DataFrame or None
    """
    events_file = (Path(BIDS_ROOT) / f'sub-{subject_id}' / 'eeg'
                   / f'sub-{subject_id}_task-casinos_events.tsv')

    if not events_file.exists():
        print(f"   Events file not found: {events_file}")
        return None

    try:
        return pd.read_csv(events_file, sep='\t')
    except Exception as e:
        print(f"  Error loading events for sub-{subject_id}: {e}")
        return None


def load_events_for_mne(subject_id, raw):
    """
    Convert TSV events to MNE events array (sample, 0, event_id).

    All stimulus events are collapsed to a single ID (= 1) for the
    preliminary screening ERP. Condition-level separation happens in
    the main analysis pipeline.

    Duplicate sample indices are removed (warn if any found).

    Returns
    -------
    events : np.ndarray or None
    event_id : dict or None
    """
    events_df = load_events_tsv(subject_id)
    if events_df is None:
        return None, None

    stim_events = events_df[
        events_df['trial_type'].str.contains('Stimulus', na=False)
    ].copy()

    if len(stim_events) == 0:
        return None, None

    sfreq = raw.info['sfreq']
    samples = (stim_events['onset'].values * sfreq).astype(int)

    # Remove duplicates that arise from simultaneous trigger codes
    unique_samples, unique_indices = np.unique(samples, return_index=True)
    if len(unique_samples) < len(samples):
        n_removed = len(samples) - len(unique_samples)
        print(f"  Warning: Removed {n_removed} duplicate event sample(s) "
              f"for sub-{subject_id}")
        samples = unique_samples
        stim_events = stim_events.iloc[unique_indices]

    events = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        np.ones(len(samples), dtype=int),
    ])

    return events, {'stimulus': 1}


def calculate_win_rates_from_tsv(subject_id):
    """
    Calculate per-task win rates from the events.tsv.

    Event code reference (stimulus value column):
        Low-task       : S  6 (win)  / S  7 (loss)  - 50 % probability
        Mid-task low   : S 16 (win)  / S 17 (loss)  - 50 % probability
        Mid-task high  : S 26 (win)  / S 27 (loss)  - 80 % probability (learnable)
        High-task      : S 36 (win)  / S 37 (loss)  - 80 % probability (learnable)

    Threshold criterion (v2, canonical):
        Subject included if AVERAGE(mid_high_win_rate, high_win_rate) >= 60 %.
        Using the average rather than requiring both individually matches the
        8-subject subset reported in the paper (27, 28, 31, 34, 35, 36, 37, 38).

    Returns
    -------
    dict with per-task rates and threshold flag.
    """
    events_df = load_events_tsv(subject_id)
    if events_df is None:
        return {'subject_id': subject_id, 'error': 'Could not load events'}

    stim = events_df[
        events_df['trial_type'].str.contains('Stimulus', na=False)
    ].copy()

    if len(stim) == 0:
        return {'subject_id': subject_id, 'error': 'No stimulus events found'}

    def _count(pattern):
        return len(stim[stim['value'].str.match(pattern, na=False)])

    # Per-task counts
    low_w,  low_l  = _count(r'S\s+6$'), _count(r'S\s+7$')
    ml_w,   ml_l   = _count(r'S\s16$'), _count(r'S\s17$')
    mh_w,   mh_l   = _count(r'S\s26$'), _count(r'S\s27$')
    hi_w,   hi_l   = _count(r'S\s36$'), _count(r'S\s37$')

    def _rate(wins, total):
        return (wins / total * 100) if total > 0 else 0.0

    mh_rate = _rate(mh_w, mh_w + mh_l)
    hi_rate  = _rate(hi_w, hi_w + hi_l)

    # Averaged criterion (see docstring)
    meets_threshold = ((mh_rate + hi_rate) / 2.0) >= WIN_RATE_THRESHOLD

    total_wins   = low_w + ml_w + mh_w + hi_w
    total_losses = low_l + ml_l + mh_l + hi_l
    total_trials = total_wins + total_losses

    return {
        'subject_id':         subject_id,
        'total_trials':       total_trials,
        'total_wins':         total_wins,
        'total_losses':       total_losses,
        'overall_win_rate':   _rate(total_wins, total_trials),
        'low_task_rate':      _rate(low_w, low_w + low_l),
        'mid_low_rate':       _rate(ml_w,  ml_w  + ml_l),
        'mid_high_rate':      mh_rate,
        'high_rate':          hi_rate,
        'avg_high_cue_rate':  (mh_rate + hi_rate) / 2.0,
        'meets_60pct_threshold': meets_threshold,
        'method': 'averaged_high_value_cues',
    }

# ============================================================================
# DATA QUALITY HELPERS
# ============================================================================

def get_subject_metadata(subject_id):
    """
    Load raw BIDS file and return basic recording metadata.

    A lightweight check: we load but do not process the data, so the
    returned values reflect the as-recorded state.
    """
    try:
        bids_path = BIDSPath(
            subject=subject_id, task='casinos',
            datatype='eeg', root=BIDS_ROOT,
        )
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()
        return {
            'subject_id':   subject_id,
            'n_channels':   len(raw.ch_names),
            'n_samples':    raw.n_times,
            'duration_sec': raw.times[-1],
            'sfreq':        raw.info['sfreq'],
            'status':       'OK',
        }
    except Exception as e:
        return {'subject_id': subject_id, 'status': f'ERROR: {e}'}


def quick_signal_quality_check(subject_id):
    """
    Inspect the first 30 s of raw data for obvious quality problems.

    Checks:
      - Flat channels  (std < 0.1 µV) → hardware / connectivity issue
      - Noisy channels (std > 100 µV) → usually EMG or electrode pop

    Note: A non-zero artifact percentage is *expected* in raw, unprocessed
    EEG and is not used as an exclusion criterion here.
    """
    try:
        bids_path = BIDSPath(
            subject=subject_id, task='casinos',
            datatype='eeg', root=BIDS_ROOT,
        )
        raw = read_raw_bids(bids_path, verbose=False)
        raw.load_data()

        # Use only the first 30 s for speed
        n_stop = int(30 * raw.info['sfreq'])
        data = raw.get_data(stop=n_stop)

        stds = np.std(data, axis=1) * 1e6          # convert to µV
        flat_channels       = int(np.sum(stds < 0.1))
        high_noise_channels = int(np.sum(stds > 100))
        artifact_pct        = np.sum(np.abs(data) > 100e-6) / data.size * 100

        quality = 'GOOD' if flat_channels == 0 else 'NEEDS_REVIEW'

        return {
            'subject_id':          subject_id,
            'flat_channels':       flat_channels,
            'high_noise_channels': high_noise_channels,
            'artifact_pct':        artifact_pct,
            'quality':             quality,
            'note':                'High artifact % is NORMAL for raw EEG!',
        }
    except Exception as e:
        return {'subject_id': subject_id, 'error': str(e)}

# ============================================================================
# VISUALISATION
# ============================================================================

def plot_preliminary_comparison(subjects_to_compare, filename_suffix=''):
    """
    Generate a stacked ERP overview for the given subject list.

    Each subplot shows the average FCz waveform after minimal filtering
    (0.1–30 Hz, no ICA). This is a screening visual only; the purpose is
    to catch obviously broken recordings before committing to full
    preprocessing. The RewP window (240–320 ms) is shaded for reference.

    Parameters
    ----------
    subjects_to_compare : list of str
    filename_suffix : str
        Appended to the output filename, e.g. '_good_subjects'.
    """
    if not subjects_to_compare:
        print("  No subjects to plot.")
        return

    n = len(subjects_to_compare)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle(f"Preliminary Subject Comparison (FCz)", fontsize=16, fontweight='bold', y=1.02)

    for idx, ax in enumerate(axes):
        if idx >= n:
            ax.axis('off')
            continue
            
        subject_id = subjects_to_compare[idx]
        try:
            bids_path = BIDSPath(
                subject=subject_id, task='casinos',
                datatype='eeg', root=BIDS_ROOT,
            )
            raw = read_raw_bids(bids_path, verbose=False)
            raw.load_data()
            raw.filter(0.1, PREVIEW_FILTER_HIGH, verbose=False)

            events, event_id = load_events_for_mne(subject_id, raw)

            if events is None or len(events) < 10:
                ax.text(0.5, 0.5, f'sub-{subject_id}: insufficient events',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            epochs = mne.Epochs(
                raw, events, event_id=event_id,
                tmin=-0.2, tmax=0.6,
                baseline=(None, 0),
                reject=dict(eeg=PREVIEW_REJECT_THRESHOLD),
                preload=True, verbose=False, on_missing='warn',
            )

            if len(epochs) < 5:
                ax.text(0.5, 0.5, f'sub-{subject_id}: too few valid epochs ({len(epochs)})',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            evoked = epochs.average()

            if 'FCz' in evoked.ch_names:
                ch_idx = evoked.ch_names.index('FCz')
                amp = evoked.data[ch_idx, :] * 1e6
                ax.plot(evoked.times, amp,
                        linewidth=2, color='steelblue', label=f'sub-{subject_id}')
                ax.axvspan(*REWP_WINDOW, alpha=0.2, color='red', label='RewP window')
                ax.axvline(0, color='k', linestyle='--', alpha=0.4, label='Stimulus onset')
                ax.axhline(0, color='k', linestyle='-', alpha=0.2)
                ax.set_ylabel('Amplitude (µV)', fontsize=10)
                y_range = np.max(np.abs(amp))
                ax.set_ylim([-y_range * 1.2, y_range * 1.2])
            else:
                ax.text(0.5, 0.5, f'sub-{subject_id}: FCz not found',
                        ha='center', va='center', transform=ax.transAxes)

            ax.set_title(f'sub-{subject_id} – FCz (n={len(epochs)} epochs)',
                         fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.2, 0.6])
            ax.set_xlabel('Time (s)', fontsize=10)

        except Exception as e:
            msg = str(e)[:80]
            ax.text(0.5, 0.5, f'sub-{subject_id}: {msg}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)

    plt.tight_layout()
    out_file = OUTPUT_DIR / f'preliminary_subject_comparison{filename_suffix}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_file}")
    plt.close()


def plot_win_rate_summary(win_rate_df):
    """
    Bar chart of per-task win rates for all subjects.

    Visualises the performance breakdown that drives subject selection.
    The dashed red line marks the 60 % threshold. Bars are colour-coded
    by task context (low / mid-low / mid-high / high).
    """
    if win_rate_df.empty:
        return

    valid = win_rate_df.dropna(subset=['mid_high_rate', 'high_rate']).copy()
    if valid.empty:
        return

    x = np.arange(len(valid))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
    rate_cols = ['low_task_rate', 'mid_low_rate', 'mid_high_rate', 'high_rate']
    labels_   = ['Low task', 'Mid-Low cue', 'Mid-High cue', 'High task']

    for i, (col, lbl, col_) in enumerate(zip(rate_cols, labels_, colors)):
        if col in valid.columns:
            ax.bar(x + i * width, valid[col], width, label=lbl, color=col_, alpha=0.85)

    ax.axhline(WIN_RATE_THRESHOLD, color='red', linestyle='--',
               linewidth=1.5, label=f'{WIN_RATE_THRESHOLD:.0f}% threshold')

    # Mark subjects that meet the criterion
    for i, row in enumerate(valid.itertuples()):
        if row.meets_60pct_threshold:
            ax.annotate('*', xy=(i + 1.5 * width, 103), ha='center',
                        fontsize=18, color='green', fontweight='bold')

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"sub-{s}" for s in valid['subject_id']], rotation=45, ha='right')
    ax.set_ylabel('Win rate (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Per-Task Win Rates & Inclusion Threshold')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    out_file = OUTPUT_DIR / 'win_rate_summary.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_file}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def run_preliminary_analysis():
    """
    Main entry point.

    Steps
    -----
    1. Metadata check  – load every subject's raw file; confirm accessibility.
    2. Win-rate screen – compute high-cue win rates; apply 60 % threshold.
    3. Signal quality  – scan first 30 s for flat/noisy channels.
    4. Visualisations  – ERP preview plots + win-rate bar chart.
    5. Hand-off        – write good_subjects.json for the main pipeline.
    """
    print("\n" + "=" * 70)
    print("PRELIMINARY ANALYSIS – SUBJECT SELECTION (v3)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Metadata
    # ------------------------------------------------------------------
    print("\n[1/4] Checking data availability & metadata...")
    metadata_list = []
    for subject_id in ALL_SUBJECTS:
        print(f"  sub-{subject_id}...", end=' ')
        meta = get_subject_metadata(subject_id)
        metadata_list.append(meta)
        print(meta['status'])

    metadata_df = pd.DataFrame(metadata_list)
    available_subjects = (
        metadata_df[metadata_df['status'] == 'OK']['subject_id'].tolist()
    )
    metadata_df.to_csv(OUTPUT_DIR / 'subject_metadata.csv', index=False)
    print(f"\n  Available: {len(available_subjects)}/{len(ALL_SUBJECTS)}")

    if not available_subjects:
        print("\n  No subjects available. Check BIDS_ROOT.")
        return []

    # ------------------------------------------------------------------
    # Step 2: Win rates
    # ------------------------------------------------------------------
    print("\n[2/4] Calculating win rates from events.tsv...")
    print(f"  Criterion: mean(mid-high cue, high task) >= {WIN_RATE_THRESHOLD}%")

    win_rate_list = []
    for subject_id in available_subjects:
        print(f"  sub-{subject_id}...", end=' ')
        wr = calculate_win_rates_from_tsv(subject_id)
        win_rate_list.append(wr)
        if 'error' not in wr:
            flag = "[PASS]" if wr['meets_60pct_threshold'] else "[FAIL]"
            print(f"Mid-High: {wr['mid_high_rate']:.1f}%  "
                  f"High: {wr['high_rate']:.1f}%  "
                  f"Avg: {wr['avg_high_cue_rate']:.1f}%  {flag}")
        else:
            print(f"ERROR: {wr.get('error')}")

    win_rate_df = pd.DataFrame(win_rate_list)
    win_rate_df.to_csv(OUTPUT_DIR / 'win_rates.csv', index=False)

    good_subjects = [
        r['subject_id'] for r in win_rate_list
        if r.get('meets_60pct_threshold') is True
    ]
    print(f"\n  Subjects meeting threshold: {len(good_subjects)} → {good_subjects}")

    # ------------------------------------------------------------------
    # Step 3: Signal quality
    # ------------------------------------------------------------------
    print("\n[3/4] Signal quality scan (first 30 s, raw)...")
    print("  Note: elevated artifact % in raw data is expected and normal.")

    quality_list = []
    for subject_id in available_subjects:
        print(f"  sub-{subject_id}...", end=' ')
        q = quick_signal_quality_check(subject_id)
        quality_list.append(q)
        if 'error' not in q:
            print(f"{q['quality']} (flat: {q['flat_channels']}, "
                  f"noisy: {q['high_noise_channels']}, "
                  f"artifact: {q['artifact_pct']:.1f}%)")
        else:
            print(f"ERROR: {q.get('error')}")

    quality_df = pd.DataFrame(quality_list)
    quality_df.to_csv(OUTPUT_DIR / 'signal_quality.csv', index=False)

    # ------------------------------------------------------------------
    # Step 4: Visualisations
    # ------------------------------------------------------------------
    print("\n[4/4] Generating visualisations...")

    # Win-rate bar chart for all subjects
    plot_win_rate_summary(win_rate_df)

    # ERP preview for subjects passing the threshold
    if good_subjects:
        print(f"  ERP preview: {len(good_subjects)} included subjects...")
        plot_preliminary_comparison(good_subjects, filename_suffix='_good_subjects')

    # ERP preview for all available subjects (reference / comparison)
    print(f"  ERP preview: all {len(available_subjects)} subjects...")
    plot_preliminary_comparison(available_subjects, filename_suffix='_all_subjects')

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    full_df = metadata_df.merge(win_rate_df, on='subject_id', how='left')
    full_df = full_df.merge(quality_df, on='subject_id', how='left')
    full_df.to_csv(OUTPUT_DIR / 'full_subject_report.csv', index=False)

    n_good_quality = len(quality_df[quality_df.get('quality', '') == 'GOOD'])

    print("\n" + "=" * 70)
    print("SUBJECT SELECTION SUMMARY")
    print("=" * 70)
    print(f"  Total subjects:              {len(ALL_SUBJECTS)}")
    print(f"  Data available:              {len(available_subjects)}")
    print(f"  Meeting 60% win threshold:   {len(good_subjects)}")
    print(f"  Good signal quality:         {n_good_quality}")
    print(f"\n  Included subjects: {good_subjects}")
    print("\n  Reference (original paper):")
    print("    36 subjects → 24 after exclusions (12 failed 60% threshold)")
    print("    Expected subset from this BIDS slice: 27,28,31,34,35,36,37,38")

    # ------------------------------------------------------------------
    # Hand-off JSON for main pipeline
    # ------------------------------------------------------------------
    # Ensure it goes to the project root (parent of scripts/)
    script_dir = Path(__file__).parent.absolute()
    subjects_file = script_dir.parent / 'good_subjects.json'
    
    with open(subjects_file, 'w') as f:
        json.dump(good_subjects, f, indent=2)
    print(f"\n  good_subjects.json written to {subjects_file.name} ({len(good_subjects)} subjects)")
    print(f"  All reports → {OUTPUT_DIR}")
    print("=" * 70)

    return good_subjects


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    good_subjects = run_preliminary_analysis()

    if good_subjects:
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print(f"1. Review: {OUTPUT_DIR}full_subject_report.csv")
        print(f"2. Review plots in {OUTPUT_DIR}")
        print(f"3. Run complete-analysis-script_v3.py")
        print(f"   (will automatically read good_subjects.json)")
        print("=" * 70)
