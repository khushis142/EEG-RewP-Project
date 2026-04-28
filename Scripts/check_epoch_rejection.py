import json
import warnings
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt

import os
import importlib.util
import sys

# Get the directory of the current script
script_dir = Path(__file__).parent.absolute()

# Load complete-analysis-script_v3.py from the same directory
spec = importlib.util.spec_from_file_location(
    "complete_analysis", 
    script_dir / "complete-analysis-script_v3.py"
)
complete_analysis = importlib.util.module_from_spec(spec)
sys.modules["complete_analysis"] = complete_analysis
spec.loader.exec_module(complete_analysis)

preprocess_subject = complete_analysis.preprocess_subject
parse_events_by_condition = complete_analysis.parse_events_by_condition

warnings.filterwarnings('ignore')

# ============================================================================
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

BASE_OUTPUT_DIR = project_root / 'results' / 'rejection_checks'

# As requested by user
PRIMARY_CHANNEL  = 'FCz'
REWP_WINDOW      = (0.24, 0.32)   # s
BASELINE         = (-0.2, 0.0)    # s

THRESHOLDS = {
    '100uV': 100e-6,
    '120uV': 120e-6
}

# ============================================================================
# MAIN CHECK SCRIPT
# ============================================================================

def main():
    print(f"\n{'='*60}")
    print("EPOCH REJECTION THRESHOLD CHECK (ALL SUBJECTS & DROPPED EPOCHS)")
    print(f"Parameters: Channel {PRIMARY_CHANNEL}, Window {REWP_WINDOW}s, Baseline {BASELINE}s")
    print(f"{'='*60}")
    
    # Load subjects (check root first, then script dir)
    subjects_file = Path('good_subjects.json')
    if not subjects_file.exists():
        subjects_file = script_dir.parent / 'good_subjects.json'
        
    if not subjects_file.exists():
        print("Error: good_subjects.json not found.")
        return
        
    with open(subjects_file, 'r') as f:
        subjects_to_check = json.load(f)
        
    print(f"Found {len(subjects_to_check)} subjects in good_subjects.json.")
    print(f"Checking subjects: {subjects_to_check}")

    for sub_id in subjects_to_check:
        print(f"\n{'─'*60}")
        print(f"PROCESSING sub-{sub_id}")
        print(f"{'─'*60}")
        
        # Subject Specific Folder
        sub_dir = BASE_OUTPUT_DIR / f'sub-{sub_id}'
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        # We need a directory for preprocess_subject to load/save ICA cache
        out_temp = project_root / 'results' / 'complete_analysis' / f"sub-{sub_id}"
        out_temp.mkdir(parents=True, exist_ok=True)
        
        # 1. Load data just like the main pipeline
        raw_cl, labels, excluded, viz_data = preprocess_subject(sub_id, out_temp)
        events, ev_id, perf = parse_events_by_condition(sub_id, raw_cl)

        # Pre-calculate an identical epochs object without rejection to easily extract the rejected data later
        epochs_all = mne.Epochs(
            raw_cl, events, ev_id,
            tmin=-0.2, tmax=0.6,
            baseline=BASELINE,
            reject=None,
            preload=True, verbose=False,
        )
        
        # Create a combined figure
        # 4 Rows: 0=Image Heatmap, 1=Image Evoked, 2=Kept Traces, 3=Dropped Traces
        # 2 Columns: 0=100uV, 1=120uV
        fig, axes = plt.subplots(4, 2, figsize=(16, 22), gridspec_kw={'height_ratios': [2.5, 1, 1.5, 1.5]})
        fig.suptitle(f"sub-{sub_id} Epoch Rejection Report: {PRIMARY_CHANNEL}", fontsize=18, fontweight='bold')
        
        for col, (name, thresh) in enumerate(THRESHOLDS.items()):
            print(f"\n  Testing threshold: {name} ({thresh*1e6} µV)")
            
            # Epoch the data with rejection enabled
            epochs_clean = mne.Epochs(
                raw_cl, events, ev_id,
                tmin=-0.2, tmax=0.6,
                baseline=BASELINE,
                reject=dict(eeg=thresh),
                preload=True, verbose=False,
            )
            
            # Find which precisely were kept and dropped
            keep_mask = np.isin(epochs_all.selection, epochs_clean.selection)
            
            n_total = len(epochs_all)
            n_kept = len(epochs_clean)
            n_rej = n_total - n_kept
            pct_rej = (n_rej / n_total) * 100 if n_total > 0 else 0
            
            print(f"    Kept: {n_kept}/{n_total} epochs. Dropped: {n_rej} ({pct_rej:.1f}%)")
            
            if PRIMARY_CHANNEL in epochs_clean.ch_names:
                # ----------------------------------------------------
                # Top 2 rows: Heatmap + Evoked using plot_image
                # ----------------------------------------------------
                ax_img = axes[0, col]
                ax_evk = axes[1, col]
                
                # We plot the image for the clean epochs
                if n_kept > 0:
                    epochs_clean.plot_image(picks=[PRIMARY_CHANNEL], cmap='interactive', show=False, 
                                      axes=[ax_img, ax_evk], colorbar=False,
                                      title=f"Threshold: {name} | Epoch Heatmap (Kept)")
                else:
                    ax_img.set_title(f"Threshold: {name} | Epoch Heatmap (Kept)\nNO EPOCHS KEPT")
                
                ax_evk.set_ylabel("Amplitude (µV)", fontsize=10)
                ax_evk.set_xlabel("Time (s)", fontsize=10)
                
                # Extract full data for manual trace plots
                fcz_idx = epochs_all.ch_names.index(PRIMARY_CHANNEL)
                all_data = epochs_all.get_data()[:, fcz_idx, :] * 1e6  # Shape: (epochs, times) in uV
                
                kept_data = all_data[keep_mask]
                dropped_data = all_data[~keep_mask]
                
                # ----------------------------------------------------
                # Row 2 (Bottom-Mid): Kept Traces Butterfly plot
                # ----------------------------------------------------
                ax_kept = axes[2, col]
                if kept_data.ndim == 1:
                    kept_data = np.expand_dims(kept_data, axis=0) # Handle single epoch case safely
                
                for trial_idx in range(kept_data.shape[0]):
                    ax_kept.plot(epochs_clean.times, kept_data[trial_idx], color='gray', alpha=0.1, linewidth=0.5)
                
                if kept_data.shape[0] > 0:
                    ax_kept.plot(epochs_clean.times, kept_data.mean(axis=0), color='green', linewidth=2, label='Mean Kept ERP')
                else:
                    ax_kept.text(0.5, 0.5, "0 Kept Epochs", ha='center', va='center', transform=ax_kept.transAxes, fontsize=14, color='gray')

                limit_uv = (thresh * 1e6) / 2
                ax_kept.axvspan(*REWP_WINDOW, alpha=0.2, color='gold', label='RewP Window')
                ax_kept.axvline(0, color='black', linestyle='--')
                ax_kept.axhline(limit_uv, color='red', linestyle=':', alpha=0.5, label=f'+{limit_uv:.0f} µV Limit')
                ax_kept.axhline(-limit_uv, color='red', linestyle=':', alpha=0.5, label=f'-{limit_uv:.0f} µV Limit')
                
                ax_kept.set_title(f"Threshold: {name} | Kept Epochs Traces")
                ax_kept.set_xlabel("Time (s)")
                ax_kept.set_ylabel("Amplitude (µV)")
                ax_kept.set_ylim(-130, 130) # Fix y-axis to comfortably fit the bounds
                
                stats_text = f"Total Attempted: {n_total}\nKept: {n_kept}\nDropped: {n_rej} ({pct_rej:.1f}%)\nMax allowed P2P: {thresh*1e6:.0f} µV"
                ax_kept.text(0.02, 0.95, stats_text, transform=ax_kept.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
                ax_kept.legend(loc='upper right')

                # ----------------------------------------------------
                # Row 3 (Bottom): Dropped Traces Butterfly plot
                # ----------------------------------------------------
                ax_drop = axes[3, col]
                if dropped_data.ndim == 1:
                    dropped_data = np.expand_dims(dropped_data, axis=0)
                
                for trial_idx in range(dropped_data.shape[0]):
                    ax_drop.plot(epochs_clean.times, dropped_data[trial_idx], color='red', alpha=0.3, linewidth=1.0)
                
                if dropped_data.shape[0] > 0:
                    ax_drop.plot(epochs_clean.times, dropped_data.mean(axis=0), color='darkred', linewidth=2, label='Mean Dropped ERP')
                else:
                    ax_drop.text(0.5, 0.5, "0 Dropped Epochs\n(All Epochs Passed!)", ha='center', va='center', transform=ax_drop.transAxes, fontsize=14, color='green')

                ax_drop.axvspan(*REWP_WINDOW, alpha=0.2, color='gold', label='RewP Window')
                ax_drop.axvline(0, color='black', linestyle='--')
                ax_drop.axhline(limit_uv, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'+{limit_uv:.0f} µV Limit')
                ax_drop.axhline(-limit_uv, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'-{limit_uv:.0f} µV Limit')
                
                ax_drop.set_title(f"Threshold: {name} | Dropped Epochs Traces")
                ax_drop.set_xlabel("Time (s)")
                ax_drop.set_ylabel("Amplitude (µV)")
                
                # Make sure y-axis is wide enough to see the rejected peaks!
                if dropped_data.shape[0] > 0:
                    y_max = max(np.abs(dropped_data.max()), np.abs(dropped_data.min()), 130)
                    ax_drop.set_ylim(-y_max * 1.05, y_max * 1.05)
                else:
                    ax_drop.set_ylim(-130, 130)

                ax_drop.legend(loc='upper right')

            else:
                print(f"    Warning: {PRIMARY_CHANNEL} not found in channel names.")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Leave room for suptitle
        combined_file = sub_dir / f'sub-{sub_id}_rejection_summary_report.png'
        fig.savefig(combined_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved combined report to: {combined_file}")

if __name__ == "__main__":
    main()
