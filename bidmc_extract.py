import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings


warnings.filterwarnings("ignore")

BIDMC_CSV_DIR = r"E:\mimic_env\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc-ppg-and-respiration-dataset-1.0.0\bidmc_csv"      # Folder containing the 53 patients' CSV files
OUTPUT_FILE = r"E:\mimic_env\bidmc_features.csv" 
FS = 125          # BIDMC signals are sampled at 125 Hz
WINDOW_SEC = 10   # 10-second segments
STEP_SEC = 5      # 5-second overlap (sliding window)

def setup_csv():
    """Creates the final dataset file with headers if it doesn't exist."""
    if not os.path.exists(OUTPUT_FILE):
        header = [
            "record", "segment_start_sec", 
            "rr_mean", "rr_std", 
            "ptt_mean", "r_peak_amp_mean", "pulse_amp_mean", 
            "ecg_std", "ppg_std", 
            "official_hr", "label"
        ]
        pd.DataFrame(columns=header).to_csv(OUTPUT_FILE, index=False)

def extract_features():
    setup_csv()
    print("Starting BIDMC Signal Processing & Extraction...")
    print(f"Reading from: {BIDMC_CSV_DIR}")
    print(f"Saving to: {OUTPUT_FILE}\n")

    # Get all the 'Signals.csv' files from the folder
    try:
        signal_files = [f for f in os.listdir(BIDMC_CSV_DIR) if f.endswith("_Signals.csv")]
    except FileNotFoundError:
        print(f"Error: Could not find folder {BIDMC_CSV_DIR}. Please check the path.")
        return

    if len(signal_files) == 0:
        print(f"Error: No '_Signals.csv' files found in {BIDMC_CSV_DIR}.")
        return

    for file in signal_files:
        # Get patient ID (e.g., "bidmc_01")
        record_name = file.replace("_Signals.csv", "")
        signals_path = os.path.join(BIDMC_CSV_DIR, file)
        numerics_path = os.path.join(BIDMC_CSV_DIR, f"{record_name}_Numerics.csv")
        
        try:
            # 1. LOAD DATA
            df_signals = pd.read_csv(signals_path)
            df_numerics = pd.read_csv(numerics_path)
            
            # Clean column names (BIDMC CSVs have hidden spaces like ' PLETH')
            df_signals.columns = df_signals.columns.str.strip()
            df_numerics.columns = df_numerics.columns.str.strip()
            
            # Extract raw signal arrays
            ppg_raw = df_signals['PLETH'].values
            ecg_raw = df_signals['II'].values if 'II' in df_signals.columns else df_signals['V'].values
            
            samples_win = WINDOW_SEC * FS
            samples_step = STEP_SEC * FS
            segments_processed = 0
            
            # 2. SLIDING WINDOW SEGMENTATION
            for start in range(0, len(ecg_raw) - samples_win, samples_step):
                # Cut the 10-second arrays
                ecg_seg = ecg_raw[start:start + samples_win]
                ppg_seg = ppg_raw[start:start + samples_win]
                
                # Calculate the exact second this segment starts and ends
                start_sec = int(start / FS)
                end_sec = int((start + samples_win) / FS)
                
                # 3. GET GROUND TRUTH FROM HOSPITAL MONITOR
                # The Numerics file logs data at 1 Hz (1 row per second).
                hr_window = df_numerics['HR'].iloc[start_sec:end_sec]
                official_hr = hr_window.mean()
                
                # Skip if hospital monitor was disconnected or reading 0
                if pd.isna(official_hr) or official_hr <= 0:
                    continue 
                
                # 4. SIGNAL QUALITY CHECK
                if np.isnan(ecg_seg).any() or np.isnan(ppg_seg).any():
                    continue
                if np.std(ecg_seg) < 0.05 or np.std(ppg_seg) < 0.01: # Skip flatlines
                    continue
                    
                try:
                    # 5. NEUROKIT PREPROCESSING (Filtering)
                    ecg_clean = nk.ecg_clean(ecg_seg, sampling_rate=FS, method="neurokit", powerline=60)
                    ppg_clean = nk.ppg_clean(ppg_seg, sampling_rate=FS, method="elgendi")
                    
                    # 6. PEAK DETECTION
                    _, ecg_info = nk.ecg_process(ecg_clean, sampling_rate=FS)
                    _, ppg_info = nk.ppg_process(ppg_clean, sampling_rate=FS)
                    
                    r_peaks = np.array(ecg_info.get("ECG_R_Peaks", []))
                    ppg_points = np.array(ppg_info.get("PPG_Onsets", ppg_info.get("PPG_Peaks", [])))
                    
                    # Remove any accidental NaNs
                    r_peaks = r_peaks[~np.isnan(r_peaks)]
                    ppg_points = ppg_points[~np.isnan(ppg_points)]
                    
                    # Require at least 5 heartbeats in 10 seconds to do math safely
                    if len(r_peaks) < 5 or len(ppg_points) < 5: 
                        continue
                        
                    # 7. EXTRACT MACHINE LEARNING FEATURES
                    rr_intervals = np.diff(r_peaks) / FS
                    rr_mean = np.mean(rr_intervals)
                    rr_std = np.std(rr_intervals)
                    
                    r_peak_amp_mean = np.mean(ecg_clean[r_peaks])
                    pulse_amp_mean = np.mean(ppg_clean[ppg_points])
                    
                    # Pulse Transit Time (PTT): Time from R-peak to the very next PPG pulse
                    ptts = []
                    for r in r_peaks:
                        after = ppg_points[ppg_points > r]
                        if len(after) > 0:
                            ptts.append((after[0] - r) / FS)
                            
                    mean_ptt = np.mean(ptts) if len(ptts) > 0 else np.nan
                    
                    # If any math failed, skip this row
                    if np.isnan([mean_ptt, rr_mean, rr_std, r_peak_amp_mean]).any():
                        continue 
                    
                    # 8. CLINICAL LABELING
                    # Label 1 (Unhealthy/Arrhythmia) if Official HR is extreme or HRV is very high
                    label = 1 if (official_hr > 100 or official_hr < 60 or rr_std > 0.1) else 0
                    
                    # 9. SAVE TO CSV
                    row_data = pd.DataFrame([{
                        "record": record_name,
                        "segment_start_sec": start_sec,
                        "rr_mean": rr_mean,
                        "rr_std": rr_std,
                        "ptt_mean": mean_ptt,
                        "r_peak_amp_mean": r_peak_amp_mean,
                        "pulse_amp_mean": pulse_amp_mean,
                        "ecg_std": np.std(ecg_clean),
                        "ppg_std": np.std(ppg_clean),
                        "official_hr": official_hr,
                        "label": label
                    }])
                    
                    row_data.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                    segments_processed += 1
                    
                except Exception:
                    # Ignore neurokit math errors on messy segments and continue to the next 10s window
                    continue 
                
            print(f"Processed {record_name}: Extracted {segments_processed} clean segments.")
            
        except Exception as e:
            print(f"Failed on {record_name}: {e}")

    print(f"\nExtraction Complete! Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_features()