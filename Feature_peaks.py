import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def generate_inline_annotated_figure(record_path, start_sec=20, duration=2.0):
    try:
        # 1. Load Data
        signals, fields = wfdb.rdsamp(record_path)
        fs = fields['fs']
        sig_names = [s.replace(',', '').strip() for s in fields['sig_name']]
        
        ppg_idx = sig_names.index('PLETH')
        ecg_idx = sig_names.index('II') if 'II' in sig_names else sig_names.index('V')
        
        # 2. Extract Window
        start_idx = int(start_sec * fs)
        end_idx = int((start_sec + duration) * fs)
        ecg = signals[start_idx:end_idx, ecg_idx]
        ppg = signals[start_idx:end_idx, ppg_idx]
        time = np.linspace(0, len(ecg)/fs, len(ecg))

        # 3. Peak Detection
        r_peaks, _ = find_peaks(ecg, distance=int(fs*0.6), height=np.max(ecg)*0.4)
        p_peaks, _ = find_peaks(ppg, distance=int(fs*0.6), height=np.mean(ppg))

        # 4. Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        plt.subplots_adjust(hspace=0.1)

        # --- ECG TOP PANEL ---
        ax1.plot(time, ecg, color='#d62728', lw=1.5, label='Lead-II ECG')
        ax1.scatter(time[r_peaks], ecg[r_peaks], color='black', marker='v', s=100)
        
        # R-R Duration: In-line with ECG peaks
        if len(r_peaks) >= 2:
            r1, r2 = r_peaks[0], r_peaks[1]
            y_r_line = ecg[r1] # Same height as the peak
            ax1.annotate('', xy=(time[r1], y_r_line), xytext=(time[r2], y_r_line),
                         arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
            ax1.text((time[r1]+time[r2])/2, y_r_line + 0.05, f'R-R Duration: {time[r2]-time[r1]:.3f}s', 
                     ha='center', color='blue', fontweight='bold', fontsize=10)
        
        ax1.set_ylim(np.min(ecg)-0.2, np.max(ecg)+0.5) # Extra space for labels
        ax1.set_ylabel('Signal Type:\nECG (mV)', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- PPG BOTTOM PANEL ---
        ax2.plot(time, ppg, color='#1f77b4', lw=1.5, label='Finger PLETH')
        ax2.scatter(time[p_peaks], ppg[p_peaks], color='black', marker='o', s=100)

        # PLETH Peak Duration: IN-LINE with PPG peaks
        if len(p_peaks) >= 2:
            p1, p2 = p_peaks[0], p_peaks[1]
            y_p_line = ppg[p1] # Align exactly with the peak height
            ax2.annotate('', xy=(time[p1], y_p_line), xytext=(time[p2], y_p_line),
                         arrowprops=dict(arrowstyle='<->', color='#00ced1', lw=2))
            ax2.text((time[p1]+time[p2])/2, y_p_line + 0.03, f'PLETH Peak Duration: {time[p2]-time[p1]:.3f}s', 
                     ha='center', color='#00ced1', fontweight='bold', fontsize=10)

        # --- SEQUENTIAL PTT ---
        if len(r_peaks) > 0:
            r_t = time[r_peaks[0]]
            following_p = p_peaks[p_peaks > r_peaks[0]]
            if len(following_p) > 0:
                p_t = time[following_p[0]]
                for ax in [ax1, ax2]: ax.axvline(r_t, color='gray', ls='--', alpha=0.4)
                ax2.axvline(p_t, color='gray', ls='--', alpha=0.4)
                
                # PTT Arrow (Middle of signal to avoid overlap)
                y_mid = np.mean(ppg)
                ax2.annotate('', xy=(r_t, y_mid), xytext=(p_t, y_mid),
                             arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=3))
                ax2.text((r_t + p_t)/2, y_mid + 0.05, f'PTT: {p_t - r_t:.3f}s', 
                         ha='center', color='#2ca02c', fontweight='bold')

        ax2.set_ylim(np.min(ppg)-0.1, np.max(ppg)+0.3) # Extra space for labels
        ax2.set_ylabel('Signal Type:\nPPG (AU)', fontweight='bold')
        ax2.set_xlabel('Time (Seconds)')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.suptitle(f"Final In-Line Feature Mapping: {os.path.basename(record_path)}", fontsize=15)
        plt.savefig('Project_InLine_Annotated.png', dpi=300, bbox_inches='tight')
        print("SUCCESS: In-line figure saved.")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    search_dir = r"E:\mimic_env"
    target_record = "bidmc12"
    found_path = None
    for root, dirs, files in os.walk(search_dir):
        if f"{target_record}.hea" in files:
            found_path = os.path.join(root, target_record)
            break
    if found_path:
        generate_inline_annotated_figure(found_path)