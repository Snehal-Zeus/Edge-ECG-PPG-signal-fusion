Edge-Efficient Heart Health Analysis Using ECG-PPG Signal Fusion

📌 Project Overview
Cardiovascular diseases, particularly paroxysmal arrhythmias like Atrial Fibrillation (AFib), often evade detection in standard clinical environments due to their intermittent nature. While continuous monitoring via wearables is essential, highly accurate diagnostic systems typically rely on computationally heavy deep learning models, making them unsuitable for memory-constrained edge devices (like smartwatches or microcontrollers).

This project proposes a highly optimized, clinically interpretable machine learning pipeline that fuses Electrocardiogram (ECG) and Photoplethysmogram (PPG) signals. By extracting physiological biomarkers—most notably Pulse Transit Time (PTT)—and applying Cost Complexity Pruning (CCP) to a Random Forest classifier, the model size is reduced by 71% (down to ~216 KB). The system achieves 100% patient-level diagnostic accuracy on the unseen test cohort, proving its viability for low-power wearable IoT devices.

🚀 Key Features
Sensor Fusion Pipeline: Fuses electrical (ECG) and mechanical (PPG) data to build a robust diagnostic system resilient to single-sensor artifacts.

Edge-AI Optimized: Uses Cost Complexity Pruning (CCP) to compress the Random Forest model to fit within standard microcontroller SRAM limits (e.g., ESP32, ARM Cortex-M).

Multi-Class Arrhythmia Detection: Classifies 10-second physiological windows into four clinical states: Normal, Bradycardia, Tachycardia, and Irregular/AFib.

Patient-Level Diagnosis: Employs a "Majority Voting" mechanism across patient segments to eliminate localized noise and achieve 100% clinical diagnosis accuracy.

📂 Repository Structure & Pipeline
Phase 1: Data Processing
bidmc_extract.py: Processes raw BIDMC signals using neurokit2, applies bandpass filters, detects peaks, and extracts biological features (Heart Rate, HRV, PTT, etc.) using a 10-second sliding window.

Feature_peaks.py: Visualizes synchronized ECG and PPG signals, explicitly mapping the calculation of Pulse Transit Time (PTT).

Phase 2: Data Preparation
spliting_data.py: Splits the extracted features into Training, Validation, and Testing sets by patient to prevent data leakage, and assigns multi-class clinical labels.

Phase 3: Model Training & Validation
1_segment_classification.py: Trains the baseline Multi-Class Random Forest model on 10-second segments and generates feature importance metrics.

ablation_study.py: Conducts an ablation experiment comparing ECG-only, PPG-only, and Fusion approaches to scientifically validate the necessity of multi-sensor fusion.

Phase 4: Edge Optimization & Clinical Diagnosis
pruning_1.py: Applies Cost Complexity Pruning (CCP) to restrict tree depth, shrinking the model by 71% while evaluating latency and memory footprint.

patient_class.py: Translates 10-second segment predictions into overall clinical diagnoses using Majority Voting to evaluate final patient-level accuracy.

📊 Dataset
This project uses the BIDMC PPG and Respiration Dataset from PhysioNet, containing synchronized ECG and PPG recordings from 53 ICU patients.
(Note: To allow for immediate reproducibility without downloading the 50GB+ raw dataset, the extracted mathematical features train_features.csv and test_features.csv are included in this repository).

🛠️ Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies using:

Bash
pip install pandas numpy scikit-learn matplotlib seaborn neurokit2 wfdb joblib
💻 How to Run the Project
(Optional) Extract Features: If using raw BIDMC data, update paths and run python bidmc_extract.py. Otherwise, skip to step 2 using the provided CSVs.

Split Data: Run python spliting_data.py to organize the patient cohorts.

Train Baseline: Run python 1_segment_classification.py to evaluate the base accuracy and view feature importance.

Run Ablation Study: Run python ablation_study.py to view the diagnostic impact of signal fusion.

Optimize Model: Run python pruning_1.py to compress the model for edge deployment.

Evaluate Final Diagnosis: Run python patient_class.py to apply majority voting and view the final patient-level confusion matrix.
