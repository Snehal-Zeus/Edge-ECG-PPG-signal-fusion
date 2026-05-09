import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


# 1. LOAD THE DATASET

train_file = r"E:\mimic_env\BIDMC_project\Final\dataset_splits_1\train\train_features.csv"
test_file = r"E:\mimic_env\BIDMC_project\Final\dataset_splits_1\test\test_features.csv"

train_df = pd.read_csv(train_file).dropna()
test_df = pd.read_csv(test_file).dropna()


# 2. CREATE MULTI-CLASS LABELS

def create_multiclass_label(row):
    hr = row['official_hr']
    rr_std = row['rr_std']
    if hr < 60: return 1         # Bradycardia
    elif hr > 100: return 2      # Tachycardia
    elif rr_std > 0.1: return 3  # Irregular Rhythm (High HRV)
    else: return 0               # Normal

train_df['multi_label'] = train_df.apply(create_multiclass_label, axis=1)
test_df['multi_label'] = test_df.apply(create_multiclass_label, axis=1)

# Extract biological features
features = ['rr_mean', 'rr_std', 'ptt_mean', 'r_peak_amp_mean', 'pulse_amp_mean', 'ecg_std', 'ppg_std']

X_train = train_df[features]
y_train = train_df['multi_label']

X_test = test_df[features]
y_test = test_df['multi_label']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. TRAIN BOTH MODELS

print("Training Baseline Model...")
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline_model.fit(X_train_scaled, y_train)

print("Training Pruned Edge-Model...")
pruned_model = RandomForestClassifier(n_estimators=30, max_depth=6, min_samples_split=10, random_state=42, n_jobs=-1)
pruned_model.fit(X_train_scaled, y_train)

# 4. PATIENT-LEVEL EVALUATION LOGIC

class_mapping = {0: 'Normal', 1: 'Bradycardia', 2: 'Tachycardia', 3: 'Irregular/HRV'}

def evaluate_patient_level(model, model_name):
    print(f"\n========================================")
    print(f" {model_name.upper()} - PATIENT LEVEL METRICS ")
    print(f"========================================")
    
    # 1. Predict labels for all individual 10-second segments in the test set
    test_df['predicted_label'] = model.predict(X_test_scaled)
    
    # 2. Majority Voting: Group by Patient ID ('record') and find the mode (most common label)
    #  aggregate both the True Label and the Predicted Label for the entire patient
    patient_results = test_df.groupby('record').agg(
        true_patient_label=('multi_label', lambda x: x.mode()[0]),
        pred_patient_label=('predicted_label', lambda x: x.mode()[0])
    ).reset_index()
    
    y_patient_true = patient_results['true_patient_label']
    y_patient_pred = patient_results['pred_patient_label']
    
    # 3. Calculate Overall Patient Accuracy
    acc = accuracy_score(y_patient_true, y_patient_pred)
    print(f"Patient-Level Accuracy: {acc * 100:.2f}%")
    
    # 4. Filter class names dynamically to avoid errors if Bradycardia (Class 1) is entirely missing
    present_classes = sorted(list(set(y_patient_true).union(set(y_patient_pred))))
    actual_class_names = [class_mapping[c] for c in present_classes]
    
    print("\nDetailed Patient-Level Classification Report:")
    print(classification_report(y_patient_true, y_patient_pred, 
                                labels=present_classes, 
                                target_names=actual_class_names, 
                                zero_division=0))
    
    # 5. Plot Patient-Level Confusion Matrix
    cm = confusion_matrix(y_patient_true, y_patient_pred, labels=present_classes)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if 'Pruned' in model_name else 'Blues', 
                xticklabels=actual_class_names, yticklabels=actual_class_names)
    
    plt.title(f'{model_name}\nPatient-Level Confusion Matrix (Accuracy: {acc*100:.1f}%)', pad=15)
    plt.ylabel('True Patient Diagnosis')
    plt.xlabel('Predicted Patient Diagnosis (via Majority Vote)')
    plt.tight_layout()
    
    # Save the plot securely
    filename = f"plot_patient_cm_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved Patient-Level Confusion Matrix: '{filename}'")


# 5. RUN THE EVALUATION
evaluate_patient_level(baseline_model, "Baseline Model")
evaluate_patient_level(pruned_model, "Pruned Edge Model")

print("\nEvaluation Complete! Check folder for the generated plot images.")