import os
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION PATHS
# ==========================================
# Input file containing all 53 patients
INPUT_CSV = r"E:\mimic_env\BIDMC_project\bidmc_features.csv"

# Output main folder for the dataset splits
OUTPUT_DIR = r"E:\mimic_env\BIDMC_project\dataset_splits_1"  

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "validation")
test_dir = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ==========================================
# 2. LOAD DATA & CREATE MULTI-CLASS LABELS
# ==========================================
print("Loading the complete dataset...")
df = pd.read_csv(INPUT_CSV).dropna()

# Function to create multi-class labels (4 Classes)
def create_multiclass_label(row):
    hr = row['official_hr']
    rr_std = row['rr_std']
    
    if hr < 60:
        return 1  # Bradycardia
    elif hr > 100:
        return 2  # Tachycardia
    elif rr_std > 0.1:
        return 3  # Irregular Rhythm (High HRV)
    else:
        return 0  # Normal

print("Generating clinical multi-class labels...")
df['multi_label'] = df.apply(create_multiclass_label, axis=1)

# Save a master copy with the new labels just in case you need it later
master_output = os.path.join(OUTPUT_DIR, "bidmc_features_multiclass.csv")
df.to_csv(master_output, index=False)
print(f"Master file with new labels saved to: {master_output}")

# ==========================================
# 3. PATIENT-WISE SPLIT (70% / 15% / 15%)
# ==========================================
print("Splitting dataset by patients to prevent data leakage...")

# Get unique patient IDs
unique_patients = df['record'].unique()
total_patients = len(unique_patients)

# Shuffle patients randomly (Seed 42 keeps the shuffle consistent)
np.random.seed(42)
np.random.shuffle(unique_patients)

# Calculate indices for the split
train_idx = int(0.70 * total_patients)
val_idx = int(0.85 * total_patients)

# Slice the lists of patient IDs
train_patients = unique_patients[:train_idx]
val_patients = unique_patients[train_idx:val_idx]
test_patients = unique_patients[val_idx:]

# Group actual data rows based on which list the patient ended up in
train_df = df[df['record'].isin(train_patients)]
val_df = df[df['record'].isin(val_patients)]
test_df = df[df['record'].isin(test_patients)]

# ==========================================
# 4. SAVE SPLITS TO FOLDERS
# ==========================================
train_file = os.path.join(train_dir, "train_features.csv")
val_file = os.path.join(val_dir, "validation_features.csv")
test_file = os.path.join(test_dir, "test_features.csv")

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

# ==========================================
# 5. PRINT STATISTICS
# ==========================================
print("\n" + "="*50)
print("✅ Dataset Successfully Labeled and Split!")
print("="*50)
print(f"Total Patients: {total_patients} | Total 10s Segments: {len(df)}")
print("-" * 50)
print(f"📁 TRAIN FOLDER      (70%): {len(train_patients):2d} patients | {len(train_df):4d} segments")
print(f"📁 VALIDATION FOLDER (15%): {len(val_patients):2d} patients | {len(val_df):4d} segments")
print(f"📁 TEST FOLDER       (15%): {len(test_patients):2d} patients | {len(test_df):4d} segments")
print("="*50)

# Quick look at class distributions in the test set
print("\nTest Set Class Distribution:")
print(test_df['multi_label'].value_counts().rename({0: 'Normal', 1: 'Bradycardia', 2: 'Tachycardia', 3: 'Irregular/HRV'}))