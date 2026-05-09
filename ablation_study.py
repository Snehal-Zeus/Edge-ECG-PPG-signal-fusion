import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

y_train = train_df['multi_label']
y_test = test_df['multi_label']

all_possible_names = {0: "Normal (0)", 1: "Bradycardia (1)", 2: "Tachycardia (2)", 3: "Irregular/HRV (3)"}

# 3. DEFINE ABLATION FEATURE SETS

experiments = {
    "ECG_Only": ['rr_mean', 'rr_std', 'r_peak_amp_mean', 'ecg_std'],
    "PPG_Only": ['pulse_amp_mean', 'ppg_std'],
    "Fusion": ['rr_mean', 'rr_std', 'ptt_mean', 'r_peak_amp_mean', 'pulse_amp_mean', 'ecg_std', 'ppg_std']
}


# 4. RUN ABLATION STUDY WITH DETAILED METRICS

#print("\n STARTING DETAILED FEATURE ABLATION STUDY...")
print("="*65)

results = {}

for exp_name, feature_list in experiments.items():
    print(f"\n EXPERIMENT: [{exp_name.replace('_', ' ')}]")
    print(f"Features: {', '.join(feature_list)}")
        
    X_train = train_df[feature_list]
    X_test = test_df[feature_list]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    y_test_pred = model.predict(X_test_scaled)
    

    # METRICS & REPORTS

    present_classes = sorted(list(set(y_test) | set(y_test_pred)))
    actual_class_names = [all_possible_names[c] for c in present_classes]
    
    accuracy = accuracy_score(y_test, y_test_pred) * 100
    results[exp_name] = accuracy
    
    print(f"Overall Accuracy: {accuracy:.2f}%\n")
    print(classification_report(y_test, y_test_pred, labels=present_classes, target_names=actual_class_names, zero_division=0))

    # SAVE CONFUSION MATRIX FOR EACH EXPERIMENT

    cm = confusion_matrix(y_test, y_test_pred, labels=present_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actual_class_names, yticklabels=actual_class_names)
    plt.title(f'Ablation: {exp_name.replace("_", " ")} Confusion Matrix\nAccuracy: {accuracy:.1f}%')
    plt.ylabel('True Medical Condition')
    plt.xlabel('Predicted Condition')
    plt.tight_layout()
    # Save with a specific name so they don't overwrite each other
    plt.savefig(f'plot_ablation_cm_{exp_name}.png') 
    plt.close()


# 5. PLOT FINAL COMPARISON BAR CHART

display_names = [name.replace('_', ' ') for name in results.keys()]

plt.figure(figsize=(8, 6))
bars = sns.barplot(x=display_names, y=list(results.values()), palette=['#4daf4a', '#377eb8', '#e41a1c'])

for i, acc in enumerate(results.values()):
    plt.text(i, acc - 5, f"{acc:.2f}%", ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

plt.title('Ablation Study: Impact of Signal Fusion on Diagnostic Accuracy', pad=20)
plt.ylabel('Multi-Class Classification Accuracy (%)')
plt.xlabel('Sensor Configuration')
plt.ylim([0, 105]) 

textstr = '\n'.join((
    r'$\bf{ECG\ Only:}$ rr_mean, rr_std, ecg_std, r_peak_amp',
    r'$\bf{PPG\ Only:}$ ppg_std, pulse_amp_mean',
    r'$\bf{Fusion:}$ All + ptt_mean'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('plot_ablation_bar_chart.png')
plt.close()

print("\n" + "="*65)
print(" ABLATION STUDY COMPLETE!")
print("- Saved Bar Chart: 'plot_ablation_bar_chart.png'")
print("- Saved 3 Confusion Matrices: 'plot_ablation_cm_ECG_Only.png', 'plot_ablation_cm_PPG_Only.png', 'plot_ablation_cm_Fusion.png'")