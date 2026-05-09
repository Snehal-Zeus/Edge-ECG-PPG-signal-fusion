import os
import time
import joblib
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

def create_multiclass_label(row):
    hr = row['official_hr']
    rr_std = row['rr_std']
    if hr < 60: return 1
    elif hr > 100: return 2
    elif rr_std > 0.1: return 3
    else: return 0

train_df['multi_label'] = train_df.apply(create_multiclass_label, axis=1)
test_df['multi_label'] = test_df.apply(create_multiclass_label, axis=1)

features = ['rr_mean', 'rr_std', 'ptt_mean', 'r_peak_amp_mean', 'pulse_amp_mean', 'ecg_std', 'ppg_std']

X_train, y_train = train_df[features], train_df['multi_label']
X_test, y_test = test_df[features], test_df['multi_label']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

all_possible_names = {0: "Normal", 1: "Bradycardia", 2: "Tachycardia", 3: "Irregular/HRV"}

# 2. LOAD EXISTING BASELINE MODEL
print("\nLoading existing Baseline Model...")
try:
    baseline_model = joblib.load('baseline_model.pkl')
    print("Baseline model loaded successfully!")
except FileNotFoundError:
    print("Error: 'baseline_model.pkl' not found.")
    exit()


# 3. TRAIN SWEET-SPOT PRUNED MODEL
print("\nTraining Edge-Optimized (Pruned) Model with relaxed parameters...")
# Relaxed the parameters to boost accuracy back over 90%
pruned_model = RandomForestClassifier(
    n_estimators=40,           # Slightly more trees
    max_depth=8,               # Allowed to grow a bit deeper
    min_samples_leaf=3,        # Less strict on leaves
    ccp_alpha=0.001,           # Much gentler branch pruning
    random_state=42, 
    class_weight='balanced'
)
pruned_model.fit(X_train_scaled, y_train)

joblib.dump(pruned_model, 'pruned_model.pkl')


# 4. EVALUATION FUNCTION
def evaluate_edge_model(model, model_name, filename, is_pruned=False):
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()
    
    total_time_ms = (end_time - start_time) * 1000
    latency_per_segment = total_time_ms / len(y_test)
    
    acc = accuracy_score(y_test, y_pred) * 100
    model_size_kb = os.path.getsize(filename) / 1024
    
    print(f"\n" + "="*40)
    print(f" {model_name} METRICS ")
    print("="*40)
    print(f"Accuracy         : {acc:.2f}%")
    print(f"Model Size       : {model_size_kb:.1f} KB")
    print(f"Inference Latency: {latency_per_segment:.4f} ms per segment")
    
    if is_pruned:
        present_classes = sorted(list(set(y_test) | set(y_pred)))
        actual_class_names = [all_possible_names[c] for c in present_classes]
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, labels=present_classes, target_names=actual_class_names, zero_division=0))
        
        # Plot Pruned Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=present_classes)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=actual_class_names, yticklabels=actual_class_names)
        plt.title(f'Pruned Model Confusion Matrix\nAccuracy: {acc:.1f}% | Size: {model_size_kb:.1f} KB')
        plt.ylabel('True Medical Condition')
        plt.xlabel('Predicted Condition')
        plt.tight_layout()
        plt.savefig('plot_pruned_cm_1.png')
        plt.close()
        
        # Plot Pruned Feature Importance
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(7, 5))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title('Feature Importance (Pruned Edge Model)')
        plt.xlabel('Importance')
        plt.ylabel('Biological Feature')
        plt.tight_layout()
        plt.savefig('plot_pruned_feature_importance_1.png')
        plt.close()
        print("Saved: 'plot_pruned_cm_1.png' and 'plot_pruned_feature_importance_1.png'")
        
    return acc, model_size_kb, latency_per_segment

base_acc, base_size, base_lat = evaluate_edge_model(baseline_model, "BASELINE MODEL", "baseline_model.pkl", is_pruned=False)
prun_acc, prun_size, prun_lat = evaluate_edge_model(pruned_model, "PRUNED MODEL", "pruned_model.pkl", is_pruned=True)

# 5. VISUALIZE COMPARISON (BAR CHART)
labels = ['Accuracy (%)', 'Model Size (KB)', 'Latency (ms x 100)']
base_metrics = [base_acc, base_size, base_lat * 100]
prun_metrics = [prun_acc, prun_size, prun_lat * 100]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, base_metrics, width, label='Baseline', color='#e41a1c')
rects2 = ax.bar(x + width/2, prun_metrics, width, label='Pruned (Sweet Spot)', color='#4daf4a')

ax.set_ylabel('Scores / Size / Time')
ax.set_title('Edge Deployment: Baseline vs Pruned Model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('plot_pruning_comparison_1.png')
plt.close()