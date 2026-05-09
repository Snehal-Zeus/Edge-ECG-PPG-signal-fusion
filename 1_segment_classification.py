import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib 
# 1. LOAD DATA

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

features = ['rr_mean', 'rr_std', 'ptt_mean', 'r_peak_amp_mean', 'pulse_amp_mean', 'ecg_std', 'ppg_std']

X_train, y_train = train_df[features], train_df['multi_label']
X_test, y_test = test_df[features], test_df['multi_label']


# 3. SCALE & TRAIN MODEL

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#print("Training Multi-Class Random Forest...\n")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 4. PRINT CLASS DISTRIBUTION IN TEST SET

all_possible_names = {0: "Normal (0)", 1: "Bradycardia (1)", 2: "Tachycardia (2)", 3: "Irregular/HRV (3)"}

print("=== TEST SET CLASS DISTRIBUTION ===")
class_counts = y_test.value_counts().sort_index()
for class_num, count in class_counts.items():
    print(f"{all_possible_names[class_num]}: {count} segments")
#print("===================================\n")


# 5. TEST SET EVALUATION METRICS

y_test_pred = model.predict(X_test_scaled)

# Dynamically find which classes are actually in the test set to prevent crashes
present_classes = sorted(list(set(y_test) | set(y_test_pred)))
actual_class_names = [all_possible_names[c] for c in present_classes]

accuracy = accuracy_score(y_test, y_test_pred)
print("=== FINAL TEST SET EVALUATION ===")
print(f"Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Report:")
print(classification_report(y_test, y_test_pred, labels=present_classes, target_names=actual_class_names, zero_division=0))


# 6. PLOT CONFUSION MATRIX

cm = confusion_matrix(y_test, y_test_pred, labels=present_classes)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actual_class_names, yticklabels=actual_class_names)
plt.title(f'Test Set Confusion Matrix\nAccuracy: {accuracy * 100:.1f}%')
plt.ylabel('True Medical Condition')
plt.xlabel('Predicted Condition')
plt.tight_layout()
plt.savefig('plot_multiclass_cm.png')
plt.close()
print("Saved Confusion Matrix: 'plot_multiclass_cm.png'")


# 7. PLOT FEATURE IMPORTANCE

importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='magma')
plt.title('Feature Importance (Multi-Class RF)')
plt.xlabel('Importance (%)')
plt.ylabel('Biological Feature')
plt.tight_layout()
plt.savefig('plot_multiclass_feature_importance.png')
plt.close()
print("Saved Feature Importance: 'plot_multiclass_feature_importance.png'")

# 8. SAVE PREDICTIONS 

test_results_df = pd.DataFrame({
    'record': test_df['record'],
    'true_segment': y_test,
    'pred_segment': y_test_pred
})
test_results_df.to_csv('segment_predictions_test.csv', index=False)
print("\n Predictions saved to 'segment_predictions_test.csv'.")


joblib.dump(model, 'baseline_model.pkl')
print("Model saved as 'baseline_model.pkl'")