import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import joblib
from sklearn.preprocessing import LabelEncoder

print("🔍 ANALYZING SVM BRAIN: CALCULATING FEATURE IMPORTANCE...")

# 1. Load the exact model and scaler we are using
try:
    svm_model = joblib.load('models/svm_vehicle_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    df = pd.read_csv('data/loan_approval_dataset.csv')
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("❌ ERROR: Could not find model, scaler, or data files.")
    exit()

# 2. Re-create the exact preprocessing steps
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])
df['loan_status'] = df['loan_status'].map({' Approved': 0, ' Rejected': 1})

X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

# 3. Scale the data exactly like the live website does
X_scaled = scaler.transform(X)

# 4. RUN PERMUTATION IMPORTANCE
# This shuffles each column 5 times to see how much the AI relies on it
print("⚙️ Shuffling data matrices to test model dependency. Please wait...")
result = permutation_importance(svm_model, X_scaled, y, n_repeats=5, random_state=42, n_jobs=-1)

# 5. Build the Results Table
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

# 6. Visualize the Results
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

# Plotting the impact of each feature
sns.barplot(x='Importance', y='Feature', hue='Feature', data=importance_df, palette='magma', legend=False)

plt.title('What Drives the Zenith AI? (SVM Permutation Importance)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Impact on Accuracy (Higher Bar = More Important)', fontsize=12)
plt.ylabel('Applicant Attribute', fontsize=12)

plt.tight_layout()
print("📈 Analysis complete. Rendering chart...")
plt.show()