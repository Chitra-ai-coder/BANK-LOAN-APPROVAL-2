import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 INITIALIZING ZENITH DATA EXPLORER...")

# 1. Load the dataset
try:
    df = pd.read_csv('data/loan_approval_dataset.csv')
    # Clean column names (Kaggle datasets often have hidden spaces)
    df.columns = df.columns.str.strip()
    print(f"✅ Loaded {len(df)} Indian loan records successfully.")
except FileNotFoundError:
    print("❌ ERROR: 'loan_approval_dataset.csv' not found in 'data/' folder.")
    exit()

# 2. Set Visual Theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

# 3. Create Dashboard Layout
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Indian Loan Approval: Feature Correlation Analysis', fontsize=22, fontweight='bold', y=0.98)

# --- Chart 1: Loan Status Distribution ---
sns.countplot(x='loan_status', data=df, ax=axes[0, 0], palette=['#10b981', '#ef4444'])
axes[0, 0].set_title('Target Variable: Loan Approval Status', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Status (Approved vs Rejected)')

# --- Chart 2: CIBIL Score vs Loan Status ---
# This is the most important chart for the Indian market
sns.kdeplot(data=df, x='cibil_score', hue='loan_status', fill=True, ax=axes[0, 1], palette=['#10b981', '#ef4444'])
axes[0, 1].set_title('The CIBIL Factor: Credit Score Density', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('CIBIL Score (300 - 900)')

# --- Chart 3: Income vs Loan Amount (Economic Heatmap) ---
sns.scatterplot(data=df, x='income_annum', y='loan_amount', hue='loan_status', alpha=0.6, ax=axes[1, 0], palette=['#10b981', '#ef4444'])
axes[1, 0].set_title('Income vs. Loan Magnitude', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Annual Income (INR)')
axes[1, 0].set_ylabel('Requested Loan (INR)')

# --- Chart 4: Self-Employed vs Education Risk ---
# We use a cross-tabulation to see how employment type affects approval
sns.countplot(x='self_employed', hue='loan_status', data=df, ax=axes[1, 1], palette=['#10b981', '#ef4444'])
axes[1, 1].set_title('Employment Type Risk Analysis', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Is Self Employed?')

# Adjust and Show
plt.tight_layout(pad=3.0)
print("📈 Dashboard ready. Rendering window...")
plt.show()