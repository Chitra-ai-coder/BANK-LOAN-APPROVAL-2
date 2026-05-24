import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')

print("🚀 INITIATING STAGE 1: HYBRID LOGIC TRAINING")

# 1. Load Data
try:
    df = pd.read_csv('data/loan_approval_dataset.csv')
    df.columns = df.columns.str.strip() 
    df['loan_status'] = df['loan_status'].map({' Approved': 0, ' Rejected': 1})
except FileNotFoundError:
    print("❌ ERROR: 'data/loan_approval_dataset.csv' not found.")
    exit()

df['total_assets'] = (df['residential_assets_value'] + 
                      df['commercial_assets_value'] + 
                      df['luxury_assets_value'] + 
                      df['bank_asset_value'])

# ==========================================
# 🧠 THE CUSTOM UNDERWRITING MATH
# ==========================================

# Feature 1: CIBIL Tiers
def categorize_cibil(score):
    if score < 600: return 0
    elif score <= 750: return 1
    else: return 2
df['cibil_tier'] = df['cibil_score'].apply(categorize_cibil)

# Feature 2: Income Tiers
df['income_to_loan'] = df['income_annum'] / df['loan_amount']
def categorize_income(ratio):
    if ratio < 0.5: return 0  
    elif ratio <= 1.5: return 1 
    else: return 2 
df['income_tier'] = df['income_to_loan'].apply(categorize_income)

# Feature 3: Asset Shortfall
df['asset_shortfall'] = df['loan_amount'] - df['total_assets']
df['asset_shortfall'] = df['asset_shortfall'].clip(lower=0) 

# Feature 4: Income Rescue Power
df['income_rescue_power'] = df['income_annum'] / (df['asset_shortfall'] + 1)

# ==========================================

# Package the Smart Variables
features = ['cibil_tier', 'income_tier', 'asset_shortfall', 'income_rescue_power']
X = df[features]
y = df['loan_status']

joblib.dump(list(X.columns), 'models/hybrid_columns.pkl')

# Train the Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⚙️ Random Forest is calculating optimal risk thresholds...")
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=6, 
    min_samples_split=5, 
    random_state=42
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n======================================")
print(f"🎯 HYBRID SYSTEM ACCURACY: {acc * 100:.2f}%")
print("======================================\n")

joblib.dump(rf_model, 'models/hybrid_loan_model.pkl')
print("💾 SUCCESS: Master Hybrid Model saved to 'models/hybrid_loan_model.pkl'")