from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

print("Loading Zenith Master Hybrid Engine...")
# Load our newly minted Hybrid model
rf_model = joblib.load('models/hybrid_loan_model.pkl')
model_columns = joblib.load('models/hybrid_columns.pkl')
print("System Ready. Listening on Port 5000...")

@app.route('/')
def home():
    # This will load whichever index.html you currently have in your templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Extract raw inputs
        cibil = float(data['cibil'])
        term = float(data['term'])
        income = float(data['income'])
        loan_amount = float(data['loan_amount'])
        total_assets = float(data['total_assets']) 
        
        # ==========================================
        # 🛡️ STAGE 1: INSTITUTIONAL KNOCKOUT RULES
        # Stops the ₹1200 income bugs instantly!
        # ==========================================
        if income < 50000:
            return jsonify({'status': 'REJECTED', 'confidence': '100.00', 'message': f'KNOCKOUT: Income (₹{int(income):,}) is below the institutional minimum.'})
            
        if loan_amount > (income * 10):
            return jsonify({'status': 'REJECTED', 'confidence': '100.00', 'message': 'KNOCKOUT: Requested loan exceeds maximum safe Debt-to-Income limits.'})
            
        if cibil < 300 or cibil > 900:
            return jsonify({'status': 'ERROR', 'message': 'Invalid CIBIL score. Must be between 300 and 900.'})
        # ==========================================

        # ==========================================
        # 🧠 STAGE 2: CUSTOM FEATURE TRANSLATION
        # ==========================================
        cibil_tier = 0 if cibil < 600 else (1 if cibil <= 750 else 2)
        
        i_to_l = income / loan_amount if loan_amount > 0 else 0
        income_tier = 0 if i_to_l < 0.5 else (1 if i_to_l <= 1.5 else 2)
        
        shortfall = max(0, loan_amount - total_assets)
        rescue_power = income / (shortfall + 1)
        
        ml_input = {
            'cibil_tier': [cibil_tier],
            'income_tier': [income_tier],
            'asset_shortfall': [shortfall],
            'income_rescue_power': [rescue_power]
        }
        
        query_df = pd.DataFrame(ml_input)[model_columns]

        # ==========================================
        # 🤖 STAGE 3: RANDOM FOREST AI DECISION
        # ==========================================
        prediction = int(rf_model.predict(query_df)[0])
        probs = rf_model.predict_proba(query_df)[0]
        
        status = "REJECTED" if prediction == 1 else "APPROVED"
        confidence = f"{probs[prediction] * 100:.2f}"
        
        # ==========================================
        # 💬 STAGE 4: INTELLIGENT MESSAGING
        # Explains EXACTLY why the decision was made
        # ==========================================
        if status == "APPROVED":
            if shortfall > 0 and rescue_power >= 1.5:
                msg = "APPROVED: Applicant lacked hard assets, but exceptional cash flow rescued the application."
            elif shortfall > 0:
                msg = "APPROVED: Missing collateral, but strong credit history and stable income justified the risk."
            else:
                msg = "APPROVED: Flawless holistic profile. Fully collateralized."
        else:
            if cibil_tier == 0:
                msg = "REJECTED: Severe credit history degradation detected."
            elif shortfall > 0 and rescue_power < 1.0:
                msg = "REJECTED: High risk of default. Insufficient assets and weak cash flow to cover the shortfall."
            else:
                msg = "REJECTED: Failed multidimensional AI risk thresholds."

        return jsonify({'status': status, 'confidence': confidence, 'message': msg})

    except Exception as e:
        # Failsafe error catcher
        return jsonify({'status': 'ERROR', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)