from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.exceptions import BadRequest

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ==========================================
# LOAD TRAINED MODEL AND ARTIFACTS
# ==========================================

# Initialize with default values
model = None
label_encoders = None
scaler = None
feature_weights = {}
feature_names = []

# Get absolute path to the directory containing this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "loan_model.pkl"))
    label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_weights = joblib.load(os.path.join(BASE_DIR, "feature_weights.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    print("✓ All artifacts loaded successfully")
except Exception as e:
    print(f"⚠ Error loading model: {e}")
    print("  Please run: python loan_approval_model.py")

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html', feature_weights=feature_weights)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for loan prediction"""
    try:
        data = request.json
        
        # Extract and prepare features
        features_dict = {
            "Gender": data.get("gender"),
            "Married": data.get("married"),
            "Dependents": data.get("dependents"),
            "Education": data.get("education"),
            "Self_Employed": data.get("self_employed"),
            "ApplicantIncome": float(data.get("applicant_income", 0)),
            "CoapplicantIncome": float(data.get("coapplicant_income", 0)),
            "LoanAmount": float(data.get("loan_amount", 0)),
            "Loan_Amount_Term": float(data.get("loan_term", 0)),
            "Credit_History": float(data.get("credit_history", 0)),
            "Property_Area": data.get("property_area")
        }
        
        # Feature Engineering (same as training)
        total_income = features_dict["ApplicantIncome"] + features_dict["CoapplicantIncome"]
        
        features_dict["TotalIncome"] = total_income
        features_dict["Applicant_Income_Log"] = np.log1p(features_dict["ApplicantIncome"])
        features_dict["Total_Income_Log"] = np.log1p(total_income)
        features_dict["Loan_Amount_Log"] = np.log1p(features_dict["LoanAmount"])
        features_dict["Income_to_Loan_Ratio"] = total_income / (features_dict["LoanAmount"] + 1)
        features_dict["DTI_Approximation"] = features_dict["LoanAmount"] / (total_income + 1)
        features_dict["Loan_Applicant_Income_Ratio"] = features_dict["LoanAmount"] / (features_dict["ApplicantIncome"] + 1)
        
        # Encode categorical variables
        categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
        
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                features_dict[col] = le.transform([str(features_dict[col])])[0]
        
        # Prepare feature array in correct order
        feature_vector = []
        for fname in feature_names:
            if fname in features_dict:
                feature_vector.append(features_dict[fname])
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        feature_scaled = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(feature_scaled)[0]
        prediction_proba = model.predict_proba(feature_scaled)[0]
        
        approval_prob = float(prediction_proba[1])
        
        # Determine risk level
        if approval_prob >= 0.80:
            risk_level = "Low Risk"
            risk_color = "green"
        elif approval_prob >= 0.60:
            risk_level = "Moderate Risk"
            risk_color = "yellow"
        else:
            risk_level = "High Risk"
            risk_color = "red"
        
        # Calculate eligibility score
        eligibility_score = int(approval_prob * 100)
        
        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "approval_probability": approval_prob,
            "eligibility_score": eligibility_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "approval_chance": round(approval_prob * 100, 2),
            "rejection_chance": round((1 - approval_prob) * 100, 2)
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/feature_weights', methods=['GET'])
def get_feature_weights():
    """Get feature weights for visualization"""
    try:
        weights_list = []
        for feature, weight in sorted(feature_weights.items(), key=lambda x: x[1], reverse=True):
            weights_list.append({
                "feature": feature,
                "weight": weight,
                "percentage": round(weight * 100, 1)
            })
        
        return jsonify({
            "status": "success",
            "weights": weights_list
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Finance MITRA API is running"
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏦 FINANCE MITRA - STARTING SERVER")
    print("="*60)
    print("\n✓ Model Loaded")
    print("✓ API Ready")
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🌐 Visit: http://0.0.0.0:{port}")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=port)