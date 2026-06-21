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

try:
    model = joblib.load("loan_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_weights = joblib.load("feature_weights.pkl")
    feature_names = joblib.load("feature_names.pkl")
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
    print(f"\n🌐 Visit: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='localhost', port=5000)

# ==========================
# TITLE
# ==========================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<h1 style="
text-align:center;
color:#38bdf8;
font-size:55px;
font-weight:800;
text-shadow: 0px 0px 20px rgba(56,189,248,0.8);
">
🏦 Finance MITRA
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="
color:#f8fafc;
font-weight:700;
margin-bottom:20px;
">
Loan Approval Prediction System
</h2>
""", unsafe_allow_html=True)

# ==========================
# USER INPUTS
# ==========================

col_left, col_right = st.columns(2)

with col_left:

    married = st.selectbox(
        "Married",
        ["No","Yes"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=100
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=120,
        step=10
    )

    credit_history = st.selectbox(
        "Credit History",
        [0,1]
    )

with col_right:

    education = st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0,
        step=100
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0,
        value=360,
        step=12
    )

# ==========================
# FEATURE ENGINEERING
# ==========================

total_income = applicant_income + coapplicant_income

# ==========================
# DASHBOARD CARDS
# ==========================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💰 Income", f"₹{applicant_income:,.0f}")

with col2:
    st.metric("💵 Loan", f"₹{loan_amount:,.0f}")

with col3:
    st.metric("📈 Total Income", f"₹{total_income:,.0f}")

# ==========================
# PREDICTION BUTTON
# ==========================

if st.button("Predict Loan Approval"):

    married_val = 1 if married == "Yes" else 0
    education_val = 0 if education == "Graduate" else 1

    features = np.array([[
        married_val,
        education_val,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        total_income
    ]])

    # Prediction
    prediction = model.predict(features)

    # Probability
    probability = model.predict_proba(features)
    approval_prob = probability[0][1]

    st.markdown("""
<h2 style="
color:#38bdf8;
text-align:center;
">
📊 AI Assessment
</h2>
""", unsafe_allow_html=True)

    st.write(
        f"### Approval Probability: {approval_prob*100:.2f}%"
    )

    st.progress(float(approval_prob))

    # ==========================
    # ELIGIBILITY SCORE
    # ==========================

    eligibility_score = int(approval_prob * 100)

    st.metric(
        "🎯 Eligibility Score",
        f"{eligibility_score}/100"
    )

    # ==========================
    # PIE CHART
    # ==========================

    labels = [
        "Approval Chance",
        "Rejection Chance"
    ]

    sizes = [
        approval_prob * 100,
        (1 - approval_prob) * 100
    ]
    fig, ax = plt.subplots(
        figsize=(3.8,3.8)
    )

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        textprops={"color":"white"}
    )

    fig.patch.set_facecolor("#1e293b")

    ax.axis("equal")

    st.subheader("🥧 Approval Distribution")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("🥧 Approval Distribution")
        st.pyplot(fig)

    # ==========================
    # RISK LEVEL
    # ==========================

    if approval_prob >= 0.80:
        st.success("🟢 Risk Level: Low Risk")

    elif approval_prob >= 0.60:
        st.warning("🟡 Risk Level: Moderate Risk")

    else:
        st.error("🔴 Risk Level: High Risk")

    # ==========================
    # DECISION FACTORS
    # ==========================

    st.subheader("📋 Decision Factors")

    reasons = []

    if credit_history == 1:
        reasons.append("✅ Strong Credit History")

    if total_income > 5000:
        reasons.append("✅ Good Combined Income")

    if loan_amount < 200:
        reasons.append("✅ Manageable Loan Amount")

    if education == "Graduate":
        reasons.append("✅ Graduate Applicant")

    if len(reasons) == 0:
        reasons.append("⚠ Few positive indicators found")

    for reason in reasons:
        st.write(reason)

    # ==========================
    # FINAL DECISION
    # ==========================

    if prediction[0] == 1:

        st.success("✅ Loan Approved")

    else:

        st.error("❌ Loan Rejected")

        st.subheader("💡 Improvement Suggestions")

        if credit_history == 0:
            st.write("• Improve Credit History")

        if total_income < 5000:
            st.write("• Increase Income")

        if loan_amount > 250:
            st.write("• Reduce Loan Amount")

    # ==========================
    # FEATURE IMPORTANCE
    # ==========================

    st.subheader("📊 Feature Importance")

    feature_names = [
        "Married",
        "Education",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "LoanTerm",
        "CreditHistory",
        "TotalIncome"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax2.set_facecolor("#1e293b")
    fig2.patch.set_facecolor("#1e293b")

    ax2.tick_params(colors="white")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")

    ax2.set_title(
    "Feature Importance",
    color="white"
)

    ax2.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    with chart_col2:
        st.subheader("📊 Feature Importance")
        st.pyplot(fig2)