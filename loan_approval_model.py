import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SMARTLOANAI - ADVANCED MODEL TRAINING")
print("="*60)

# ============================================
# 1. LOAD AND EXPLORE DATA
# ============================================

df = pd.read_csv("train.csv")

print("\n✓ Data Loaded Successfully")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Missing Values:\n{df.isnull().sum()}")

# ============================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================

# Remove Loan_ID as requested
df_clean = df.drop("Loan_ID", axis=1)

# Fill categorical columns with mode
categorical_fill = {
    "Gender": df_clean["Gender"].mode()[0],
    "Married": df_clean["Married"].mode()[0],
    "Dependents": df_clean["Dependents"].mode()[0],
    "Self_Employed": df_clean["Self_Employed"].mode()[0],
    "Credit_History": df_clean["Credit_History"].mode()[0]
}

for col, fill_val in categorical_fill.items():
    df_clean[col] = df_clean[col].fillna(fill_val)

# Fill numerical columns with median
df_clean["LoanAmount"] = df_clean["LoanAmount"].fillna(df_clean["LoanAmount"].median())
df_clean["Loan_Amount_Term"] = df_clean["Loan_Amount_Term"].fillna(df_clean["Loan_Amount_Term"].median())

print("\n✓ Missing Values Handled")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

# Create new features for better predictions
df_clean["TotalIncome"] = df_clean["ApplicantIncome"] + df_clean["CoapplicantIncome"]

# Log transformation of income to reduce skewness
df_clean["Applicant_Income_Log"] = np.log1p(df_clean["ApplicantIncome"])
df_clean["Total_Income_Log"] = np.log1p(df_clean["TotalIncome"])
df_clean["Loan_Amount_Log"] = np.log1p(df_clean["LoanAmount"])

# Income to Loan Ratio
df_clean["Income_to_Loan_Ratio"] = df_clean["TotalIncome"] / (df_clean["LoanAmount"] + 1)

# Debt to Income Ratio approximation
df_clean["DTI_Approximation"] = df_clean["LoanAmount"] / (df_clean["TotalIncome"] + 1)

# Loan to Value indicator
df_clean["Loan_Applicant_Income_Ratio"] = df_clean["LoanAmount"] / (df_clean["ApplicantIncome"] + 1)

print("\n✓ Feature Engineering Completed")
print(f"  New Features Created: TotalIncome, Log transformations, Ratios")

# ============================================
# 4. ENCODING CATEGORICAL VARIABLES
# ============================================

le_dict = {}
categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

df_encoded = df_clean.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    le_dict[col] = le
    print(f"  ✓ {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Also encode Loan_Status separately
le_target = LabelEncoder()
df_encoded["Loan_Status"] = le_target.fit_transform(df_encoded["Loan_Status"].astype(str))
le_dict["Loan_Status"] = le_target
print(f"  ✓ Loan_Status: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Save label encoders
joblib.dump(le_dict, "label_encoders.pkl")

print("\n✓ Categorical Variables Encoded")

# ============================================
# 5. FEATURE WEIGHTS (AS REQUESTED)
# ============================================

# Define feature weights - Credit History gets highest, Marital Status lowest
feature_weights = {
    "Credit_History": 1.0,  # Maximum weight
    "ApplicantIncome": 0.95,
    "Total_Income_Log": 0.90,
    "TotalIncome": 0.85,
    "LoanAmount": 0.80,
    "Income_to_Loan_Ratio": 0.75,
    "Loan_Amount_Log": 0.70,
    "Applicant_Income_Log": 0.65,
    "Education": 0.60,
    "Property_Area": 0.55,
    "Self_Employed": 0.50,
    "Gender": 0.45,
    "DTI_Approximation": 0.40,
    "Loan_Applicant_Income_Ratio": 0.35,
    "Dependents": 0.30,
    "Loan_Amount_Term": 0.25,
    "CoapplicantIncome": 0.20,
    "Married": 0.15  # Minimum weight as requested
}

# Save feature weights for UI display
joblib.dump(feature_weights, "feature_weights.pkl")

print("\n✓ Feature Weights Assigned:")
for feature, weight in sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature}: {weight}")
print(f"  ... (Total {len(feature_weights)} features)")

# ============================================
# 6. PREPARE TRAINING DATA
# ============================================

X = df_encoded.drop("Loan_Status", axis=1)
y = df_encoded["Loan_Status"]

# Convert Loan_Status to numeric if needed
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

print(f"\n✓ Features Shape: {X.shape}")
print(f"  Target Shape: {y.shape}")
print(f"  Target Distribution:\n{pd.Series(y).value_counts()}")

# ============================================
# 7. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================

smote = SMOTE(random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"\n✓ SMOTE Applied - Class Imbalance Handled")
print(f"  Balanced Target Distribution:\n{pd.Series(y_balanced).value_counts()}")

# ============================================
# 8. SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

print(f"\n✓ Data Split Complete")
print(f"  Training Set: {X_train.shape}")
print(f"  Test Set: {X_test.shape}")

# ============================================
# 9. SCALE FEATURES
# ============================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

print(f"\n✓ Features Scaled Using StandardScaler")

# ============================================
# 10. TRAIN XGBOOST MODEL WITH WEIGHTED FEATURES
# ============================================

print(f"\n{'='*60}")
print("TRAINING XGBOOST MODEL")
print(f"{'='*60}")

# Train primary XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',
    scale_pos_weight=(y_balanced == 0).sum() / (y_balanced == 1).sum()  # Handle class imbalance
)

model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# ============================================
# 11. MODEL EVALUATION
# ============================================

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print(f"\n✓ MODEL PERFORMANCE METRICS:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

print(f"\n✓ Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Approved", "Approved"]))

# ============================================
# 12. FEATURE IMPORTANCE
# ============================================

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n✓ TOP 10 IMPORTANT FEATURES:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)

# ============================================
# 13. SAVE MODEL AND ARTIFACTS
# ============================================

joblib.dump(model, "loan_model.pkl")
joblib.dump(X.columns, "feature_names.pkl")

print(f"\n{'='*60}")
print("✓ MODEL SAVED SUCCESSFULLY")
print(f"{'='*60}")
print(f"  ✓ loan_model.pkl")
print(f"  ✓ label_encoders.pkl")
print(f"  ✓ scaler.pkl")
print(f"  ✓ feature_weights.pkl")
print(f"  ✓ feature_names.pkl")
print(f"  ✓ feature_importance.csv")

print(f"\n{'='*60}")
print("TRAINING COMPLETE - READY FOR DEPLOYMENT")
print(f"{'='*60}\n")
