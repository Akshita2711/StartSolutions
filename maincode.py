# ======== 1. Import Required Libraries ========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

# Set dark pastel theme
plt.style.use('dark_background')
plt.rcParams.update({
    "axes.facecolor": "#000000",
    "figure.facecolor": "#000000",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#eeeeee",
    "xtick.color": "#dddddd",
    "ytick.color": "#dddddd",
    "text.color": "#ffffff",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "grid.color": "#444444",
    "font.size": 11,
    "legend.edgecolor": "#ffffff",
    "legend.facecolor": "#111111"
})

sns.set_theme(style="darkgrid", palette="pastel")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.decomposition import PCA

# ======== 2. Load Dataset ========
print("\U0001F4E5 Loading dataset...")
df = pd.read_csv('train.csv')
print(f"✅ Dataset loaded with shape: {df.shape}")

# ======== 3. Label Encode Categorical Columns ========
print("🌤 Encoding categorical columns...")
le = LabelEncoder()
cols_to_encode = ['gender', 'Partner', 'Dependents', 'Contract', 'PhoneService',
                  'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'PaperlessBilling', 'PaymentMethod']

for col in cols_to_encode:
    df[col] = le.fit_transform(df[col])

# ======== 4. Handle Missing and Non-Numeric Values ========
print("🧹 Cleaning TotalCharges column...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# ======== 5. Feature and Label Selection ========
feature_cols = cols_to_encode + ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[feature_cols]
y = df['churned']

# ======== 6. Standardize the Features ========
print("📏 Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======== 7. Apply PCA (95% Variance) ========
print("📉 Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"✅ PCA reduced dimensions to {X_pca.shape[1]} components")

# ======== 8. Train-Test Split for PCA Model ========
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=1)

# ======== 9. Train Logistic Regression on PCA Data ========
print("🧠 Training Logistic Regression (PCA)...")
pca_model = LogisticRegression(max_iter=10000)
pca_model.fit(X_train_pca, y_train_pca)
pca_pred = pca_model.predict(X_test_pca)
pca_prob = pca_model.predict_proba(X_test_pca)[:, 1]
pca_auc = roc_auc_score(y_test_pca, pca_prob)
print(f"✅ AUC (PCA model): {pca_auc:.3f}")

# ======== 10. Approximate Feature Importance ========
print("📊 Approximating feature importance from PCA...")
original_space_coef = np.dot(pca_model.coef_, pca.components_)
sorted_idx = np.argsort(original_space_coef[0])

plt.figure(figsize=(12, 8))
colors = sns.color_palette("pastel", len(sorted_idx))
bars = plt.barh(np.array(feature_cols)[sorted_idx], original_space_coef[0][sorted_idx], color=colors)
plt.xlabel("Approximated Feature Importance")
plt.title("🔍 Feature Importance from Logistic Regression (PCA reversed)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ======== 11. ROC Curve for PCA Model ========
fpr_pca, tpr_pca, _ = roc_curve(y_test_pca, pca_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_pca, tpr_pca, label=f'PCA Logistic (AUC = {pca_auc:.3f})', color='#B0E0E6', linewidth=2.5)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('📈 ROC Curve: PCA Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ======== 12. Train Logistic Regression (No PCA) ========
print("🧠 Training Logistic Regression (no PCA)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=10000, C=0.7)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]
log_auc = roc_auc_score(y_test, log_prob)
print(f"✅ AUC (Logistic Regression): {log_auc:.3f}")

# ======== 13. ROC Curve for No PCA Model ========
fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {log_auc:.3f})', color='#FFB6C1', linewidth=2.5)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('📈 ROC Curve: Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ======== 14. SHAP Analysis ========
print("📈 Generating SHAP values for logistic regression...")
explainer = shap.Explainer(log_model, X_train_scaled, feature_names=feature_cols)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)

# ======== 15. Random Forest Classifier ========
print("🌳 Training Random Forest...")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_prob)
print(f"✅ AUC (Random Forest): {rf_auc:.3f}")

# ======== 16. XGBoost Classifier ========
print("⚡ Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False,
                    eval_metric='logloss',
                    learning_rate=0.1,
                    max_depth=6,
                    n_estimators=100,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_prob)
print(f"✅ AUC (XGBoost): {xgb_auc:.3f}")

# ======== 17. Classification Reports ========
print("\n📋 Classification Reports:")
print("\n🔹 Logistic Regression (No PCA):")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("AUC:", log_auc)
print(classification_report(y_test, log_pred))

print("\n🔹 Random Forest:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("AUC:", rf_auc)
print(classification_report(y_test, rf_pred))

print("\n🔹 XGBoost:")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("AUC:", xgb_auc)
print(classification_report(y_test, xgb_pred))

# ======== 18. ROC Curve Comparison ========
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

plt.figure(figsize=(10, 7))
plt.plot(fpr_log, tpr_log, label=f'LogReg (AUC = {log_auc:.3f})', color='#FFB6C1', linewidth=2.5)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})', color='#ADD8E6', linewidth=2.5)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.3f})', color='#98FB98', linewidth=2.5)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('📋 ROC Curves: Model Comparison')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ======== 19. Load Test Data for Final Prediction ========
print("\n📥 Loading test.csv for final prediction...")
test_df = pd.read_csv('test.csv')

for col in cols_to_encode:
    test_df[col] = le.fit_transform(test_df[col])

test_df['TotalCharges'] = pd.to_numeric(test_df['TotalCharges'], errors='coerce')
test_df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

X_test_final = test_df[feature_cols]
X_test_final_scaled = scaler.transform(X_test_final)

# ======== 20. Make Predictions on Test Set ========
print("🔮 Predicting churn on test set using Logistic Regression...")
test_predictions = log_model.predict(X_test_final_scaled)
test_probabilities = log_model.predict_proba(X_test_final_scaled)[:, 1]

# ======== 21. Save Predictions ========
output = pd.DataFrame({
    'customerID': test_df['customerID'] if 'customerID' in test_df.columns else np.arange(len(test_df)),
    'churn_prediction': test_predictions,
    'churn_probability': test_probabilities
})

output.to_csv('predictions.csv', index=False)
print("✅ Predictions saved to predictions.csv using Logistic Regression")


import os
os.environ["STREAMLIT_SUPPRESS_RUN_CONTEXT_WARNING"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set pastel seaborn style
sns.set_style("darkgrid")
plt.style.use('seaborn-v0_8-dark-palette')

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title with icon
st.markdown(
    "<h1 style='text-align: center; color: #7a5195;'>📊 Customer Churn Prediction Dashboard</h1><br>",
    unsafe_allow_html=True
)
# Theme Toggle in Sidebar
theme_mode = st.sidebar.radio("🌓 Select Theme", ["Light", "Dark"])

# Set background and text colors based on theme
if theme_mode == "Dark":
    st.markdown(
        """
        <style>
            body {
                background-color: #1e1e1e;
                color: white;
            }
            .stApp {
                background-color: #1e1e1e;
            }
            .css-1d391kg, .css-ffhzg2, .css-1v3fvcr {
                background-color: #1e1e1e !important;
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
            body {
                background-color: white;
                color: black;
            }
            .stApp {
                background-color: white;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar for file upload
st.sidebar.header("📤 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

# Load model and tools
@st.cache_resource
def load_model_and_tools():
    df = pd.read_csv("train.csv")
    le = LabelEncoder()
    cols_to_encode = ['gender', 'Partner', 'Dependents', 'Contract', 'PhoneService',
                      'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                      'PaperlessBilling', 'PaymentMethod']
    for col in cols_to_encode:
        df[col] = le.fit_transform(df[col])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    feature_cols = cols_to_encode + ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = df[feature_cols]
    y = df['churned']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=10000, C=0.7)
    model.fit(X_scaled, y)

    return model, le, scaler, feature_cols

model, le, scaler, feature_cols = load_model_and_tools()

# Prediction logic
def preprocess_and_predict(data):
    for col in feature_cols:
        if col in data.columns and data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    X = data[feature_cols]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    return predictions, probabilities

# New: Manual user comparison interface
with st.expander("🔍 Compare Two Customers Manually"):
    st.markdown("Fill the details for both users to compare their churn risk and suggested engagement strategy.")
    user_inputs = []
    for i in range(2):
        st.markdown(f"*User {i+1}*")
        inputs = {}
        for col in feature_cols:
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                inputs[col] = st.number_input(f"{col} (User {i+1})", key=f"{col}_{i}")
            else:
                options = ['Yes', 'No'] if col not in ['gender', 'Contract', 'InternetService', 'PaymentMethod'] else []
                if col == 'gender': options = ['Male', 'Female']
                if col == 'Contract': options = ['Month-to-month', 'One year', 'Two year']
                if col == 'InternetService': options = ['DSL', 'Fiber optic', 'No']
                if col == 'PaymentMethod': options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
                inputs[col] = st.selectbox(f"{col} (User {i+1})", options, key=f"{col}_{i}")
        user_inputs.append(inputs)

    if st.button("🔍 Compare Users"):
        df_users = pd.DataFrame(user_inputs)
        for col in feature_cols:
            if df_users[col].dtype == 'object':
                df_users[col] = le.fit_transform(df_users[col])
        df_users_scaled = scaler.transform(df_users[feature_cols])
        churn_probs = model.predict_proba(df_users_scaled)[:, 1]
        st.success(f"User 1 churn probability: {churn_probs[0]:.2f}")
        st.success(f"User 2 churn probability: {churn_probs[1]:.2f}")

        if churn_probs[0] > churn_probs[1]:
            attention = "User 1 needs more attention."
        elif churn_probs[1] > churn_probs[0]:
            attention = "User 2 needs more attention."
        else:
            attention = "Both users have equal churn risk."

        st.markdown(f"### 🔔 {attention}")

        def action(prob):
            if prob > 0.9:
                return "📝 Send feedback form"
            elif prob > 0.75:
                return "💸 Offer cashback or discount"
            elif prob > 0.5:
                return "🎁 Provide loyalty reward"
            else:
                return "✅ Maintain current engagement"

        st.markdown(f"*Suggested Plan for User 1:* {action(churn_probs[0])}")
        st.markdown(f"*Suggested Plan for User 2:* {action(churn_probs[1])}")

# Main UI
if uploaded_file:
    st.subheader("🔍 Uploaded Data Preview")
    test_df = pd.read_csv(uploaded_file)
    st.dataframe(test_df.head(), use_container_width=True)

    if st.button("📈 Predict Churn"):
        preds, probs = preprocess_and_predict(test_df)
        output_df = test_df.copy()
        output_df["Churn Prediction"] = preds
        output_df["Churn Probability"] = probs

        st.subheader("✅ Prediction Results")
        result_cols = ["customerID", "Churn Prediction", "Churn Probability"] if "customerID" in test_df.columns else ["Churn Prediction", "Churn Probability"]
        st.dataframe(output_df[result_cols], use_container_width=True)

        st.subheader("🎯 Smart Re-engagement Suggestions")
        required_cols = ["Churn Probability", "tenure", "MonthlyCharges"]

        if all(col in output_df.columns for col in required_cols):
            risky_df = output_df.sort_values("Churn Probability", ascending=False).head(10).copy()
            def suggest_action(row):
                if row["tenure"] < 1:
                    return "💸 Offer cashback or discount"
                elif row["tenure"] < 3:
                    return "🎁 Give 7-day premium trial"
                elif row["Churn Probability"] > 0.9:
                    return "📝 Send feedback form"
                else:
                    return "📞 Schedule a call or send engagement email"
            risky_df["Suggested Action"] = risky_df.apply(suggest_action, axis=1)
            display_cols = ["customerID", "tenure", "MonthlyCharges", "Churn Probability", "Suggested Action"] \
                if "customerID" in risky_df.columns else ["tenure", "MonthlyCharges", "Churn Probability", "Suggested Action"]
            st.dataframe(risky_df[display_cols], use_container_width=True)

        else:
            st.warning("⚠ Smart suggestions not available — missing required columns like 'tenure' or 'MonthlyCharges' in uploaded data.")

        # Churn Probability Distribution
        st.subheader("🌈 Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(probs, kde=True, bins=30, color="#FFA07A", edgecolor="black", ax=ax)
        ax.set_title("Churn Probability Histogram", fontsize=14, color='#7a5195')
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.tick_params(colors='gray')
        st.pyplot(fig)

        # ROC Curve
        if "churned" in test_df.columns:
            fpr, tpr, _ = roc_curve(test_df["churned"], probs)
            auc_score = roc_auc_score(test_df["churned"], probs)
            st.subheader(f"📉 ROC Curve (AUC = {auc_score:.3f})")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(fpr, tpr, label="ROC Curve", color="#66c2a5", linewidth=2)
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate", fontsize=12)
            ax2.set_ylabel("True Positive Rate", fontsize=12)
            ax2.set_title("Receiver Operating Characteristic", fontsize=14, color='#7a5195')
            ax2.legend()
            ax2.tick_params(colors='gray')
            st.pyplot(fig2)

else:
    st.info("📂 Please upload a test CSV file from the sidebar to start.")

    st.markdown("---")
    st.subheader("👥 Compare Two Customers Manually")

    with st.form("compare_customers_form"):
        st.markdown("### 🧍‍♂ Customer 1")
        c1_data = {}
        for col in feature_cols:
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                c1_data[col] = st.number_input(f"Customer 1 - {col}", min_value=0.0)
            else:
                c1_data[col] = st.selectbox(f"Customer 1 - {col}", options=['Yes', 'No', 'Male', 'Female', 'Month-to-month', 'One year', 'Two year',
                                                                              'DSL', 'Fiber optic', 'No', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                                            key=f"c1_{col}")

        st.markdown("### 🧍‍♀ Customer 2")
        c2_data = {}
        for col in feature_cols:
            if col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                c2_data[col] = st.number_input(f"Customer 2 - {col}", min_value=0.0, key=f"c2_{col}")
            else:
                c2_data[col] = st.selectbox(f"Customer 2 - {col}", options=['Yes', 'No', 'Male', 'Female', 'Month-to-month', 'One year', 'Two year',
                                                                              'DSL', 'Fiber optic', 'No', 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                                            key=f"c2_{col}")

        submitted = st.form_submit_button("🔍 Compare and Recommend")

    if submitted:
        df_compare = pd.DataFrame([c1_data, c2_data])
        preds, probs = preprocess_and_predict(df_compare)

        df_compare["Churn Probability"] = probs
        df_compare["Suggested Action"] = df_compare.apply(lambda row: (
            "💸 Offer cashback" if row["tenure"] < 1 else
            "🎁 Free trial" if row["tenure"] < 3 else
            "📝 Collect feedback" if row["Churn Probability"] > 0.9 else
            "📞 Send engagement email"
        ), axis=1)

        st.subheader("🧾 Results")
        st.dataframe(df_compare[["Churn Probability", "Suggested Action"]])

        if probs[0] > probs[1]:
            st.markdown(f"🎯 *Customer 1 has a higher churn risk ({probs[0]:.2f}) — Prioritize them.*")
        elif probs[1] > probs[0]:
            st.markdown(f"🎯 *Customer 2 has a higher churn risk ({probs[1]:.2f}) — Prioritize them.*")
        else:
            st.markdown("✅ Both customers have equal churn risk.")