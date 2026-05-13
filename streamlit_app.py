import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📞 Customer Churn Prediction")
st.markdown("### Predict whether a customer will leave the company")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/churn_model.pkl')

model = load_model()

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, max_value=150.0, value=65.0)

with col2:
    contract_type = st.selectbox("Contract Type", 
                               options=[0,1,2],
                               format_func=lambda x: "Month-to-Month" if x==0 else "One Year" if x==1 else "Two Year")
    
    internet_service = st.selectbox("Internet Service", 
                                  options=[0,1,2],
                                  format_func=lambda x: "No" if x==0 else "Fiber Optic" if x==1 else "DSL")
    
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

# Prediction Button
if st.button("🔮 Predict Churn", type="primary"):
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'monthly_charges': [monthly_charges],
        'contract_type': [contract_type],
        'internet_service': [internet_service],
        'payment_method': [1],           # default value
        'senior_citizen': [senior_citizen]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk - Customer may CHURN** ({probability*100:.1f}% probability)")
    else:
        st.success(f"✅ **Low Risk - Customer likely to STAY** ({(1-probability)*100:.1f}% probability)")

st.caption("Customer Churn Prediction Model | Built with Logistic Regression")