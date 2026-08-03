
import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"
gender_mapping = {"Male": 0, "Female": 1}
st.title('Privacy Based Customer Churn Prediction')
st.markdown("### Enter customer data to predict churn probability")

# Input fields for customer data
with st.form("customer_form"):
    customer_data = {
        "Gender": st.selectbox("Select Gender 0: Male, 1: Female", options=list(gender_mapping.keys()), index=0),
        "Age": st.slider("Age", min_value=12, max_value=90, value=35, step=1),
        "Under 30": st.slider("Under 30", min_value=0, max_value=1, value=0, step=1),
        "Senior Citizen": st.slider("Senior Citizen", min_value=0, max_value=1, value=0, step=1),
        "Married": st.slider("Married", min_value=0, max_value=1, value=0, step=1),
        "Dependents": st.slider("Dependents", min_value=0, max_value=1, value=0, step=1),
        "Number of Dependents": st.slider("Number of Dependents", min_value=0, max_value=5, value=0, step=1),
        "Zip Code": st.slider("Zip Code", min_value=10000, max_value=99999, value=50000, step=1),
        "Latitude": st.slider("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01),
        "Longitude": st.slider("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01),
        "Population": st.slider("Population", min_value=0, max_value=1000000, value=50000, step=1),
        "Referred a Friend": st.slider("Referred a Friend", min_value=0, max_value=1, value=0, step=1),
        "Number of Referrals": st.slider("Number of Referrals", min_value=0, max_value=10, value=0, step=1),
        "Tenure in Months": st.slider("Tenure in Months", min_value=0, max_value=72, value=12, step=1),
        "Phone Service": st.slider("Phone Service", min_value=0, max_value=1, value=0, step=1),
        "Avg Monthly Long Distance Charges": st.slider("Avg Monthly Long Distance Charges", min_value=0.0, max_value=200.0, value=50.0, step=0.01),
        "Multiple Lines": st.slider("Multiple Lines", min_value=0, max_value=1, value=0, step=1),
        "Internet Service": st.slider("Internet Service", min_value=0, max_value=3, value=0, step=1),
        "Avg Monthly GB Download": st.slider("Avg Monthly GB Download", min_value=0, max_value=1000, value=50, step=1),
        "Online Security": st.slider("Online Security", min_value=0, max_value=1, value=0, step=1),
        "Online Backup": st.slider("Online Backup", min_value=0, max_value=1, value=0, step=1),
        "Device Protection Plan": st.slider("Device Protection Plan", min_value=0, max_value=1, value=0, step=1),
        "Premium Tech Support": st.slider("Premium Tech Support", min_value=0, max_value=1, value=0, step=1),
        "Streaming TV": st.slider("Streaming TV", min_value=0, max_value=1, value=0, step=1),
        "Streaming Movies": st.slider("Streaming Movies", min_value=0, max_value=1, value=0, step=1),
        "Streaming Music": st.slider("Streaming Music", min_value=0, max_value=1, value=0, step=1),
        "Unlimited Data": st.slider("Unlimited Data", min_value=0, max_value=1, value=0, step=1),
        "Paperless Billing": st.slider("Paperless Billing", min_value=0, max_value=1, value=0, step=1),
        "Monthly Charge": st.slider("Monthly Charge", min_value=0.0, max_value=200.0, value=70.0, step=0.01),
        "Total Charges": st.slider("Total Charges", min_value=0.0, max_value=5000.0, value=800.0, step=0.01),
        "Total Refunds": st.slider("Total Refunds", min_value=0.0, max_value=500.0, value=10.0, step=0.01),
        "Total Extra Data Charges": st.slider("Total Extra Data Charges", min_value=0, max_value=500, value=0, step=1),
        "Total Long Distance Charges": st.slider("Total Long Distance Charges", min_value=0.0, max_value=1000.0, value=50.0, step=0.01),
        "Total Revenue": st.slider("Total Revenue", min_value=0.0, max_value=10000.0, value=500.0, step=0.01),
        "Satisfaction Score": st.slider("Satisfaction Score", min_value=1, max_value=5, value=3, step=1),
        "CLTV": st.slider("CLTV", min_value=0, max_value=1200, value=300, step=1),
        "Offer_No Offer": st.slider("Offer_No Offer", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Offer_Offer A": st.slider("Offer_Offer A", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Offer_Offer B": st.slider("Offer_Offer B", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Offer_Offer C": st.slider("Offer_Offer C", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Offer_Offer D": st.slider("Offer_Offer D", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Offer_Offer E": st.slider("Offer_Offer E", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Internet Type_Cable": st.slider("Internet Type_Cable", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Internet Type_DSL": st.slider("Internet Type_DSL", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Internet Type_Fiber Optic": st.slider("Internet Type_Fiber Optic", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Internet Type_No Internet": st.slider("Internet Type_No Internet", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Contract_Month-to-Month": st.slider("Contract_Month-to-Month", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Contract_One Year": st.slider("Contract_One Year", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Contract_Two Year": st.slider("Contract_Two Year", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Payment Method_Bank Withdrawal": st.slider("Payment Method_Bank Withdrawal", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Payment Method_Credit Card": st.slider("Payment Method_Credit Card", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
        "Payment Method_Mailed Check": st.slider("Payment Method_Mailed Check", min_value=0.0, max_value=1.0, value=0.0, step=0.01),
    }

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    response = requests.post(API_URL, json=customer_data)
    if response.ok:
        st.success(f"Prediction: {response.json()}")
    else:
        st.error("Prediction request failed.")
