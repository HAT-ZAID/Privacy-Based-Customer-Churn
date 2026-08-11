
import time
import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"
gender_mapping = {0: "Male", 1: "Female"}
st.title(' ')


st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CSS ----------
st.markdown(
    """
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(51, 192, 173, 0.10), transparent 25%),
            radial-gradient(circle at bottom right, rgba(51, 192, 173, 0.07), transparent 22%),
            #0E1514;
    }

    /* Main spacing */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    /* Hide default clutter */
    #MainMenu, footer, header {
        display: none;
    }

    /* Hero card */
    .hero {
        background: linear-gradient(135deg, rgba(23, 33, 31, 0.92), rgba(14, 21, 20, 0.98));
        border: 1px solid rgba(51, 192, 173, 0.18);
        border-radius: 28px;
        padding: 28px 28px 22px 28px;
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(16px);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        line-height: 1.05;
        color: #f3f8f0;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin-top: 0.6rem;
        margin-bottom: 0;
        color: rgba(243, 248, 240, 0.78);
        font-size: 1rem;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }

    .chip {
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(51, 192, 173, 0.22);
        background: rgba(23, 33, 31, 0.8);
        color: #f3f8f0;
        font-size: 0.85rem;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(23, 33, 31, 0.78);
        border: 1px solid rgba(51, 192, 173, 0.16);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(14px);
    }

    .section-title {
        color: #f3f8f0;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .section-subtitle {
        color: rgba(243, 248, 240, 0.72);
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #33c0ad, #29a895);
        color: #0E1514;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 1.2rem;
        font-weight: 800;
        box-shadow: 0 10px 28px rgba(51, 192, 173, 0.25);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 34px rgba(51, 192, 173, 0.32);
    }

    /* Inputs */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div,
    .stSlider,
    .stRadio div {
        font-family: Manrope, sans-serif !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17211F, #0E1514);
        border-right: 1px solid rgba(51, 192, 173, 0.12);
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, rgba(23, 33, 31, 0.95), rgba(14, 21, 20, 0.98));
        border: 1px solid rgba(51, 192, 173, 0.18);
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.3);
    }

    .big-status {
        font-size: 2rem;
        font-weight: 900;
        color: #f3f8f0;
        margin: 0;
    }

    .muted {
        color: rgba(243, 248, 240, 0.72);
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HERO ----------
st.markdown(
    """
    <div class="hero">
        <h1>Privacy-Based Customer Churn Prediction</h1>
        <p>Model is based on telco churn dataset by IBM.</p>


    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2 = st.columns([0.2, 1])

with col1:
    st.image("https://media.tenor.com/SLteuYShPF4AAAAi/skeleton-flowers.gif", width=200)



st.write("Want to Predict if your customer will Churn or not ? Please fill in the customer data below:")


submitted = False
with st.expander("Customer Data Form"):
    with st.form("customer_form"):
        customer_data = {
            "Gender": st.radio("Select Gender -> 0: Male, 1: Female", options=list(gender_mapping.keys()), index=0),
            "Age": st.slider("Age", min_value=12, max_value=90, value=35, step=1),
            "Under 30": st.radio("Under 30 -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Senior Citizen": st.radio("Senior Citizen -> 0: No, 1: Yes", options=[0, 1]),
            "Married": st.radio("Married -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Dependents": st.radio("Dependents -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Number of Dependents": st.slider("Number of Dependents", min_value=0, max_value=5, value=0, step=1),
            "Zip Code": st.slider("Zip Code", min_value=10000, max_value=99999, value=50000, step=1),
            "Latitude": st.slider("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.01),
            "Longitude": st.slider("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.01),
            "Population": st.slider("Population", min_value=0, max_value=1000000, value=50000, step=1),
            "Referred a Friend": st.radio("Referred a Friend -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Number of Referrals": st.slider("Number of Referrals", min_value=0, max_value=10, value=0, step=1),
            "Tenure in Months": st.slider("Tenure in Months", min_value=0, max_value=72, value=12, step=1),
            "Phone Service": st.radio("Phone Service -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Avg Monthly Long Distance Charges": st.slider("Avg Monthly Long Distance Charges", min_value=0.0, max_value=200.0, value=50.0, step=0.01),
            "Multiple Lines": st.radio("Multiple Lines -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Internet Service": st.radio("Internet Service -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Avg Monthly GB Download": st.slider("Avg Monthly GB Download", min_value=0, max_value=1000, value=50, step=1),
            "Online Security": st.radio("Online Security -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Online Backup": st.radio("Online Backup -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Device Protection Plan": st.radio("Device Protection Plan -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Premium Tech Support": st.radio("Premium Tech Support -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Streaming TV": st.radio("Streaming TV -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Streaming Movies": st.radio("Streaming Movies -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Streaming Music": st.radio("Streaming Music -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Unlimited Data": st.radio("Unlimited Data -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Paperless Billing": st.radio("Paperless Billing -> 0: No, 1: Yes", options=[0, 1], index=0),
            "Monthly Charge": st.slider("Monthly Charge", min_value=0.0, max_value=200.0, value=70.0, step=0.01),
            "Total Charges": st.slider("Total Charges", min_value=0.0, max_value=5000.0, value=800.0, step=0.01),
            "Total Refunds": st.slider("Total Refunds", min_value=0.0, max_value=500.0, value=10.0, step=0.01),
            "Total Extra Data Charges": st.slider("Total Extra Data Charges", min_value=0, max_value=500, value=0, step=1),
            "Total Long Distance Charges": st.slider("Total Long Distance Charges", min_value=0.0, max_value=1000.0, value=50.0, step=0.01),
            "Total Revenue": st.slider("Total Revenue", min_value=0.0, max_value=10000.0, value=500.0, step=0.01),
            "Satisfaction Score": st.slider("Satisfaction Score", min_value=1, max_value=5, value=3, step=1),
            "CLTV": st.slider("CLTV", min_value=0, max_value=1200, value=300, step=1),
            "Offer_No Offer": st.radio("Offer_No Offer -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Offer_Offer A": st.radio("Offer_Offer A -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Offer_Offer B": st.radio("Offer_Offer B -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Offer_Offer C": st.radio("Offer_Offer C -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Offer_Offer D": st.radio("Offer_Offer D -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Offer_Offer E": st.radio("Offer_Offer E -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Internet Type_Cable": st.radio("Internet Type_Cable -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Internet Type_DSL": st.radio("Internet Type_DSL -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Internet Type_Fiber Optic": st.radio("Internet Type_Fiber Optic -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Internet Type_No Internet": st.radio("Internet Type_No Internet -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Contract_Month-to-Month": st.radio("Contract_Month-to-Month -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Contract_One Year": st.radio("Contract_One Year -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Contract_Two Year": st.radio("Contract_Two Year -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Payment Method_Bank Withdrawal": st.radio("Payment Method_Bank Withdrawal -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Payment Method_Credit Card": st.radio("Payment Method_Credit Card -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
            "Payment Method_Mailed Check": st.radio("Payment Method_Mailed Check -> 0: No, 1: Yes", options=[0.0, 1.0], index=0),
        }

        submitted = st.form_submit_button("Predict Churn")
        
with st.sidebar:
    with st.expander("About this App"):
        st.write("This app predicts customer churn based on various features.,Fill in the customer data and click 'Predict Churn' to see    the prediction.")

with col2: 
    status = st.empty()
    with st.spinner("Are you worried about your customer churning ? Don't worry, we got you covered. We are predicting the churn probability for you..."):
        while not submitted :
            status.markdown("##  Will Churn ; )")
            time.sleep(1.5)

            status.markdown("##  Will Not Churn : )")
            time.sleep(1.5)
            
if submitted:
    st.balloons()
    response = requests.post(API_URL, json=customer_data) # type: ignore
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress.progress(i + 1)
    if response.ok:
        
        st.success(f"Prediction: {response.json()}")
    else:
        st.error("Prediction request failed.")


# Disabled code from css hero section
        # <div class="chip-row">
            # <span class="chip">Fast prediction</span>
            # <span class="chip">Privacy-focused</span>
            # <span class="chip">Modern UI</span>
            # <span class="chip">ML-powered</span>
        # </div>