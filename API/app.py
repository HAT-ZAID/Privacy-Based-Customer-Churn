# FastAPI file for our churn model

# Imports
from pathlib import Path
from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from typing import Annotated, Dict
import pandas as pd  # type: ignore
from contextlib import asynccontextmanager
import joblib  # type: ignore

# General Configurations
## Model Loading
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "Telco_xgb_v1.joblib"
MODEL = joblib.load(MODEL_PATH)                          

## API Configuration
API_TITLE = "Privacy Based Churn Prediction API"
API_DESCRIPTION = "This API predicts customer while maintaining privacy via federated learning."
API_VERSION = "1.0.0"

## Feature and Class Configurations
NUM_CLASSES = 2  # Churn or Not Churn
NUM_FEATURES = 52    # Number of features expected by the model
CLASS_LABELS = {
    0: "Not Churn", 
    1: "Churn"
}

## Feature Names in the same exact order as expected by the model
FEATURE_NAMES: list[str] = [
    "Gender",
    "Age",
    "Under 30",
    "Senior Citizen",
    "Married",
    "Dependents",
    "Number of Dependents",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "Referred a Friend",
    "Number of Referrals",
    "Tenure in Months",
    "Phone Service",
    "Avg Monthly Long Distance Charges",
    "Multiple Lines",
    "Internet Service",
    "Avg Monthly GB Download",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Paperless Billing",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Long Distance Charges",
    "Total Revenue",
    "Satisfaction Score",
    "CLTV",
    "Offer_No Offer",
    "Offer_Offer A",
    "Offer_Offer B",
    "Offer_Offer C",
    "Offer_Offer D",
    "Offer_Offer E",
    "Internet Type_Cable",
    "Internet Type_DSL",
    "Internet Type_Fiber Optic",
    "Internet Type_No Internet",
    "Contract_Month-to-Month",
    "Contract_One Year",
    "Contract_Two Year",
    "Payment Method_Bank Withdrawal",
    "Payment Method_Credit Card",
    "Payment Method_Mailed Check",
]

# response model
class response_model(BaseModel):
    prediction: str = Field(..., description='0: will not churn, 1: will churn')
    confidence: float = Field(...)
    class_probabilities: Dict[str, float] = Field(...)

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "Churn",
                "confidence": 0.95,
                "class_probabilities": {
                    "Not Churn": 0.95,
                    "Churn": 0.05,
                },
            }
        }
    }




# loading model at startup so it doesn't have to be loaded every time a request is made, 
# also yielding the app state to keep the model in memory for future requests. 
# This is done using the lifespan context manager provided by FastAPI.
@asynccontextmanager 
async def lifespan(app: FastAPI):
    print("Please wait while the model is being loaded...")
    app.state.model = joblib.load(MODEL_PATH)
    print("Model Loaded successfully!")
    yield
    print("Shutting down the API....")
    
# API Initialization
app = FastAPI(lifespan=lifespan, title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)


# Defining the request model (pydantic model) for the API endpoint
class ChurnRequest(BaseModel):
    Gender: Annotated[int, Field(...)]
    Age: Annotated[int, Field(...)]
    Under_30: Annotated[int, Field(..., alias="Under 30")]
    Senior_Citizen: Annotated[int, Field(..., alias="Senior Citizen")]
    Married: Annotated[int, Field(...)]
    Dependents: Annotated[int, Field(...)]
    Number_of_Dependents: Annotated[int, Field(..., alias="Number of Dependents")]
    Zip_Code: Annotated[int, Field(..., alias="Zip Code")]
    Latitude: Annotated[float, Field(...)]
    Longitude: Annotated[float, Field(...)]
    Population: Annotated[int, Field(...)]
    Referred_a_Friend: Annotated[int, Field(..., alias="Referred a Friend")]
    Number_of_Referrals: Annotated[int, Field(..., alias="Number of Referrals")]
    Tenure_in_Months: Annotated[int, Field(..., alias="Tenure in Months")]
    Phone_Service: Annotated[int, Field(..., alias="Phone Service")]
    Avg_Monthly_Long_Distance_Charges: Annotated[float, Field(..., alias="Avg Monthly Long Distance Charges")]
    Multiple_Lines: Annotated[int, Field(..., alias="Multiple Lines")]
    Internet_Service: Annotated[int, Field(..., alias="Internet Service")]
    Avg_Monthly_GB_Download: Annotated[int, Field(..., alias="Avg Monthly GB Download")]
    Online_Security: Annotated[int, Field(..., alias="Online Security")]
    Online_Backup: Annotated[int, Field(..., alias="Online Backup")]
    Device_Protection_Plan: Annotated[int, Field(..., alias="Device Protection Plan")]
    Premium_Tech_Support: Annotated[int, Field(..., alias="Premium Tech Support")]
    Streaming_TV: Annotated[int, Field(..., alias="Streaming TV")]
    Streaming_Movies: Annotated[int, Field(..., alias="Streaming Movies")]
    Streaming_Music: Annotated[int, Field(..., alias="Streaming Music")]
    Unlimited_Data: Annotated[int, Field(..., alias="Unlimited Data")]
    Paperless_Billing: Annotated[int, Field(..., alias="Paperless Billing")]
    Monthly_Charge: Annotated[float, Field(..., alias="Monthly Charge")]
    Total_Charges: Annotated[float, Field(..., alias="Total Charges")]
    Total_Refunds: Annotated[float, Field(..., alias="Total Refunds")]
    Total_Extra_Data_Charges: Annotated[int, Field(..., alias="Total Extra Data Charges")]
    Total_Long_Distance_Charges: Annotated[float, Field(..., alias="Total Long Distance Charges")]
    Total_Revenue: Annotated[float, Field(..., alias="Total Revenue")]
    Satisfaction_Score: Annotated[int, Field(..., alias="Satisfaction Score")]
    CLTV: Annotated[int, Field(...)]
    Offer_No_Offer: Annotated[float, Field(..., alias="Offer_No Offer")]
    Offer_Offer_A: Annotated[float, Field(..., alias="Offer_Offer A")]
    Offer_Offer_B: Annotated[float, Field(..., alias="Offer_Offer B")]
    Offer_Offer_C: Annotated[float, Field(..., alias="Offer_Offer C")]
    Offer_Offer_D: Annotated[float, Field(..., alias="Offer_Offer D")]
    Offer_Offer_E: Annotated[float, Field(..., alias="Offer_Offer E")]
    Internet_Type_Cable: Annotated[float, Field(..., alias="Internet Type_Cable")]
    Internet_Type_DSL: Annotated[float, Field(..., alias="Internet Type_DSL")]
    Internet_Type_Fiber_Optic: Annotated[float, Field(..., alias="Internet Type_Fiber Optic")]
    Internet_Type_No_Internet: Annotated[float, Field(..., alias="Internet Type_No Internet")]
    Contract_Month_to_Month: Annotated[float, Field(..., alias="Contract_Month-to-Month")]
    Contract_One_Year: Annotated[float, Field(..., alias="Contract_One Year")]
    Contract_Two_Year: Annotated[float, Field(..., alias="Contract_Two Year")]
    Payment_Method_Bank_Withdrawal: Annotated[float, Field(..., alias="Payment Method_Bank Withdrawal")]
    Payment_Method_Credit_Card: Annotated[float, Field(..., alias="Payment Method_Credit Card")]
    Payment_Method_Mailed_Check: Annotated[float, Field(..., alias="Payment Method_Mailed Check")]
    


@app.get("/")
def root():
    return {"message": "Welcome to the Churn Prediction API",
            "docs" :  "use /docs for API documentation and testing",
            "health_check" : "use /Health to check the health of the API",}

def validate_features(features_dict: dict[str, float]) : # This function needs to fixed or you can say called
    misssing_features = set(FEATURE_NAMES) - set(features_dict.keys())
    if misssing_features:
        raise ValueError (f"Some features are missing: {', '.join(sorted(misssing_features))}") 
     
@app.get("/Health")
def Health_check():

    if app.state.model:
        return {"message": "API is healthy and ready to serve requests."}
    else:
        return {"message": "API is not healthy."}



@app.post("/predict", response_model=response_model)
def predict_churn(request: ChurnRequest):
    pdf = pd.DataFrame([request.model_dump(by_alias=True)])
    pdf = pdf.reindex(columns=FEATURE_NAMES, fill_value=0)

    prediction = int(app.state.model.predict(pdf)[0])
    class_probabilities = app.state.model.predict_proba(pdf)[0]

    output = response_model(
        prediction=CLASS_LABELS[prediction],
        confidence=float(class_probabilities.max()),
        class_probabilities={
            CLASS_LABELS[idx]: float(prob)
            for idx, prob in enumerate(class_probabilities)
        },
    )
    return JSONResponse(content=output.model_dump())


# Debugging notes:
# - `response_model` is returned as a class instead of an instance, so the API would never send a valid prediction payload.
# - `request.dict(by_alias=True)` is deprecated in Pydantic v2; `model_dump(by_alias=True)` should be used.
# - The model input DataFrame needs to be aligned to `FEATURE_NAMES` in the exact expected order.
