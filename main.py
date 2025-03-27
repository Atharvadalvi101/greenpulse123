from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

# Load the trained model and encoders
model = joblib.load("irrigation_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Initialize FastAPI app
app = FastAPI(title="Irrigation Prediction API", description="API for predicting irrigation requirements based on sensor and crop data.")

# Define valid categories
valid_crops = list(label_encoders["Crop"].classes_)
valid_soils = list(label_encoders["soil_type"].classes_)
valid_stages = list(label_encoders["Seedling Stage"].classes_)

# Input validation model using Pydantic
class IrrigationInput(BaseModel):
    crop: str
    soil_type: str
    seedling_stage: str
    soil_moisture: float
    temperature: float
    humidity: float

@app.post("/predict")
def predict_irrigation(data: IrrigationInput):
    """Predict whether irrigation is required based on input data."""
    
    # Convert inputs to lowercase
    crop = data.crop.lower()
    soil = data.soil_type.lower()
    stage = data.seedling_stage.lower()

    # Validate input values
    if crop not in valid_crops or soil not in valid_soils or stage not in valid_stages:
        raise HTTPException(status_code=400, detail="Invalid crop, soil type, or seedling stage")

    # Encode categorical inputs
    crop_encoded = label_encoders["Crop"].transform([crop])[0]
    soil_encoded = label_encoders["soil_type"].transform([soil])[0]
    stage_encoded = label_encoders["Seedling Stage"].transform([stage])[0]

    # Prepare input data
    input_data = np.array([[crop_encoded, soil_encoded, stage_encoded, data.soil_moisture, data.temperature, data.humidity]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Irrigation Required" if prediction == 1 else "No Irrigation Needed"

    return {"prediction": result}

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Irrigation Prediction API"}
