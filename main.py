from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
import os

app = FastAPI(title="Car Price Prediction API", description="Predict the price of a car based on its features")

# Load the trained model
with open('car_price_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_columns = model_data['feature_columns']

class CarFeatures(BaseModel):
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float  # mileage(km/ltr/kg)
    engine: float
    max_power: float
    seats: float
    car_age: int

@app.get("/", response_class=HTMLResponse)
async def get_form():
    # Serve the HTML file
    if os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    else:
        return HTMLResponse("<h1>Error: index.html file not found</h1>", status_code=404)

@app.post("/predict")
async def predict_price(features: CarFeatures):
    try:
        # Prepare the input data in the correct order
        input_data = np.array([[
            features.km_driven,
            label_encoders['fuel'].transform([features.fuel])[0],
            label_encoders['seller_type'].transform([features.seller_type])[0],
            label_encoders['transmission'].transform([features.transmission])[0],
            label_encoders['owner'].transform([features.owner])[0],
            features.mileage,
            features.engine,
            features.max_power,
            features.seats,
            features.car_age
        ]])
        
        # Make prediction (returns scaled value)
        scaled_prediction = model.predict(input_data)
        
        # Inverse transform to get actual price
        actual_prediction = scaler.inverse_transform(scaled_prediction.reshape(-1, 1))
        predicted_price = float(actual_prediction[0][0])
        
        return {"predicted_price": predicted_price}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)