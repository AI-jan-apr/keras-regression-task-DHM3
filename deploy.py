import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


with open("model_weights.pkl", "rb") as f:
    weights = pickle.load(f)

with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)


model = Sequential([
    Dense(19, activation="relu"),
    Dense(19, activation="relu"),
    Dense(19, activation="relu"),
    Dense(19, activation="relu"),
    Dense(1)
])
model.build(input_shape=(None, 18))
model.set_weights(weights)

app = FastAPI()

class House(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

@app.get("/")
def home():
    return {"message": "🏠 House Price API is running!"}

@app.post("/predict")
def predict(house: House):
    data = np.array([[
        house.bedrooms, house.bathrooms, house.sqft_living,
        house.sqft_lot, house.floors, house.waterfront,
        house.view, house.condition, house.grade,
        house.sqft_above, house.sqft_basement, house.yr_built,
        house.yr_renovated, house.zipcode, house.lat,
        house.long, house.sqft_living15, house.sqft_lot15
    ]])
    data_scaled = scaler.transform(data)
    price = model.predict(data_scaled)[0][0]
    return {"predicted_price": f"${price:,.2f}"}