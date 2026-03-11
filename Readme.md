# 🏠 King County House Price Prediction

A deep learning regression model built with **Keras** to predict house prices, deployed as a REST API using **FastAPI**.

---

## 📁 Project Structure

```
├── keras-regression-task.ipynb  ← EDA + Training + Evaluation
├── deploy.py                    ← FastAPI REST API
├── model_weights.pkl            ← Saved Keras model weights
├── scaler_weights.pkl           ← Saved MinMaxScaler
└── kc_house_data.csv            ← Dataset (King County House Sales)
```

---

## 📊 Dataset

- **Source**: [Kaggle - House Sales in King County](https://www.kaggle.com/harlfoxem/housesalesprediction)
- **Size**: 21,597 houses
- **Target**: `price`
- **Features**: 18 numeric features (bedrooms, bathrooms, sqft_living, location, etc.)

---

## 🧠 Model Architecture

```
Input (18 features)
  → Dense(19, ReLU)
  → Dense(19, ReLU)
  → Dense(19, ReLU)
  → Dense(19, ReLU)
  → Dense(1)  ← Predicted Price
```

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss | MSE |
| Early Stopping | patience=25 |
| Scaler | MinMaxScaler |
| Train/Test Split | 70% / 30% |

---

## ⚙️ Installation

```bash
pip install tensorflow==2.13.0 "numpy<2" scikit-learn fastapi uvicorn
```

---

## 🚀 Run the API

```bash
uvicorn deploy:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message |
| POST | `/predict` | Predict house price |

### Sample Request

```json
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 7500,
  "floors": 2.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 8,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "yr_built": 1990,
  "yr_renovated": 0,
  "zipcode": 98052,
  "lat": 47.6740,
  "long": -122.1215,
  "sqft_living15": 1750,
  "sqft_lot15": 7200
}
```

### Sample Response

```json
{
  "predicted_price": "$487,320.50"
}
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-red)
