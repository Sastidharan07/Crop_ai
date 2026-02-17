from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# ==============================
# Load trained system
# ==============================
with open("agri_ai_system.pkl", "rb") as f:
    system = pickle.load(f)

model_crop = system["crop_model"]
model_price = system["price_model"]
label_encoders = system["label_encoders"]


# ==============================
# Helper: Smart Encode
# ==============================

def smart_encode(le, value, field_name="Value"):
    value = value.strip().lower()
    classes = le.classes_
    lower_classes = [c.lower() for c in classes]

    # Exact match
    if value in lower_classes:
        return lower_classes.index(value)

    # Partial match (e.g., "paddy" → "Paddy (Common)")
    for i, c in enumerate(lower_classes):
        if value in c:
            return i

    raise ValueError(f"{field_name} not found. Available: {classes}")


# ==============================
# Request Models
# ==============================

class CompareRequest(BaseModel):
    state: str
    month: int
    soil: str
    irrigation: str
    year: int


class CropDetailRequest(BaseModel):
    state: str
    crop: str
    month: int
    year: int
    land_area: float = 1


# ==============================
# Home
# ==============================

@app.get("/")
def home():
    return {"message": "🌾 Agri AI API Running Successfully"}


# ==============================
# 1️⃣ Compare Crops Endpoint
# ==============================

@app.post("/compare/")
def compare_crops(request: CompareRequest):

    state_enc = smart_encode(label_encoders["State"], request.state, "State")
    soil_enc = smart_encode(label_encoders["Soil Type"], request.soil, "Soil Type")
    irrigation_enc = smart_encode(label_encoders["Irrigation Type"], request.irrigation, "Irrigation Type")

    input_data = np.array([[state_enc, request.month, soil_enc, irrigation_enc]])

    probs = model_crop.predict_proba(input_data)[0]
    top3_indices = np.argsort(probs)[-3:][::-1]

    results = []
    best_profit = -999999
    best_crop = None

    for idx in top3_indices:
        crop_name = label_encoders["Crop"].inverse_transform([idx])[0]

        price_input = np.array([[idx, state_enc, request.month, request.year]])
        predicted_price = float(model_price.predict(price_input)[0])

        avg_yield = 20  # simplified average
        avg_cost = 30000

        revenue = predicted_price * avg_yield
        profit = revenue - avg_cost

        results.append({
            "crop": crop_name,
            "predicted_price": round(predicted_price, 2),
            "estimated_profit": round(profit, 2)
        })

        if profit > best_profit:
            best_profit = profit
            best_crop = crop_name

    return {
        "state": request.state,
        "comparison": results,
        "best_crop_to_cultivate": best_crop
    }


# ==============================
# 2️⃣ Crop Detail Report Endpoint
# ==============================

@app.post("/crop-detail/")
def crop_detail_report(request: CropDetailRequest):

    state_enc = smart_encode(label_encoders["State"], request.state, "State")
    crop_enc = smart_encode(label_encoders["Crop"], request.crop, "Crop")

    best_price = 0
    best_month = request.month
    best_year = request.year

    # Forecast next 6 months
    for i in range(6):
        future_month = (request.month + i - 1) % 12 + 1
        future_year = request.year
        if request.month + i > 12:
            future_year += 1

        input_data = np.array([[crop_enc, state_enc, future_month, future_year]])
        predicted_price = float(model_price.predict(input_data)[0])

        if predicted_price > best_price:
            best_price = predicted_price
            best_month = future_month
            best_year = future_year

    avg_yield = 20
    avg_cost = 30000

    revenue = best_price * avg_yield * request.land_area
    cost = avg_cost * request.land_area
    profit = revenue - cost

    return {
        "crop": request.crop,
        "state": request.state,
        "best_selling_month": best_month,
        "best_selling_year": best_year,
        "best_predicted_price": round(best_price, 2),
        "estimated_revenue": round(revenue, 2),
        "estimated_cost": round(cost, 2),
        "estimated_profit": round(profit, 2)
    }
