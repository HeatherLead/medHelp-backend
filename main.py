from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from app.mongodb_database import fetch_medicine_by_id
import numpy as np
from app.review_prediction import get_recommendation

app = FastAPI()

try:
    df = pd.read_csv("data/medicineDetails.csv")
    df.fillna("", inplace=True)

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    knn_model = joblib.load("models/knn_model.pkl")
    kmeans_model = joblib.load("models/kmeans_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading files: {str(e)}")

class MedicineSearchRequest(BaseModel):
    medicine_name: Optional[str] = ""
    composition: Optional[str] = ""
    uses: Optional[str] = ""

class MedicineReviewRequest(BaseModel):
    excellent_review: float
    average_review: float
    poor_review: float

@app.get("/")
def read_root():
    return {"Wellcome to medHelp"}

@app.post("/recommend")
def recommend_medicine(request: MedicineSearchRequest):
    try:
        search_text = " ".join(filter(None, [request.medicine_name, request.composition, request.uses])).lower()

        if not search_text:
            raise HTTPException(status_code=400, detail="At least one field must be provided.")

        user_vector = vectorizer.transform([search_text])
        distances, indices = knn_model.kneighbors(user_vector)

        recommended_ids = [knn_model.classes_[idx] for idx in indices[0]]

        recommended_medicines = [fetch_medicine_by_id(med_id) for med_id in recommended_ids]

        return recommended_medicines

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/reviewAnalysis")
def cluster_medicine(request: MedicineReviewRequest):
    try:
        new_medicine = np.array([[request.excellent_review, request.average_review, request.poor_review]])

        cluster = kmeans_model.predict(new_medicine)[0]

        recommendation = get_recommendation(request.dict())

        return {"cluster": int(cluster), "recommendation": recommendation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")