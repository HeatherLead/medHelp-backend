from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from app.mongodb_database import fetch_medicine_by_text

app = FastAPI()

try:
    df = pd.read_csv("data/Medicine_Details.csv")
    df.fillna("", inplace=True)

    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    knn_model = joblib.load("models/knn_model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading files: {str(e)}")

class MedicineSearchRequest(BaseModel):
    medicine_name: Optional[str] = ""
    composition: Optional[str] = ""
    uses: Optional[str] = ""

@app.post("/recommend")
def recommend_medicine(request: MedicineSearchRequest):
    try:
        search_text = " ".join(filter(None, [request.medicine_name, request.composition, request.uses])).lower()

        if not search_text:
            raise HTTPException(status_code=400, detail="At least one field must be provided.")

        user_vector = vectorizer.transform([search_text])

        distances, indices = knn_model.kneighbors(user_vector)
        recommended_queries = [knn_model.classes_[idx] for idx in indices[0]]

        return {"recommended_queries": recommended_queries}

        # If fetching full medicine data from MongoDB:
        # recommended_medicines = [fetch_medicine_by_text(query) for query in recommended_queries]
        # return {"alternatives": recommended_medicines}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
