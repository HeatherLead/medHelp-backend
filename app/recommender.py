import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

file_path = "../data/medicineDetails.csv"
df = pd.read_csv(file_path)
df.fillna("", inplace=True)

df["_id"] = df["_id"].astype(str)

df["searchText"] = df.apply(
    lambda row: " ".join(filter(None, [row["Medicine Name"], row["Composition"], row["Uses"]])),
    axis=1,
).str.lower()

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["searchText"])

knn_model = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn_model.fit(tfidf_matrix, df["_id"])

joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(knn_model, "../models/knn_model.pkl")

print("✅ Model trained using `_id` for structured recommendations!")
