import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

file_path = "../data/Medicine_Details.csv"
df = pd.read_csv(file_path)
df.fillna("", inplace=True)

df["searchText"] = df.apply(
    lambda row: " ".join(filter(None, [row["Medicine Name"], row["Composition"], row["Uses"]])),
    axis=1,
).str.lower()

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["searchText"])

knn_model = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn_model.fit(tfidf_matrix, df["searchText"])

joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(knn_model, "../models/knn_model.pkl")

print("✅ Model trained using combined medicine data for better recommendations!")
