import pandas as pd
from sklearn.cluster import KMeans
import joblib

file_path = "data/medicineDetails.csv"
df = pd.read_csv(file_path)

df.rename(columns={
    "Excellent Review %": "excellent_review",
    "Average Review %": "average_review",
    "Poor Review %": "poor_review"
}, inplace=True)

df = df.dropna(subset=['excellent_review', 'average_review', 'poor_review'])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df[['excellent_review', 'average_review', 'poor_review']])

joblib.dump(kmeans, "models/kmeans_model.pkl")

print("Model saved successfully.")

def get_recommendation(row):
    total_positive = row['excellent_review'] + row['average_review']

    if total_positive >= 80:
        return "✅ Highly Recommended! Most users had a great experience."
    elif total_positive >= 50:
        return "⚠️ Good Choice. Works well for many, but some had mixed results."
    else:
        return "❌ Not Recommended. Many users reported poor results."

df['recommendation'] = df.apply(get_recommendation, axis=1)
