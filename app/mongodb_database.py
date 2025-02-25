from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["medicine_db"]
collection = db["medicines"]


def fetch_medicine_by_text(query_text):
    result = collection.find_one({"searchText": query_text.lower()})

    if result:
        return {
            "Medicine Name": result.get("Medicine Name", ""),
            "Composition": result.get("Composition", ""),
            "Uses": result.get("Uses", ""),
            "Manufacturer": result.get("Manufacturer", ""),
            "Price": result.get("Price", ""),
            "Side Effects": result.get("Side Effects", ""),
            "Storage": result.get("Storage", ""),
        }
    return {"error": "Medicine not found"}
