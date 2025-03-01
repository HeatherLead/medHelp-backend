from bson import ObjectId
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from app.config import settings

try:
    client = MongoClient(
        settings.mongoDB_url,
        server_api=ServerApi('1'))
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    client = None

if client:
    db = client["medHelp"]
    collection = db["medicineDetails"]
else:
    db = None
    collection = None


def fetch_medicine_by_id(medicine_id):
    if collection is None:
        return {"error": "Database connection not established."}

    try:
        query = {"_id": ObjectId(medicine_id)}
        result = collection.find_one(query)

        if result:
            result["_id"] = str(result["_id"])
            return result

        return {"error": "Medicine not found"}

    except Exception as e:
        return {"error": f"Invalid _id or database error: {str(e)}"}
