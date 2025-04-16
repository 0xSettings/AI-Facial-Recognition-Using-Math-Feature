from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client['face_db']
collection = db['features']

def save_user(name, feature_vector):
    doc = {"name": name, "vector": feature_vector.tolist()}
    collection.insert_one(doc)

def fetch_all_vectors():
    return list(collection.find({}))