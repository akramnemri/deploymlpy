# backend-python/models/loaders.py
import os
import pickle
import requests
from io import BytesIO

# sklearn imports needed for unpickling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Cache directory for downloaded models
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/models")
os.makedirs(CACHE_DIR, exist_ok=True)

# Only include models that exist on HuggingFace
MODEL_URLS = {
    "rules": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/recommendation_rules.pkl"
}

def download_model(url, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Downloading: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(r.content)
    return pickle.load(BytesIO(r.content))

def load_all_models():
    models = {}
    for key, url in MODEL_URLS.items():
        local_path = os.path.join(CACHE_DIR, f"{key}.pkl")
        try:
            models[key] = download_model(url, local_path)
            print(f"✅ Loaded model: {key}")
        except Exception as e:
            print(f"❌ Failed to load {key}: {e}")
            models[key] = None

    # Canonical keys for Flask
    if "rules" in models:
        models.setdefault("rules", models["rules"])

    return models
