# backend-python/models/loaders.py
import pickle
import requests
from io import BytesIO

# HuggingFace raw URLs for each model
MODEL_URLS = {
    "calories": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/calories_model_gradient_boosting.pkl",
    "weight": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/weight_model_gradient_boosting.pkl",
    "rules": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/recommendation_rules.pkl",
    "ml": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/recommendation_ml.pkl",
    "ml_v2": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/recommendation_ml_v2.pkl",
    "recommendation_gradient_boosting": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/recommendation_gradient_boosting.pkl",
    "recommendation_logistic_regression": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/recommendation_logistic_regression.pkl",
    "recommendation_svm": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/recommendation_svm.pkl",
    "calories_svm": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/calories_model_svm.pkl",
    "weight_svm": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/weight_model_svm.pkl",
    "weight_gradient_boosting": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/weight_model_gradient_boosting.pkl",
    "model": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/model.pkl",
    "encoder": "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main/workout_encoder.pkl"
}

def download_model(url):
    print(f"Downloading: {url}")
    r = requests.get(url)
    r.raise_for_status()
    return pickle.load(BytesIO(r.content))

def load_all_models():
    models = {}
    for key, url in MODEL_URLS.items():
        try:
            models[key] = download_model(url)
            print(f"✅ Loaded model: {key}")
        except Exception as e:
            print(f"❌ Failed to load {key}: {e}")
            models[key] = None

    # Add canonical keys for Flask app
    if "calories" in models:
        models.setdefault("calories", models["calories"])
    if "weight" in models:
        models.setdefault("weight", models["weight"])
    if "rules" in models:
        models.setdefault("rules", models["rules"])
    if "ml" in models:
        models.setdefault("ml", models["ml"])
    if "encoder" in models:
        models.setdefault("encoder", models["encoder"])

    return models
