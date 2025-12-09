# backend-python/models/loaders.py
import os
import joblib
import requests
from io import BytesIO
from typing import Dict, Optional

# directory where this file lives -> backend-python/models
MODELS_DIR = os.path.dirname(__file__)
# Hugging Face base (used only as fallback)
HF_BASE = "https://huggingface.co/Hushfire/fitness-ml-model/resolve/main"

# map the filenames (without .pkl) to their HF raw urls (fallback)
MODEL_URLS = {
    "calories_model_gradient_boosting": f"{HF_BASE}/calories_model_gradient_boosting.pkl",
    "calories_model_svm":             f"{HF_BASE}/calories_model_svm.pkl",
    "weight_model_gradient_boosting": f"{HF_BASE}/weight_model_gradient_boosting.pkl",
    "weight_model_svm":               f"{HF_BASE}/weight_model_svm.pkl",
    "recommendation_rules":           f"{HF_BASE}/recommendation_rules.pkl",
    "recommendation_ml":              f"{HF_BASE}/recommendation_ml.pkl",
    "recommendation_ml_v2":           f"{HF_BASE}/recommendation_ml_v2.pkl",
    "recommendation_gradient_boosting": f"{HF_BASE}/recommendation_gradient_boosting.pkl",
    "recommendation_logistic_regression": f"{HF_BASE}/recommendation_logistic_regression.pkl",
    "recommendation_svm":             f"{HF_BASE}/recommendation_svm.pkl",
    "model":                          f"{HF_BASE}/model.pkl",
}

def _local_path(name: str) -> str:
    return os.path.join(MODELS_DIR, f"{name}.pkl")

def _download_to_local(url: str, local_path: str, timeout: int = 60) -> bool:
    try:
        print(f"Downloading: {url}")
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"❌ Download failed for {url}: {e}")
        return False

def _load_model_from_path(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        # try classic pickle fallback (some pickles saved with pickle)
        try:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            raise RuntimeError(f"Failed to load model at {path}: {e} / {e2}")

def _canonical_key(name: str) -> str:
    n = name.lower()
    if "calor" in n: return "calories"
    if "weight" in n: return "weight"
    if "rule" in n: return "rules"
    if n in ("model", "recommendation_ml", "recommendation_ml_v2", "recommendation_gradient_boosting", "recommendation_logistic_regression", "recommendation_svm"):
        return "ml"
    if "encoder" in n or "enc" in n: return "encoder"
    return name

def load_all_models() -> Dict[str, Optional[object]]:
    """
    Looks for local models in backend-python/models/*.pkl first.
    If not present, attempts to download from HuggingFace (MODEL_URLS).
    Returns a dict mapping raw filenames (no .pkl) plus canonical keys ('calories','weight','rules','ml','encoder') where detected.
    """
    models: Dict[str, Optional[object]] = {}
    for name, hf_url in MODEL_URLS.items():
        local = _local_path(name)
        loaded = None

        if os.path.exists(local):
            try:
                loaded = _load_model_from_path(local)
                print(f"✅ Loaded local model: {name}")
            except Exception as e:
                print(f"❌ Failed to load local {name}: {e}")

        if loaded is None:
            # try HF fallback, write into local path if success
            ok = _download_to_local(hf_url, local)
            if ok:
                try:
                    loaded = _load_model_from_path(local)
                    print(f"✅ Downloaded & loaded model: {name}")
                except Exception as e:
                    print(f"❌ Failed to load downloaded {name}: {e}")

        models[name] = loaded
        # set canonical short name if not present
        canon = _canonical_key(name)
        if canon and canon not in models:
            models[canon] = loaded

    # ensure presence of basic canonical keys even if None
    for key in ("calories", "weight", "rules", "ml", "encoder"):
        models.setdefault(key, None)

    print("✅ Models keys present:", [k for k,v in models.items() if v is not None])
    return models
