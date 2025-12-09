# backend-python/app.py
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# 1. Load models FIRST
# ---------------------------------------------------
from models.loaders import load_all_models
models = load_all_models()
app.config['models'] = models

# ---------------------------------------------------
# 2. Import route MODULES (so we can set their variables)
# ---------------------------------------------------
import routes.calories
import routes.weight
import routes.recommend
import routes.recommend_ml
import routes.plan_week
import routes.workout

# ---------------------------------------------------
# 3. Import blueprints from modules
# ---------------------------------------------------
from routes.calories import calories_bp
from routes.weight import weight_bp
from routes.recommend import recommend_bp
from routes.recommend_ml import ml_bp
from routes.plan_week import plan_bp
from routes.workout import workout_bp

# ---------------------------------------------------
# 4. Inject models into MODULES (only if available)
# ---------------------------------------------------
# safe assignment for optional models
routes.calories.model = models.get("calories") if models.get("calories") is not None else None
routes.weight.model = models.get("weight") if models.get("weight") is not None else None
routes.recommend.rules = models.get("rules") if models.get("rules") is not None else None

# For ML endpoints, require both model and encoder to be present
ml_model = models.get("ml")
encoder = models.get("encoder")
if ml_model is not None and encoder is not None:
    routes.recommend_ml.model = ml_model
    routes.recommend_ml.encoder = encoder
else:
    # leave attributes unset (or explicitly set to None) so route handlers can respond cleanly
    routes.recommend_ml.model = None
    routes.recommend_ml.encoder = None
    print("⚠️ ML model and/or encoder not available. /api/recommend-ml will return a safe error response until both are present.")

# ---------------------------------------------------
# 5. Register blueprints
# ---------------------------------------------------
app.register_blueprint(calories_bp, url_prefix="/api")
app.register_blueprint(weight_bp, url_prefix="/api")
app.register_blueprint(recommend_bp, url_prefix="/api")
app.register_blueprint(ml_bp, url_prefix="/api")
app.register_blueprint(plan_bp, url_prefix="/api")
app.register_blueprint(workout_bp, url_prefix="/api/workout")


@app.route("/api/status")
def status():
    loaded = [k for k, v in models.items() if v is not None]
    return {
        "status": "OK",
        "models_loaded": loaded
    }

if __name__ == "__main__":
    print("✅ Models keys present:", list(models.keys()))
    print(f"🔥 Calories model ready: {routes.calories.model is not None}")
    print(f"🔥 Weight model ready: {routes.weight.model is not None}")
    print(f"🔥 Rules model ready: {routes.recommend.rules is not None}")
    print(f"🔥 ML model+encoder ready: {routes.recommend_ml.model is not None and routes.recommend_ml.encoder is not None}")
    app.run(host="0.0.0.0", port=5000, debug=True)
