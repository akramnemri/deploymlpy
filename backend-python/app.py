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
# 4. Inject models into MODULES (not just blueprints)
# ---------------------------------------------------
routes.calories.model = models.get("calories")
routes.weight.model = models.get("weight")
routes.recommend.rules = models.get("rules")

if models.get("ml"):
    routes.recommend_ml.model = models["ml"]
    routes.recommend_ml.encoder = models["encoder"]

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
    return {
        "status": "OK",
        "models_loaded": list(models.keys())
    }

if __name__ == "__main__":
    print("✅ Models loaded:", list(models.keys()))
    print(f"🔥 Calories model ready: {routes.calories.model is not None}")
    print(f"🔥 Weight model ready: {routes.weight.model is not None}")
    app.run(host="0.0.0.0", port=5000, debug=True)