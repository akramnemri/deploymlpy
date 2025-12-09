# routes/weight.py
from flask import Blueprint, jsonify, request
import pandas as pd
import traceback

weight_bp = Blueprint('weight', __name__)
model = None

def get_model():
    """Lazy-load model to avoid circular imports"""
    from app import models
    return models.get("weight")

@weight_bp.route("/predict-weight", methods=["POST"])
def predict_weight():
    global model
    
    # Lazy load if not already loaded
    if model is None:
        model = get_model()
    
    if model is None:
        return jsonify({"error": "Weight model not loaded"}), 500

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No JSON payload received"}), 400
        
        print("Weight payload:", payload)  # DEBUG

        rows = [payload] if isinstance(payload, dict) else payload
        
        # Validate required fields
        required = ['current_day', 'days_ahead', 'steps', 'calories_burned', 'workout_type']
        for row in rows:
            missing = set(required) - set(row.keys())
            if missing:
                return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = pd.DataFrame(rows)
        print("Weight DataFrame:\n", df.head())  # DEBUG

        # Clean numeric columns
        numeric_cols = ['current_day', 'days_ahead', 'steps', 'calories_burned']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for NaN values
        if df[numeric_cols].isna().any().any():
            return jsonify({"error": "Invalid numeric values in input"}), 400
            
        # Create feature
        df["days_future"] = df["current_day"].astype(int) + df["days_ahead"].astype(int)
        
        # Select features in correct order
        X = df[["days_future", "steps", "calories_burned", "workout_type"]]
        print("Final X:\n", X.head())  # DEBUG

        prediction = model.predict(X).tolist()
        print("Weight predictions:", prediction)  # DEBUG

        return jsonify({"predicted_weights": prediction})

    except Exception as e:
        tb = traceback.format_exc()
        print("WEIGHT ERROR:\n", tb)  # SERVER-SIDE LOG
        error_msg = str(e) if str(e) else "Unknown server error"
        return jsonify({"error": error_msg}), 500