# routes/calories.py
from flask import Blueprint, jsonify, request
import pandas as pd
import traceback

calories_bp = Blueprint('calories', __name__)
model = None

@calories_bp.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Calories model not loaded"}), 500

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No JSON payload received"}), 400
        
        print(f"📥 CALORIES PAYLOAD: {payload}")  # DEBUG
        
        rows = [payload] if isinstance(payload, dict) else payload
        if not rows:
            return jsonify({"error": "Empty payload"}), 400

        # CRITICAL FIX: Create DataFrame without columns parameter
        # Let pandas use the keys from the JSON directly
        df = pd.DataFrame(rows)
        print(f"📊 DataFrame columns: {df.columns.tolist()}")  # DEBUG
        
        # Define required columns
        required_cols = [
            "Avg_BPM", "Max_BPM", "Session_Duration (hours)",
            "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage", "Workout_Type"
        ]
        
        # Check for missing columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return jsonify({"error": f"Missing columns: {list(missing_cols)}"}), 400
        
        # Reorder to match model's expected order
        df = df[required_cols]
        
        # Clean numeric columns - FIX regex from r'\\t' to r'\t'
        numeric_cols = required_cols[:-1]  # All except Workout_Type
        for col in numeric_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'\t', '', regex=True).str.strip(),
                errors="coerce"
            )
        
        # Check for NaN values
        if df[numeric_cols].isna().any().any():
            return jsonify({"error": "Invalid numeric values after cleaning"}), 400
        
        # Ensure Workout_Type is string
        df["Workout_Type"] = df["Workout_Type"].astype(str)
        
        print(f"🎯 FINAL DATAFRAME:\n{df.head()}")  # DEBUG
        
        preds = model.predict(df).tolist()
        print(f"✅ PREDICTIONS: {preds}")  # DEBUG
        
        return jsonify({"predictions": preds})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"🔥 CALORIES ERROR:\n{tb}")  # SERVER-SIDE LOG
        # Return clean error message
        return jsonify({"error": str(e)}), 500