# routes/recommend_ml.py
from flask import Blueprint, jsonify, request, current_app
import pandas as pd
import traceback

ml_bp = Blueprint('ml', __name__)

@ml_bp.route("/recommend-ml", methods=["POST"])
def recommend_ml():
    # Get models from app config - NO MORE GLOBAL VARIABLES
    models = current_app.config.get('models', {})
    model = models.get('ml')
    workout_encoder = models.get('encoder')
    
    if model is None or workout_encoder is None:
        return jsonify({"error": "ML model or workout encoder not loaded"}), 500

    try:
        payload = request.get_json()
        if isinstance(payload, dict):
            users = [payload]
        elif isinstance(payload, list):
            users = payload
        else:
            return jsonify({"error": "Expected object or array"}), 400

        results = []
        goal_map = {"lose_weight": 0, "gain_muscle": 1, "maintain": 2}

        for idx, user in enumerate(users):
            # Extract & validate
            weight = float(user.get("current_weight", 70))
            height = float(user.get("height", 1.75))
            age = int(user.get("age", 30))
            fat_pct = float(user.get("fat_percentage", 20))
            duration = float(user.get("avg_duration", 1.0))
            calories = float(user.get("avg_calories", 500))
            goal_str = str(user.get("goal", "maintain")).lower().replace(" ", "_")
            gender_str = str(user.get("gender", "Male")).capitalize()

            goal_encoded = goal_map.get(goal_str, 2)
            gender_encoded = 0 if gender_str == "Male" else 1
            bmi = weight / (height ** 2)

            # Build 7-column DataFrame
            row = pd.DataFrame([[
                bmi, fat_pct, age, duration, calories, goal_encoded, gender_encoded
            ]], columns=[
                'BMI', 'Fat_Percentage', 'Age', 'Session_Duration (hours)', 
                'Calories_Burned', 'goal', 'Gender'
            ])

            # Predict
            pred_encoded = model.predict(row)[0]
            proba = model.predict_proba(row)[0].tolist()
            confidence = max(proba)
            top_workout = workout_encoder.inverse_transform([pred_encoded])[0]

            # Explanation
            explanations = {
                'Cardio': f"Steady calorie burn. Best for BMI > 25 and longer sessions.",
                'HIIT': f"Max fat loss in short bursts. Ideal for high fat % ({fat_pct}%) and lose-weight goal.",
                'Strength': f"Builds muscle → boosts resting metabolism. Common for males and BMI ~{bmi:.1f}.",
                'Yoga': f"Recovery & flexibility. Low calorie burn, great for maintain goal."
            }
            explanation = explanations.get(top_workout, "Based on your profile.")

            results.append({
                "goal": goal_str.replace("_", " ").title(),
                "gender": gender_str,
                "recommended_workout": top_workout,
                "confidence": round(confidence, 3),
                "duration_hours": round(duration, 2),
                "bmi": round(bmi, 2),
                "all_probas": proba,
                "all_workouts": workout_encoder.classes_.tolist(),
                "explanation": explanation
            })

        return jsonify({"recommendations": results})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"🔥 ML RECOMMEND ERROR:\n{tb}")
        return jsonify({"error": str(e)}), 500