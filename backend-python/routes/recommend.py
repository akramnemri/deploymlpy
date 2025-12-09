# routes/recommend.py
from flask import Blueprint, jsonify, request, current_app
import random
import traceback

recommend_bp = Blueprint('recommend', __name__)

@recommend_bp.route("/recommend", methods=["POST"])
def recommend_workout():
    # Get rules from app config - NO MORE GLOBAL VARIABLES
    rules = current_app.config.get('models', {}).get('rules')
    
    if rules is None:
        return jsonify({"error": "Recommendation engine not loaded"}), 500

    try:
        payload = request.get_json()
        if isinstance(payload, dict):
            users = [payload]
        elif isinstance(payload, list):
            users = payload
        else:
            return jsonify({"error": "Expected object or array"}), 400

        results = []
        for user in users:
            goal = user.get("goal", "maintain").lower().replace(" ", "_")
            current_weight = user.get("current_weight", 70)
            target_weight = user.get("target_weight", current_weight)
            days_ahead = user.get("days_ahead", 30)

            rule = next((r for r in rules if r["goal"] == goal), None)
            if not rule:
                rule = next(r for r in rules if r["goal"] == "maintain")

            weight_diff = target_weight - current_weight
            total_cal = abs(weight_diff) * 7700
            daily_cal = total_cal / days_ahead if days_ahead > 0 else 0

            base = rule["min_duration"]
            if goal == "lose_weight" and weight_diff < 0:
                duration = max(1.0, base + (abs(daily_cal) / 1000))
            elif goal == "gain_muscle" and weight_diff > 0:
                duration = base + 0.3
            else:
                duration = base

            workout_type = random.choice(rule["workout_types"])

            results.append({
                "goal": goal.replace("_", " ").title(),
                "recommended_workout": workout_type,
                "duration_hours": round(duration, 2),
                "intensity": rule["intensity"],
                "frequency_per_week": rule["frequency_per_week"],
                "estimated_daily_calorie_shift": round(daily_cal, 0),
                "tip": f"Focus on {workout_type.lower()} sessions to hit your goal!"
            })

        return jsonify({"recommendations": results})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"🔥 RULES RECOMMEND ERROR:\n{tb}")
        return jsonify({"error": str(e)}), 500