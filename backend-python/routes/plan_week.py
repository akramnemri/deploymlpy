# routes/plan_week.py
from flask import Blueprint, jsonify, request
import random
from config.settings import WEEKLY_PLAN_TEMPLATES

plan_bp = Blueprint('plan', __name__)

@plan_bp.route("/plan-week", methods=["POST"])
def plan_week():
    try:
        payload = request.get_json()
        if isinstance(payload, dict):
            users = [payload]
        elif isinstance(payload, list):
            users = payload
        else:
            return jsonify({"error": "Expected object or array"}), 400

        results = []

        # Base templates per goal
        base_plans = {
            "lose_weight": [
                {"day": "Mon", "workout": "HIIT", "base_duration": 45},
                {"day": "Tue", "workout": "Cardio", "base_duration": 60},
                {"day": "Wed", "workout": "Strength", "base_duration": 50},
                {"day": "Thu", "workout": "HIIT", "base_duration": 40},
                {"day": "Fri", "workout": "Cardio", "base_duration": 50},
                {"day": "Sat", "workout": "Yoga", "base_duration": 30},
                {"day": "Sun", "workout": "Rest", "base_duration": 0},
            ],
            "gain_muscle": [
                {"day": "Mon", "workout": "Strength", "base_duration": 60},
                {"day": "Tue", "workout": "Strength", "base_duration": 60},
                {"day": "Wed", "workout": "Rest", "base_duration": 0},
                {"day": "Thu", "workout": "Strength", "base_duration": 60},
                {"day": "Fri", "workout": "Strength", "base_duration": 60},
                {"day": "Sat", "workout": "Cardio", "base_duration": 30},
                {"day": "Sun", "workout": "Rest", "base_duration": 0},
            ],
            "maintain": [
                {"day": "Mon", "workout": "Cardio", "base_duration": 45},
                {"day": "Tue", "workout": "Yoga", "base_duration": 45},
                {"day": "Wed", "workout": "Strength", "base_duration": 45},
                {"day": "Thu", "workout": "Cardio", "base_duration": 45},
                {"day": "Fri", "workout": "HIIT", "base_duration": 30},
                {"day": "Sat", "workout": "Yoga", "base_duration": 45},
                {"day": "Sun", "workout": "Rest", "base_duration": 0},
            ]
        }

        for idx, user in enumerate(users):
            # === Identify User ===
            user_id = user.get("user_id", f"user_{idx+1}")
            goal = user.get("goal", "maintain").lower()
            ai_workout = user.get("recommended_workout", "Cardio")
            current_weight = user.get("current_weight", 70)
            target_weight = user.get("target_weight", current_weight)
            days_ahead = max(1, user.get("days_ahead", 30))

            # === Calorie Math ===
            weight_diff = target_weight - current_weight
            total_cal = abs(weight_diff) * 7700
            daily_cal_shift = total_cal / days_ahead  # positive = surplus, negative = deficit

            # === Pick Base Plan ===
            plan_template = base_plans.get(goal, base_plans["maintain"])

            # === Build Personalized Plan ===
            week_plan = []
            for day in plan_template:
                if day["workout"] == "Rest":
                    week_plan.append(day.copy())
                    continue

                # 60% chance to use AI recommendation
                workout = ai_workout if random.random() < 0.6 else day["workout"]

                # Scale duration by calorie shift
                base_dur = day["base_duration"]
                if goal == "lose_weight" and weight_diff < 0:
                    duration = int(base_dur + (abs(daily_cal_shift) / 25))  # +1 min per 25 cal deficit
                elif goal == "gain_muscle" and weight_diff > 0:
                    duration = int(base_dur + (daily_cal_shift / 30))       # +1 min per 30 cal surplus
                else:
                    duration = base_dur

                duration = max(20, min(90, duration))  # clamp

                week_plan.append({
                    "day": day["day"],
                    "workout": workout,
                    "duration": duration if duration > 0 else 0
                })

            results.append({
                "user_id": user_id,
                "goal": goal.replace("_", " ").title(),
                "ai_recommended": ai_workout,
                "daily_calorie_shift": round(daily_cal_shift, 0),
                "week_plan": week_plan
            })

        return jsonify({"plans": results})
    except Exception as e:
        print("ERROR in /api/plan-week:", str(e))
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    