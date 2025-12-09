# backend-python/routes/workout.py
from flask import Blueprint, jsonify, request
from config.settings import EXERCISE_DB, WEEKLY_SPLIT
from datetime import datetime, timedelta
import random

workout_bp = Blueprint('workout', __name__)

def build_day_dict(user: dict, day_offset: int) -> dict:
    """Returns a plain Python dict (NOT a Flask response)"""
    weight = user.get("weightKg", 70)
    goal = user.get("goal", "maintain").lower().replace(" ", "_")
    ai_rec = user.get("recommended_workout", "HIIT")

    focus = WEEKLY_SPLIT.get(goal, WEEKLY_SPLIT["maintain"])[day_offset % 7]

    if focus == "rest":
        return {
            "day": (datetime.now() + timedelta(days=day_offset)).strftime("%A"),
            "date": (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d"),
            "focus": "Rest Day",
            "message": "Recovery day – light walk or yoga optional",
            "exercises": [],
            "total_duration_min": 0,
            "total_calories": 0
        }

    # Exercise selection
    if focus == "strength":
        pool, count = EXERCISE_DB["strength"], 5 if goal == "gain_muscle" else 4
    elif focus == "hiit":
        pool, count = EXERCISE_DB["hiit"], 5
    elif focus == "cardio":
        pool, count = EXERCISE_DB["cardio"], 1
    else:
        pool, count = EXERCISE_DB["yoga"], 6

    selected = random.sample(pool, min(count, len(pool)))
    exercises = []
    total_cal = total_dur = 0
    week_factor = 1 + (day_offset // 7) * 0.03  # progressive overload

    for ex in selected:
        sets = 4 if focus == "strength" else 3
        reps = 10 if focus == "strength" else 20
        dur_min = sets * (1 if focus == "hiit" else 2)
        cal = ex["base_cal"] * sets * (weight / 70) * week_factor

        base_weight = weight * 0.7
        exercises.append({
            "name": ex["name"],
            "sets": sets,
            "reps": reps,
            "weight_kg": round(base_weight * week_factor, 1) if focus == "strength" else 0,
            "duration_min": dur_min,
            "calories_burned": round(cal)
        })
        total_cal += cal
        total_dur += dur_min

    return {
        "day": (datetime.now() + timedelta(days=day_offset)).strftime("%A"),
        "date": (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d"),
        "focus": focus.replace("_", " ").title(),
        "exercises": exercises,
        "total_duration_min": total_dur,
        "total_calories": round(total_cal)
    }

# ─────────────────────────────────────────────────────────────────────
@workout_bp.route("/daily-workout", methods=["POST"])
def daily_workout():
    data = request.get_json()
    day_dict = build_day_dict(data.get("user", {}), data.get("day_offset", 0))
    return jsonify(day_dict)                     # ← returns Response

@workout_bp.route("/weekly-workout", methods=["POST"])
def weekly_workout():
    data = request.get_json()
    user = data.get("user", {})
    week_plan = [build_day_dict(user, i) for i in range(7)]

    return jsonify({
        "week_start": datetime.now().strftime("%Y-%m-%d"),
        "goal": user.get("goal", "maintain"),
        "week_plan": week_plan
    })