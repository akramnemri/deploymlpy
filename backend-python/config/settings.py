# config/settings.py
from datetime import datetime, timedelta

EXERCISE_DB = {
    "strength": [
        {"name": "Bench Press", "muscle": "chest", "base_cal": 8},
        {"name": "Squat", "muscle": "legs", "base_cal": 10},
        {"name": "Deadlift", "muscle": "back", "base_cal": 12},
        {"name": "Pull-Ups", "muscle": "back", "base_cal": 9},
        {"name": "Overhead Press", "muscle": "shoulders", "base_cal": 8},
        {"name": "Barbell Row", "muscle": "back", "base_cal": 9},
        {"name": "Lunges", "muscle": "legs", "base_cal": 7},
    ],
    "hiit": [
        {"name": "Burpees", "base_cal": 15},
        {"name": "Mountain Climbers", "base_cal": 12},
        {"name": "Jump Squats", "base_cal": 14},
        {"name": "High Knees", "base_cal": 13},
        {"name": "Box Jumps", "base_cal": 16},
    ],
    "cardio": [
        {"name": "Running", "base_cal": 10},
        {"name": "Cycling", "base_cal": 8},
        {"name": "Rowing", "base_cal": 9},
        {"name": "Jump Rope", "base_cal": 12},
    ],
    "yoga": [
        {"name": "Sun Salutation", "base_cal": 4},
        {"name": "Warrior Flow", "base_cal": 5},
        {"name": "Downward Dog Hold", "base_cal": 3},
    ]
}

WEEKLY_SPLIT = {
    "lose_weight": ["hiit", "strength", "cardio", "hiit", "strength", "cardio", "rest"],
    "gain_muscle": ["strength", "strength", "rest", "strength", "strength", "cardio", "rest"],
    "maintain": ["strength", "cardio", "yoga", "hiit", "strength", "yoga", "rest"]
}

WEEKLY_PLAN_TEMPLATES = {
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