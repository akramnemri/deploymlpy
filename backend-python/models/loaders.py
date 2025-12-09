# models/loaders.py
import joblib

def load_all_models():
    models = {}
    
    try:
        models['calories'] = joblib.load('calories_model_gradient_boosting.pkl')
        print("✅ Loaded calories model")
    except FileNotFoundError:
        print("⚠️ Calories model not found")
    
    try:
        models['weight'] = joblib.load('weight_model_gradient_boosting.pkl')
        print("✅ Loaded weight model")
    except FileNotFoundError:
        print("⚠️ Weight model not found")
    
    try:
        models['rules'] = joblib.load('recommendation_rules.pkl')
        print("✅ Loaded rules")
    except FileNotFoundError:
        print("⚠️ Rules not found")
    
    try:
        rec_dict = joblib.load('recommendation_gradient_boosting.pkl')
        models['ml'] = rec_dict['model']
        models['encoder'] = rec_dict['workout_encoder']
        print("✅ Loaded ML recommendation model")
    except FileNotFoundError:
        print("⚠️ ML recommendation not found")
    
    return models