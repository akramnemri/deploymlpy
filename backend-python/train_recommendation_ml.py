# backend/train_recommendation_ml.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading dataset...")
df = pd.read_csv('../datasets/gym_members_exercise_tracking_synthetic_data.csv')

# === 1. Clean numeric columns ===
numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage',
                'Avg_BPM', 'Max_BPM', 'Session_Duration (hours)', 'Calories_Burned']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\\t', '', regex=True).str.strip(), errors='coerce')

df = df.dropna(subset=numeric_cols + ['Workout_Type', 'Gender'])

# === 2. Create 'goal' from weight change (this is what was missing!) ===
df = df.sort_values(['Weight (kg)'])  # simulate time
df['prev_weight'] = df['Weight (kg)'].shift(1)
df['weight_change'] = df['prev_weight'] - df['Weight (kg)']

# Define goal based on weight trend
df['goal'] = 'maintain'
df.loc[df['weight_change'] > 0.5, 'goal'] = 'lose_weight'     # lost >0.5 kg
df.loc[df['weight_change'] < -0.5, 'goal'] = 'gain_muscle'   # gained >0.5 kg

print(f"Goal distribution:\n{df['goal'].value_counts()}")

# === 3. Features (7 total) ===
features = [
    'BMI', 'Fat_Percentage', 'Age',
    'Session_Duration (hours)', 'Calories_Burned',
    'goal', 'Gender'
]

X = df[['BMI', 'Fat_Percentage', 'Age', 'Session_Duration (hours)', 'Calories_Burned']].copy()
X['goal'] = df['goal'].map({'lose_weight': 0, 'gain_muscle': 1, 'maintain': 2})
X['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

y = df['Workout_Type']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 4. Train high-confidence model ===
print("Training RandomForest (500 trees)...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X, y_encoded)

# === 5. Save ===
joblib.dump({
    'model': model,
    'workout_encoder': le,
    'feature_cols': features
}, 'recommendation_ml.pkl')

print("Model saved → recommendation_ml.pkl")
print("Classes:", le.classes_)