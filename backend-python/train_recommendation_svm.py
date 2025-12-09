import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

print("Loading dataset...")
df = pd.read_csv('../datasets/gym_members_exercise_tracking_synthetic_data.csv')

# === 1. Clean numeric columns ===
numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage',
                'Avg_BPM', 'Max_BPM', 'Session_Duration (hours)', 'Calories_Burned']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\t', '', regex=True).str.strip(), errors='coerce')

# === 1.5 Clean categorical columns ===
df['Gender'] = df['Gender'].astype(str).str.strip()
df['Workout_Type'] = df['Workout_Type'].astype(str).str.replace(r'\\t|\\n|\\r', '', regex=True).str.strip()
df['Workout_Type'].replace(['nan', ''], np.nan, inplace=True)

critical = numeric_cols + ['Workout_Type', 'Gender']
df = df.dropna(subset=critical)

# === 2. Create 'goal' based on BMI ===
df['goal'] = 'maintain'
df.loc[df['BMI'] < 18.5, 'goal'] = 'gain_muscle'
df.loc[df['BMI'] > 25, 'goal'] = 'lose_weight'

print(f"Goal distribution:\n{df['goal'].value_counts()}")

# === 3. Features ===
features = [
    'BMI', 'Fat_Percentage', 'Age', 'Session_Duration (hours)', 'Calories_Burned',
    'goal', 'Gender'
]

X = df[features].copy()
le_goal = LabelEncoder()
X['goal'] = le_goal.fit_transform(df['goal'])
X['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = X.dropna()

y = df['Workout_Type'].loc[X.index]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 4. Train model ===
print("Training SVM...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(
        kernel='rbf',
        C=1000,
        gamma=0.1,
        class_weight='balanced',
        probability=True,
        random_state=42
    ))
])
model.fit(X, y_encoded)

# === 5. Save ===
joblib.dump({
    'model': model,
    'workout_encoder': le,
    'goal_encoder': le_goal,
    'feature_cols': features
}, 'recommendation_svm.pkl')

print("SVM model saved!")
print("Classes:", le.classes_)