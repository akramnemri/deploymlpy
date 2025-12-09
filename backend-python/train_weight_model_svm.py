# backend/train_weight_model_svm.py
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# === 1. Load and CLEAN the CSV ===
csv_path = '../datasets/gym_members_exercise_tracking_synthetic_data.csv'
df = pd.read_csv(csv_path)

print(f"Original shape: {df.shape}")

# === 2. Clean numeric columns: remove tabs, strip, convert to float ===
numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 
                'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']

for col in numeric_cols:
    if col in df.columns:
        # Replace tabs and strip whitespace, handle mixed types
        df[col] = df[col].astype(str).str.replace(r'\t', '', regex=True).str.strip()
        # Convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col], errors='coerce')

# === 2.5 Clean categorical columns ===
if 'Workout_Type' in df.columns:
    df['Workout_Type'] = df['Workout_Type'].astype(str).str.replace(r'\\t|\\n|\\r', '', regex=True).str.strip()
    df['Workout_Type'].replace(['nan', ''], np.nan, inplace=True)  # Handle 'nan' strings and empties as NaN

# === 3. Drop rows with NaN in critical columns ===
critical_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Session_Duration (hours)', 'Calories_Burned', 'Workout_Type']
df = df.dropna(subset=critical_cols)
print(f"After cleaning: {df.shape}")

# === 4. Create synthetic weight prediction data ===
np.random.seed(42)

# Create time-based features
df['days_future'] = np.random.uniform(1, 365, len(df))
df['steps'] = np.random.uniform(2000, 20000, len(df))

# Calculate weight effects
df['weight_loss'] = df['Calories_Burned'] * df['days_future'] / 7700  # 7700 cal = 1kg
df['step_effect'] = df['steps'] * df['days_future'] / 1e6
df['workout_effect'] = df['Workout_Type'].map({'Strength': 0.3, 'Cardio': -1.2, 'HIIT': -1.0, 'Yoga': -0.5})

# Predict weight with some noise
df['predicted_weight'] = df['Weight (kg)'] - df['weight_loss'] - df['step_effect'] + df['workout_effect'] * df['days_future'] / 30 + np.random.normal(0, 1.5, len(df))

# Drop any rows where predicted_weight is NaN (in case of unmapped Workout_Type or other issues)
df = df.dropna(subset=['predicted_weight'])

# === 5. Prepare features and target ===
X = df[['days_future', 'steps', 'Calories_Burned', 'Workout_Type']]
y = df['predicted_weight']

# === 6. Split data for evaluation ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 7. Build pipeline with SVM ===
numeric_features = ['days_future', 'steps', 'Calories_Burned']
categorical_features = ['Workout_Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

weight_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR(
        kernel='rbf',
        C=1000,
        gamma=0.1,
        epsilon=0.1
    ))
])

# === 8. Train and evaluate ===
weight_model.fit(X_train, y_train)
y_pred = weight_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5  # Manual calculation for compatibility
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Weight SVM Results:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# === 9. Train on full dataset and save ===
weight_model.fit(X, y)
joblib.dump(weight_model, 'weight_model_svm.pkl')
print("Weight SVM model saved!")