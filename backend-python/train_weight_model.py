# backend/train_weight_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Synthetic time-series data
np.random.seed(42)
n = 10000
data = {
    'days_future': np.random.uniform(1, 365, n),
    'steps': np.random.uniform(2000, 20000, n),
    'calories_burned': np.random.uniform(100, 1500, n),
    'workout_type': np.random.choice(['Strength', 'Cardio', 'HIIT'], n)
}



df = pd.DataFrame(data)
df['weight_loss'] = df['calories_burned'] * df['days_future'] / 7700
df['step_effect'] = df['steps'] * df['days_future'] / 1e6
df['workout_effect'] = df['workout_type'].map({'Strength': 0.3, 'Cardio': -1.2, 'HIIT': -1.0})
df['predicted_weight'] = 80 - df['weight_loss'] - df['step_effect'] + df['workout_effect'] * df['days_future'] / 30 + np.random.normal(0, 1.5, n)

X = df[['days_future', 'steps', 'calories_burned', 'workout_type']]
y = df['predicted_weight']

df['calories_burned'] = pd.to_numeric(df['calories_burned'], errors='coerce')
df['steps'] = pd.to_numeric(df['steps'], errors='coerce')
df = df.dropna()

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['workout_type'])],
    remainder='passthrough'
)

weight_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

weight_model.fit(X, y)

with open('weight_model.pkl', 'wb') as f:
    pickle.dump(weight_model, f)
print("Weight model saved as weight_model.pkl")