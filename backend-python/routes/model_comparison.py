# backend/model_comparison.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to clean dataframe
def clean_dataframe(df):
    numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                    'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 
                    'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'\t', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Workout_Type' in df.columns:
        df['Workout_Type'] = df['Workout_Type'].astype(str).str.replace(r'\\t|\\n|\\r', '', regex=True).str.strip()
        df['Workout_Type'].replace(['nan', ''], np.nan, inplace=True)
    
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip()
    
    return df

# Load and prepare data for calories prediction
def load_and_prepare_data():
    csv_path = '../datasets/gym_members_exercise_tracking_synthetic_data.csv'
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)
    
    feature_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM',
                    'Session_Duration (hours)', 'Fat_Percentage', 'Workout_Frequency (days/week)', 
                    'Experience_Level', 'BMI', 'Workout_Type']
    
    df = df.dropna(subset=feature_cols + ['Calories_Burned'])
    
    X = df[feature_cols]
    y = df['Calories_Burned']
    
    return X, y

# Create synthetic weight data
def create_weight_data():
    csv_path = '../datasets/gym_members_exercise_tracking_synthetic_data.csv'
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)
    
    critical_cols = ['Weight (kg)', 'Calories_Burned', 'Workout_Type']
    df = df.dropna(subset=critical_cols)
    
    # Create synthetic weight prediction data
    np.random.seed(42)

    df['days_future'] = np.random.uniform(1, 365, len(df))
    df['steps'] = np.random.uniform(2000, 20000, len(df))

    df['weight_loss'] = df['Calories_Burned'] * df['days_future'] / 7700  # 7700 cal = 1kg
    df['step_effect'] = df['steps'] * df['days_future'] / 1e6
    df['workout_effect'] = df['Workout_Type'].map({'Strength': 0.3, 'Cardio': -1.2, 'HIIT': -1.0, 'Yoga': -0.5})

    df['predicted_weight'] = df['Weight (kg)'] - df['weight_loss'] - df['step_effect'] + df['workout_effect'] * df['days_future'] / 30 + np.random.normal(0, 1.5, len(df))

    df = df.dropna(subset=['predicted_weight', 'days_future', 'steps', 'Calories_Burned', 'Workout_Type'])

    X = df[['days_future', 'steps', 'Calories_Burned', 'Workout_Type']]
    y = df['predicted_weight']
    
    return X, y

# Load and prepare data for recommendation (classification)
def load_and_prepare_recommendation_data():
    csv_path = '../datasets/gym_members_exercise_tracking_synthetic_data.csv'
    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)
    
    features = ['BMI', 'Fat_Percentage', 'Age', 'Session_Duration (hours)', 'Calories_Burned', 'Gender']
    critical = features + ['Workout_Type']
    
    df = df.dropna(subset=critical)
    
    # Create goal
    df['goal'] = 'maintain'
    df.loc[df['BMI'] < 18.5, 'goal'] = 'gain_muscle'
    df.loc[df['BMI'] > 25, 'goal'] = 'lose_weight'
    
    print(f"Goal distribution:\n{df['goal'].value_counts()}")
    
    features = features + ['goal']
    X = df[features].copy()
    
    le_goal = LabelEncoder()
    X['goal'] = le_goal.fit_transform(df['goal'])
    
    X['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    X = X.dropna()
    
    y = df['Workout_Type'].loc[X.index]
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    return X, y_encoded

# Train and evaluate regression models
def train_and_evaluate_models(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "calories":
        numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'Avg_BPM', 'Resting_BPM',
                           'Session_Duration (hours)', 'Fat_Percentage', 'Workout_Frequency (days/week)', 
                           'Experience_Level', 'BMI']
        categorical_features = ['Workout_Type']
    else:  # weight
        numeric_features = ['days_future', 'steps', 'Calories_Burned']
        categorical_features = ['Workout_Type']
    
    # Linear Regression
    preprocessor_lr = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    lr_model = Pipeline([
        ('preprocessor', preprocessor_lr),
        ('regressor', LinearRegression())
    ])
    
    # Gradient Boosting
    preprocessor_gb = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    gb_model = Pipeline([
        ('preprocessor', preprocessor_gb),
        ('regressor', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    
    # Random Forest
    preprocessor_rf = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    rf_model = Pipeline([
        ('preprocessor', preprocessor_rf),
        ('regressor', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    ])
    
    # SVM
    preprocessor_svm = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    svm_model = Pipeline([
        ('preprocessor', preprocessor_svm),
        ('regressor', SVR(kernel='rbf', C=1000, gamma=0.1, epsilon=0.1))
    ])
    
    models = {
        'Linear Regression': lr_model,
        'Gradient Boosting': gb_model,
        'Random Forest': rf_model,
        'SVM': svm_model
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'model': model
        }
        
        print(f"{name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    return results

# Train and evaluate classification models
def train_and_evaluate_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'SVM': Pipeline([
            ('preprocessor', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=1000,
                gamma=0.1,
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ]),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1_val = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1_val,
            'model': model
        }
        
        print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1_val:.4f}")
    
    return results

# Create comparison visualizations
def create_comparison_charts(calories_results, weight_results, rec_results):
    models_reg = list(calories_results.keys())
    calories_r2 = [calories_results[m]['R2'] for m in models_reg]
    weight_r2 = [weight_results[m]['R2'] for m in models_reg]
    calories_rmse = [calories_results[m]['RMSE'] for m in models_reg]
    weight_rmse = [weight_results[m]['RMSE'] for m in models_reg]
    
    # Regression figure
    fig_reg, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(models_reg))
    width = 0.35
    
    ax1.bar(x - width/2, calories_r2, width, label='Calories', alpha=0.8)
    ax1.bar(x + width/2, weight_r2, width, label='Weight', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Regression R² Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_reg, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(x - width/2, calories_rmse, width, label='Calories', alpha=0.8)
    ax2.bar(x + width/2, weight_rmse, width, label='Weight', alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Regression RMSE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_reg, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig_reg.savefig('regression_comparison.png', dpi=300, bbox_inches='tight')
    
    # Classification figure
    models_class = list(rec_results.keys())
    f1s = [rec_results[m]['F1'] for m in models_class]
    accs = [rec_results[m]['Accuracy'] for m in models_class]
    
    fig_class, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models_class))
    width = 0.35
    
    ax.bar(x - width/2, f1s, width, label='F1 Score', alpha=0.8)
    ax.bar(x + width/2, accs, width, label='Accuracy', alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models_class, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig_class.savefig('classification_comparison.png', dpi=300, bbox_inches='tight')

# Main execution
if __name__ == "__main__":
    print("=== CALORIES PREDICTION MODELS ===")
    X_calories, y_calories = load_and_prepare_data()
    calories_results = train_and_evaluate_models(X_calories, y_calories, "calories")
    
    print("\n=== WEIGHT PREDICTION MODELS ===")
    X_weight, y_weight = create_weight_data()
    weight_results = train_and_evaluate_models(X_weight, y_weight, "weight")
    
    print("\n=== RECOMMENDATION CLASSIFICATION MODELS ===")
    X_rec, y_rec = load_and_prepare_recommendation_data()
    rec_results = train_and_evaluate_classification_models(X_rec, y_rec)
    
    print("\n=== CREATING COMPARISON VISUALIZATIONS ===")
    create_comparison_charts(calories_results, weight_results, rec_results)
    
    # Save results
    joblib.dump({
        'calories_results': calories_results,
        'weight_results': weight_results,
        'recommendation_results': rec_results
    }, 'model_comparison_results.pkl')
    
    print("\n=== BEST PERFORMING MODELS ===")
    best_calories = max(calories_results, key=lambda x: calories_results[x]['R2'])
    best_weight = max(weight_results, key=lambda x: weight_results[x]['R2'])
    best_rec = max(rec_results, key=lambda x: rec_results[x]['F1'])
    
    print(f"Best Calories Model: {best_calories} (R² = {calories_results[best_calories]['R2']:.4f})")
    print(f"Best Weight Model: {best_weight} (R² = {weight_results[best_weight]['R2']:.4f})")
    print(f"Best Recommendation Model: {best_rec} (F1 = {rec_results[best_rec]['F1']:.4f})")