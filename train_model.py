# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Simulated dataset (replace with real data if available)
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'BMI': np.random.normal(27, 4, n),
    'AgeCategory': np.random.choice(['18-24', '25-29', '30-34', '35-39', '40-44'], n),
    'Sex': np.random.choice(['Male', 'Female'], n),
    'Race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n),
    'PhysicalHealth': np.random.randint(0, 31, n),
    'MentalHealth': np.random.randint(0, 31, n),
    'SleepTime': np.random.randint(4, 13, n),
    'Smoking': np.random.choice(['Yes', 'No'], n),
    'AlcoholDrinking': np.random.choice(['Yes', 'No'], n),
    'Stroke': np.random.choice(['Yes', 'No'], n),
    'Diabetic': np.random.choice(['Yes', 'No'], n),
    'DiffWalking': np.random.choice(['Yes', 'No'], n),
    'PhysicalActivity': np.random.choice(['Yes', 'No'], n),
    'GenHealth': np.random.choice(['Poor', 'Fair', 'Good', 'Very good', 'Excellent'], n),
    'HeartDisease': np.random.choice([0, 1], n)
})

# One-hot encode categorical columns
X = pd.get_dummies(df.drop("HeartDisease", axis=1))
y = df["HeartDisease"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')

print("✅ Model and features saved.")
