import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from ml.preprocessing import load_data, preprocess_data, get_features_and_target

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'exam_score_model.pkl')


def train_linear_regression(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Trains a Linear Regression model on the dataset using an 80/20 train/test split.
    Saves the trained model to models/exam_score_model.pkl using Joblib.
    Returns comprehensive details about the training process, coefficients, and equation.
    """
    # 1. Load and preprocess data
    X, y = get_features_and_target()
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 3. Initialize and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Ensure models directory exists and save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    # 5. Extract learned coefficients & intercept
    feature_names = list(X.columns)
    coefficients = {feature: round(float(coef), 4) for feature, coef in zip(feature_names, model.coef_)}
    intercept = round(float(model.intercept_), 4)
    
    # 6. Formulate readable mathematical equation
    terms = [f"({coef:+.4f} * {feat})" for feat, coef in coefficients.items()]
    equation = f"Exam_Score = {intercept:.4f} " + " ".join(terms)
    
    training_summary = {
        'model_name': 'Linear Regression (scikit-learn)',
        'algorithm': 'Ordinary Least Squares (OLS)',
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'total_samples': int(len(X)),
        'split_ratio': f"{int((1-test_size)*100)}% Train / {int(test_size*100)}% Test",
        'feature_names': feature_names,
        'target_variable': 'Exam_Score',
        'intercept': intercept,
        'coefficients': coefficients,
        'equation': equation,
        'model_saved_path': 'models/exam_score_model.pkl',
        'status': 'Model Training Completed Successfully'
    }
    
    return training_summary


def load_trained_model():
    """
    Loads the serialized model from disk. If not found, trains a new one.
    """
    if not os.path.exists(MODEL_PATH):
        train_linear_regression()
    return joblib.load(MODEL_PATH)


if __name__ == '__main__':
    result = train_linear_regression()
    print("Training Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")
