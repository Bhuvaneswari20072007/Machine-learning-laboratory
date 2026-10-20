import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
from datetime import datetime
from ml.train_model import load_trained_model, train_linear_regression
from database.db_helper import insert_prediction


def validate_input(study_hours, attendance, previous_score, sleep_hours):
    """
    Validates user input values against defined domain bounds.
    Rules:
    - Study Hours: 0 to 24
    - Attendance: 0 to 100
    - Previous Score: 0 to 100
    - Sleep Hours: 0 to 24
    """
    try:
        sh = float(study_hours)
        att = float(attendance)
        ps = float(previous_score)
        slp = float(sleep_hours)
    except (ValueError, TypeError):
        return False, "All inputs must be valid numeric values.", None

    if not (0.0 <= sh <= 24.0):
        return False, "Study Hours must be between 0 and 24 hours.", None
    if not (0.0 <= att <= 100.0):
        return False, "Attendance must be between 0% and 100%.", None
    if not (0.0 <= ps <= 100.0):
        return False, "Previous Score must be between 0 and 100 marks.", None
    if not (0.0 <= slp <= 24.0):
        return False, "Sleep Hours must be between 0 and 24 hours.", None

    return True, "", (sh, att, ps, slp)


def predict_exam_score(study_hours, attendance, previous_score, sleep_hours, save_to_db: bool = True) -> dict:
    """
    Predicts the expected exam score for a student using the trained Linear Regression model.
    """
    # 1. Input Validation
    is_valid, error_msg, clean_inputs = validate_input(
        study_hours, attendance, previous_score, sleep_hours
    )
    if not is_valid:
        return {'success': False, 'error': error_msg}

    sh, att, ps, slp = clean_inputs

    # 2. Load model
    model = load_trained_model()

    # 3. Create feature vector matching training columns
    input_df = pd.DataFrame([{
        'Study_Hours': sh,
        'Attendance': att,
        'Previous_Score': ps,
        'Sleep_Hours': slp
    }])

    # 4. Predict score
    raw_prediction = float(model.predict(input_df)[0])
    
    # 5. Bound realistically between 0.0 and 100.0
    predicted_score = round(float(np.clip(raw_prediction, 0.0, 100.0)), 2)

    # 6. Save to SQLite database
    db_id = None
    if save_to_db:
        db_id = insert_prediction(sh, att, ps, slp, predicted_score)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        'success': True,
        'predicted_score': predicted_score,
        'raw_prediction': round(raw_prediction, 4),
        'inputs': {
            'study_hours': sh,
            'attendance': att,
            'previous_score': ps,
            'sleep_hours': slp
        },
        'db_id': db_id,
        'timestamp': timestamp,
        'model_name': 'Linear Regression'
    }


if __name__ == '__main__':
    res = predict_exam_score(5.0, 85.0, 70.0, 7.0)
    print("Prediction test result:", res)
