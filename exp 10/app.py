import os
import sys

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, jsonify

# Import ML and Database modules
from ml.preprocessing import (
    load_data, 
    get_dataset_overview, 
    get_preprocessing_details, 
    generate_default_dataset
)
from ml.train_model import train_linear_regression, load_trained_model, MODEL_PATH
from ml.evaluate_model import evaluate_model_performance, generate_eda_plots
from ml.predict import predict_exam_score, validate_input
from database.db_helper import init_db, get_all_predictions, clear_history

app = Flask(__name__)
app.config['SECRET_KEY'] = 'academic-ml-exam-score-prediction-key'


def initialize_project_assets():
    """
    Ensures dataset, trained model, evaluation plots, and SQLite database are initialized on startup.
    """
    print("--- Initializing ML Project Assets ---")
    # 1. Database
    init_db()
    
    # 2. Dataset
    load_data()
    
    # 3. Model
    if not os.path.exists(MODEL_PATH):
        print("Training baseline Linear Regression model...")
        train_linear_regression()
        
    # 4. Plots & Metrics
    print("Generating EDA & Evaluation plots...")
    evaluate_model_performance()
    print("--- Initialization Complete ---")


# ==========================================
# Frontend Routes
# ==========================================
@app.route('/')
def index():
    """
    Renders the main Academic Machine Learning Dashboard.
    """
    return render_template('index.html')


# ==========================================
# REST API Endpoints
# ==========================================
@app.route('/api/summary', methods=['GET'])
def api_summary():
    """
    High-level dashboard metrics for the top summary cards.
    """
    try:
        df = load_data()
        return jsonify({
            'success': True,
            'dataset_records': int(len(df)),
            'features_count': 4,
            'model_name': 'Linear Regression',
            'train_test_split': '80 / 20',
            'target_variable': 'Exam Score',
            'model_status': 'Trained & Active'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dataset', methods=['GET'])
def api_dataset():
    """
    STEP 1: Returns dataset details, feature names, target, and sample table rows.
    """
    try:
        overview = get_dataset_overview()
        return jsonify({'success': True, 'data': overview})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/preprocess', methods=['GET'])
def api_preprocess():
    """
    STEP 2: Returns missing value checks and descriptive statistics.
    """
    try:
        details = get_preprocessing_details()
        return jsonify({'success': True, 'data': details})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/eda', methods=['GET'])
def api_eda():
    """
    STEP 3: Returns EDA plot URLs and correlation insights.
    """
    try:
        plots = {
            'study_vs_exam': '/static/plots/study_vs_exam.png',
            'attendance_vs_exam': '/static/plots/attendance_vs_exam.png',
            'previous_vs_exam': '/static/plots/previous_vs_exam.png',
            'sleep_vs_exam': '/static/plots/sleep_vs_exam.png',
            'correlation_heatmap': '/static/plots/correlation_heatmap.png'
        }
        return jsonify({'success': True, 'plots': plots})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['GET', 'POST'])
def api_train():
    """
    STEP 4: Trains or retrains the Linear Regression model and returns learned parameters.
    """
    try:
        summary = train_linear_regression()
        # Re-evaluate and update evaluation plots with newly trained model
        evaluate_model_performance()
        return jsonify({'success': True, 'training': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    STEP 5: Predicts expected Exam Score given user inputs and logs to SQLite database.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            data = request.form

        study_hours = data.get('study_hours')
        attendance = data.get('attendance')
        previous_score = data.get('previous_score')
        sleep_hours = data.get('sleep_hours')

        # Check for missing required fields
        if study_hours is None or attendance is None or previous_score is None or sleep_hours is None:
            return jsonify({
                'success': False, 
                'error': 'Please provide all four inputs: Study Hours, Attendance, Previous Score, and Sleep Hours.'
            }), 400

        result = predict_exam_score(study_hours, attendance, previous_score, sleep_hours, save_to_db=True)
        if not result['success']:
            return jsonify(result), 400

        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/evaluate', methods=['GET'])
def api_evaluate():
    """
    STEP 6: Evaluates the model on testing data and returns MAE, MSE, RMSE, R² Score, and plot URLs.
    """
    try:
        evaluation = evaluate_model_performance()
        return jsonify({'success': True, 'evaluation': evaluation})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def api_history():
    """
    STEP 7: Returns logged prediction history from SQLite database.
    """
    try:
        history = get_all_predictions()
        return jsonify({'success': True, 'history': history, 'total_records': len(history)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    """
    Clears all saved SQLite predictions.
    """
    try:
        clear_history()
        return jsonify({'success': True, 'message': 'Prediction history cleared successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Auto-initialize on module load
initialize_project_assets()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
