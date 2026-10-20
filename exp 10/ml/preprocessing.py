import os
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'student_exam_data.csv')

def generate_default_dataset(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic student exam score dataset with reproducible random seed.
    Features:
        - Study_Hours: Study hours per day (range ~0 to 11 hours)
        - Attendance: Attendance percentage (range ~35% to 100%)
        - Previous_Score: Past examination score (range ~25 to 98)
        - Sleep_Hours: Average sleep hours per day (range ~4.5 to 9.5 hours)
    Target:
        - Exam_Score: Numerical final examination score (range ~0 to 100)
    """
    np.random.seed(random_state)
    
    # 1. Study Hours (between 0.0 and 11.0 hours)
    study_hours = np.round(np.random.uniform(0.0, 11.0, n_samples), 1)
    
    # 2. Attendance percentage (between 35.0% and 99.0%)
    attendance = np.round(np.random.uniform(35.0, 99.0, n_samples), 1)
    
    # 3. Previous Exam Score (between 25.0 and 98.0)
    previous_score = np.round(
        20.0 + (study_hours * 4.2) + (attendance * 0.30) + np.random.normal(0, 3.5, n_samples), 
        1
    )
    previous_score = np.clip(previous_score, 25.0, 98.0)
    
    # 4. Sleep Hours (between 4.5 and 9.5 hours)
    sleep_hours = np.round(np.random.uniform(4.5, 9.5, n_samples), 1)
    
    # 5. Target Exam Score: Realistic Linear Regression formulation where Study Hours is the primary determinant
    noise = np.random.normal(0, 2.0, n_samples)
    exam_score = (
        1.20 
        + (study_hours * 4.85) 
        + (attendance * 0.20) 
        + (previous_score * 0.35) 
        + (sleep_hours * 0.65) 
        + noise
    )
    # Clip exam score to a realistic max range [0, 100] and round to 1 decimal place
    exam_score = np.round(np.clip(exam_score, 10.0, 100.0), 1)
    
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Score': previous_score,
        'Sleep_Hours': sleep_hours,
        'Exam_Score': exam_score
    })
    
    # Ensure directory exists and save CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_data() -> pd.DataFrame:
    """
    Loads dataset from CSV file. If file does not exist, generates it first.
    """
    if not os.path.exists(DATA_PATH):
        return generate_default_dataset()
    return pd.read_csv(DATA_PATH)


def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Returns count of missing values per column.
    """
    return df.isnull().sum().to_dict()


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Preprocesses the dataset:
    - Checks and records missing values before and after
    - Imputes numerical missing values with column mean if any exist
    - Ensures proper numeric datatypes
    Returns: (cleaned_dataframe, preprocessing_summary_dict)
    """
    before_missing = df.isnull().sum().to_dict()
    total_missing_before = int(df.isnull().sum().sum())
    
    # Make a copy for processing
    df_clean = df.copy()
    
    # Handle missing values if any
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            mean_val = df_clean[col].mean()
            df_clean[col].fillna(mean_val, inplace=True)
    
    after_missing = df_clean.isnull().sum().to_dict()
    total_missing_after = int(df_clean.isnull().sum().sum())
    
    summary = {
        'total_rows': int(len(df_clean)),
        'total_columns': int(df_clean.shape[1]),
        'missing_before': before_missing,
        'missing_after': after_missing,
        'total_missing_before': total_missing_before,
        'total_missing_after': total_missing_after,
        'features': ['Study_Hours', 'Attendance', 'Previous_Score', 'Sleep_Hours'],
        'target': 'Exam_Score',
        'shape': list(df_clean.shape)
    }
    
    return df_clean, summary


def get_dataset_overview() -> dict:
    """
    Provides full dataset overview for Step 1 of the web dashboard.
    """
    df = load_data()
    head_records = df.head(10).to_dict(orient='records')
    columns = list(df.columns)
    
    return {
        'total_records': int(len(df)),
        'num_features': 4,
        'feature_names': ['Study_Hours', 'Attendance', 'Previous_Score', 'Sleep_Hours'],
        'target_name': 'Exam_Score',
        'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
        'columns': columns,
        'sample_data': head_records,
        'description': 'The dataset contains student academic metrics including daily study hours, attendance rate, past test performance, and sleep duration to predict overall exam score.'
    }


def get_preprocessing_details() -> dict:
    """
    Provides detailed statistics and missing value details for Step 2.
    """
    df = load_data()
    df_clean, summary = preprocess_data(df)
    
    # Generate descriptive statistics table
    stats_df = df_clean.describe().round(2).reset_index()
    stats_dict = stats_df.to_dict(orient='records')
    
    summary['statistics'] = stats_dict
    return summary


def get_features_and_target(df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataframe into X (input features) and y (target variable).
    """
    if df is None:
        df = load_data()
    df_clean, _ = preprocess_data(df)
    
    X = df_clean[['Study_Hours', 'Attendance', 'Previous_Score', 'Sleep_Hours']]
    y = df_clean['Exam_Score']
    return X, y


if __name__ == '__main__':
    # Test generation and preprocessing
    df = generate_default_dataset()
    print("Dataset generated successfully. Shape:", df.shape)
    print(df.head())
    print("\nDataset Overview:")
    print(get_dataset_overview())
