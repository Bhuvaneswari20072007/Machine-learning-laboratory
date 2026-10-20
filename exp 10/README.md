# Study Time → Exam Score Prediction
### Machine Learning Laboratory Mini-Project (Supervised Multiple Linear Regression)

---

## 1. Project Title
**Study Time → Exam Score Prediction System**

## 2. Problem Statement
Academic performance is influenced by daily study consistency, lecture attendance, prior academic foundation, and adequate sleep routine. Educational institutions and students often lack an empirical, quantitative mechanism to predict potential exam outcomes before the final examination.

This project addresses this problem by formulating a **Supervised Multiple Linear Regression Model** that learns mathematical relationships between four key academic & lifestyle metrics and the resulting examination score, enabling data-driven academic forecasting and personalized guidance.

---

## 3. Project Objectives
- Build an end-to-end, academic Machine Learning laboratory web application.
- Train and evaluate a **Multiple Linear Regression** model using scikit-learn.
- Demonstrate all seven stages of the Machine Learning lifecycle:
  1. Dataset loading & input overview
  2. Data cleaning & preprocessing
  3. Exploratory Data Analysis (EDA) with scatter plots and correlation matrices
  4. Model training (80/20 train-test split) and mathematical weight formulation
  5. Interactive real-time Exam Score prediction with strict input validation
  6. Model evaluation using standard regression metrics (**MAE, MSE, RMSE, R² Score**)
  7. Persistent prediction logging using an embedded **SQLite3** database
- Ensure complete offline execution with no external API dependencies.

---

## 4. Technologies Used
- **Programming Language:** Python 3.x
- **Web Framework:** Flask
- **Frontend:** HTML5, CSS3, JavaScript (ES6+ Vanilla)
- **Data Manipulation & Computation:** Pandas, NumPy
- **Machine Learning Library:** Scikit-learn (`LinearRegression`, `train_test_split`, `metrics`)
- **Data Visualization:** Matplotlib (Agg backend)
- **Model Serialization:** Joblib
- **Database:** SQLite3
- **IDE / Environment:** Visual Studio Code (Windows/Linux/macOS)

---

## 5. Machine Learning Type & Variable Formulation
- **Learning Paradigm:** Supervised Learning
- **Problem Type:** Multiple Linear Regression (Continuous Numerical Output)
- **Input Features ($X \in \mathbb{R}^{n \times 4}$):**
  1. `Study_Hours`: Daily study duration (in hours, Range: $0 - 24$)
  2. `Attendance`: Classroom lecture attendance rate (in %, Range: $0 - 100\%$)
  3. `Previous_Score`: Score obtained in prior examinations (in marks, Range: $0 - 100$)
  4. `Sleep_Hours`: Nightly average sleep duration (in hours, Range: $0 - 24$)
- **Target Variable ($y \in \mathbb{R}^n$):**
  - `Exam_Score`: Final numerical examination grade (Continuous, Range: $0 - 100$)

---

## 6. Dataset Description
- **File Location:** `data/student_exam_data.csv`
- **Total Records:** 200 Student Samples
- **Number of Columns:** 5
- **Reproducibility:** Seed fixed at `random_state=42`
- **Sample Records:**

| Study_Hours | Attendance (%) | Previous_Score | Sleep_Hours | Exam_Score (Target) |
|:-----------:|:--------------:|:--------------:|:-----------:|:-------------------:|
| 4.4         | 85.0           | 76.9           | 9.4         | 83.5                |
| 9.6         | 63.3           | 84.5           | 8.9         | 94.1                |
| 7.6         | 66.3           | 80.9           | 8.7         | 87.9                |
| 6.4         | 95.0           | 79.9           | 6.2         | 88.3                |
| 2.4         | 83.7           | 65.6           | 5.8         | 62.8                |

---

## 7. Machine Learning Workflow
```
Problem Definition
       ↓
Data Collection (data/student_exam_data.csv)
       ↓
Data Preprocessing (Zero Nulls, Matrix Separation)
       ↓
Exploratory Data Analysis (EDA Scatter Plots & Heatmap)
       ↓
Train-Test Split (80% Train : 20% Test)
       ↓
Model Training (scikit-learn Linear Regression)
       ↓
Model Evaluation (MAE, MSE, RMSE, R² Score)
       ↓
Prediction Inference (Trained Model in models/exam_score_model.pkl)
       ↓
Prediction History Logging (database/predictions.db)
```

---

## 8. Project Directory Structure
```
ExamScorePrediction/
│
├── app.py                      # Main Flask application and REST API endpoints
├── requirements.txt            # Project dependencies
├── README.md                   # Complete laboratory documentation
│
├── data/
│   └── student_exam_data.csv   # Local reproducible student dataset (200 records)
│
├── models/
│   └── exam_score_model.pkl    # Serialized scikit-learn Linear Regression model
│
├── ml/
│   ├── __init__.py
│   ├── preprocessing.py        # Data generation, cleaning, and descriptive statistics
│   ├── train_model.py          # 80/20 train-test split, OLS training, equation builder
│   ├── evaluate_model.py       # MAE, MSE, RMSE, R² metrics & plot generation
│   └── predict.py              # Input validation, inference, and prediction engine
│
├── database/
│   ├── __init__.py
│   ├── db_helper.py            # SQLite3 database connection and query manager
│   └── predictions.db          # Persistent SQLite database table
│
├── static/
│   ├── css/
│   │   └── style.css           # Academic laboratory theme styling
│   ├── js/
│   │   └── script.js           # Interactive frontend API controller
│   └── plots/                  # Generated high-resolution visualization figures
│       ├── study_vs_exam.png
│       ├── attendance_vs_exam.png
│       ├── previous_vs_exam.png
│       ├── sleep_vs_exam.png
│       ├── correlation_heatmap.png
│       ├── actual_vs_predicted.png
│       └── residual_plot.png
│
└── templates/
    └── index.html              # Screenshot-friendly academic web dashboard (Steps 1–7)
```

---

## 9. Mathematical Formulation

### Multiple Linear Regression Equation:
$$\hat{y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4$$

Where:
- $\hat{y}$ = Predicted Exam Score
- $\beta_0$ = Intercept (Baseline constant)
- $\beta_1$ = Coefficient for `Study_Hours`
- $\beta_2$ = Coefficient for `Attendance`
- $\beta_3$ = Coefficient for `Previous_Score`
- $\beta_4$ = Coefficient for `Sleep_Hours`

---

## 10. Installation & Setup Instructions

### Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.13 supported)
- Visual Studio Code

### Step 1: Clone or Open Project in VS Code
Open the project directory in Visual Studio Code.

### Step 2: Create and Activate Virtual Environment (Optional but Recommended)
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask Application
```bash
python app.py
```

### Step 5: Open in Local Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 11. How to Use the Application
1. **Overview & Summary:** Review the top metrics displaying 200 records, 4 features, 80/20 split, and active model status.
2. **Step 1 (Dataset):** View the first 10 rows of raw student academic data and feature descriptions.
3. **Step 2 (Preprocessing):** Inspect missing value checks (0 missing) and the statistical summary (mean, std, min, max).
4. **Step 3 (EDA):** Analyze the primary **Study Hours vs Exam Score** scatter plot, attendance trend, and correlation heatmap.
5. **Step 4 (Model Training):** Observe learned model coefficients ($\beta$) and the fitted regression equation. Click **Retrain Model** to re-execute training.
6. **Step 5 (Prediction):** Enter student details (or click quick presets like *High Achiever* or *Default Student*) and click **Predict Exam Score**. The system displays the predicted score and logs it to SQLite.
7. **Step 6 (Model Evaluation):** Review the calculated **MAE, MSE, RMSE, and R² Score**, along with the **Actual vs. Predicted** test scatter plot.
8. **Step 7 (Prediction History):** Inspect the persistent SQLite table of queries. Click **Refresh History** or **Clear History** as needed.

---

## 12. Model Evaluation Results (Testing Dataset)

| Metric | Metric Name | Calculated Value | Formula |
|:------:|:-----------:|:----------------:|:-------:|
| **MAE** | Mean Absolute Error | `1.7343` | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ |
| **MSE** | Mean Squared Error | `4.4548` | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ |
| **RMSE** | Root Mean Squared Error | `2.1106` | $\sqrt{\text{MSE}}$ |
| **R² Score** | Coefficient of Determination | `0.9901` | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ |

*Note: All evaluation metrics are calculated dynamically from the 20% unseen test split ($N=40$).*

---

## 13. Laboratory Record Screenshot Checklist

| Figure | Section | Description to Capture |
|:------:|:-------:|:-----------------------|
| **Fig 1** | VS Code | Project directory tree and terminal running `python app.py` |
| **Fig 2** | Step 1 | Dataset overview and sample 10-row student data table |
| **Fig 3** | Step 2 | Data preprocessing, missing value check, and descriptive statistics |
| **Fig 4** | Step 3 | Primary EDA scatter plot: Study Hours vs. Exam Score with trendline |
| **Fig 5** | Step 3 | Secondary EDA plots and correlation matrix heatmap |
| **Fig 6** | Step 4 | Model training completion status banner and mathematical equation |
| **Fig 7** | Step 4 | Learned parameter weights ($\beta_0$ to $\beta_4$) table |
| **Fig 8** | Step 5 | Student parameter input form with custom values |
| **Fig 9** | Step 5 | Predicted Exam Score card (e.g., `71.18 / 100`) and score meter |
| **Fig 10** | Step 6 | Model evaluation metric cards (MAE, MSE, RMSE, R² Score) |
| **Fig 11** | Step 6 | Actual vs. Predicted Exam Score scatter plot ($y = x$ line) |
| **Fig 12** | Step 7 | SQLite prediction history table with recorded logs |

---

## 14. Example Prediction Test Cases

### Case 1: Zero Study / Low Effort Student (Clear Fail)
- **Inputs:** Study Hours: `0.0`, Attendance: `50.0%`, Previous Score: `45.0`, Sleep Hours: `8.0`
- **Predicted Exam Score:** `32.85 marks` *(🔴 Fail / Critical - Below 40)*

### Case 2: Standard Average Student
- **Inputs:** Study Hours: `5.0`, Attendance: `85.0%`, Previous Score: `70.0`, Sleep Hours: `7.0`
- **Predicted Exam Score:** `71.18 marks` *(🔵 First Class / Good)*

### Case 3: High Performing Student (Distinction)
- **Inputs:** Study Hours: `8.5`, Attendance: `96.0%`, Previous Score: `90.0`, Sleep Hours: `8.0`
- **Predicted Exam Score:** `97.61 marks` *(🟢 Top Distinction)*

---

## 15. Future Enhancements
- Inclusion of additional features such as assignment completion rates and extracurricular activities.
- Comparison with polynomial regression and ensemble regression algorithms (Random Forest Regressor, Gradient Boosting).
- PDF lab-report export directly from the web interface.

---

## 16. Final Result
> **Result:** The Study Time → Exam Score Prediction system was successfully implemented using Machine Learning. The system preprocesses student data, performs exploratory data analysis, trains a Linear Regression model, evaluates the model using MAE, MSE, RMSE and R² Score, predicts an expected exam score for new student inputs, and stores prediction history using SQLite.
