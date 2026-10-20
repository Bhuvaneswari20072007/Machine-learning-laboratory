import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server generation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.preprocessing import get_features_and_target, load_data
from ml.train_model import load_trained_model

STATIC_PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')


def ensure_plots_dir():
    os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)


def set_dark_neon_plot_style(ax, fig):
    """
    Applies a dark blue/black theme with neon accents to matplotlib plots.
    """
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0b0f19')
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle=':', alpha=0.35, color='#334155')


def generate_eda_plots():
    """
    Generates all Exploratory Data Analysis (EDA) plots with dark neon theme and saves them to static/plots/.
    """
    ensure_plots_dir()
    df = load_data()
    
    # 1. Study Hours vs Exam Score (Main Highlight Graph with Trendline)
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    set_dark_neon_plot_style(ax, fig)
    
    ax.scatter(df['Study_Hours'], df['Exam_Score'], color='#00f0ff', alpha=0.85, edgecolors='#38bdf8', s=45, label='Student Data')
    
    # Add neon pink trend line
    m, b = np.polyfit(df['Study_Hours'], df['Exam_Score'], 1)
    x_line = np.linspace(df['Study_Hours'].min(), df['Study_Hours'].max(), 100)
    ax.plot(x_line, m * x_line + b, color='#ff007f', linewidth=2.5, linestyle='--', label=f'Trend: y = {m:.2f}x + {b:.2f}')
    
    ax.set_title('Study Hours vs. Exam Score (EDA Focus)', fontsize=13, fontweight='bold', pad=12, color='#f8fafc')
    ax.set_xlabel('Daily Study Hours (hrs)', fontsize=11, fontweight='600', color='#00ff9d')
    ax.set_ylabel('Exam Score (0 - 100)', fontsize=11, fontweight='600', color='#00f0ff')
    leg = ax.legend(loc='upper left', frameon=True, facecolor='#1e293b', edgecolor='#334155')
    for text in leg.get_texts():
        text.set_color('#f8fafc')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PLOTS_DIR, 'study_vs_exam.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Attendance vs Exam Score
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    set_dark_neon_plot_style(ax, fig)
    ax.scatter(df['Attendance'], df['Exam_Score'], color='#00ff9d', alpha=0.85, edgecolors='#10b981', s=40)
    m_att, b_att = np.polyfit(df['Attendance'], df['Exam_Score'], 1)
    x_att = np.linspace(df['Attendance'].min(), df['Attendance'].max(), 100)
    ax.plot(x_att, m_att * x_att + b_att, color='#05ffa1', linewidth=2.0, linestyle='--')
    ax.set_title('Attendance vs. Exam Score', fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax.set_xlabel('Attendance Percentage (%)', fontsize=10, fontweight='600', color='#00ff9d')
    ax.set_ylabel('Exam Score', fontsize=10, fontweight='600', color='#00f0ff')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PLOTS_DIR, 'attendance_vs_exam.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # 3. Previous Score vs Exam Score
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    set_dark_neon_plot_style(ax, fig)
    ax.scatter(df['Previous_Score'], df['Exam_Score'], color='#ff007f', alpha=0.85, edgecolors='#f43f5e', s=40)
    m_prev, b_prev = np.polyfit(df['Previous_Score'], df['Exam_Score'], 1)
    x_prev = np.linspace(df['Previous_Score'].min(), df['Previous_Score'].max(), 100)
    ax.plot(x_prev, m_prev * x_prev + b_prev, color='#ff2a85', linewidth=2.0, linestyle='--')
    ax.set_title('Previous Score vs. Exam Score', fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax.set_xlabel('Previous Exam Score', fontsize=10, fontweight='600', color='#ff007f')
    ax.set_ylabel('Exam Score', fontsize=10, fontweight='600', color='#00f0ff')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PLOTS_DIR, 'previous_vs_exam.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # 4. Sleep Hours vs Exam Score
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    set_dark_neon_plot_style(ax, fig)
    ax.scatter(df['Sleep_Hours'], df['Exam_Score'], color='#fbbf24', alpha=0.85, edgecolors='#f59e0b', s=40)
    m_slp, b_slp = np.polyfit(df['Sleep_Hours'], df['Exam_Score'], 1)
    x_slp = np.linspace(df['Sleep_Hours'].min(), df['Sleep_Hours'].max(), 100)
    ax.plot(x_slp, m_slp * x_slp + b_slp, color='#f59e0b', linewidth=2.0, linestyle='--')
    ax.set_title('Sleep Hours vs. Exam Score', fontsize=12, fontweight='bold', pad=10, color='#f8fafc')
    ax.set_xlabel('Sleep Hours (hrs/night)', fontsize=10, fontweight='600', color='#fbbf24')
    ax.set_ylabel('Exam Score', fontsize=10, fontweight='600', color='#00f0ff')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PLOTS_DIR, 'sleep_vs_exam.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # 5. Correlation Heatmap
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0b0f19')
    cax = ax.matshow(corr, cmap='magma', vmin=-1, vmax=1)
    cbar = fig.colorbar(cax, shrink=0.8)
    cbar.ax.tick_params(colors='#94a3b8')
    
    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    labels = ['Study Hours', 'Attendance', 'Prev Score', 'Sleep Hours', 'Exam Score']
    ax.set_xticklabels(labels, rotation=35, ha='left', fontsize=9, fontweight='600', color='#00f0ff')
    ax.set_yticklabels(labels, fontsize=9, fontweight='600', color='#00f0ff')
    
    # Print correlation values on cells
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            text_color = '#ffffff' if abs(val) > 0.4 else '#cbd5e1'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontweight='bold', fontsize=10)
            
    ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold', pad=25, color='#f8fafc')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PLOTS_DIR, 'correlation_heatmap.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def evaluate_model_performance(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Evaluates the trained Linear Regression model on unseen testing data (20%).
    Computes real MAE, MSE, RMSE, and R2 Score.
    Generates Actual vs. Predicted and Residual scatter plots in dark neon style.
    """
    ensure_plots_dir()
    
    # 1. Load data and split
    X, y = get_features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 2. Load model and predict on test set
    model = load_trained_model()
    y_pred = model.predict(X_test)
    
    # 3. Calculate actual evaluation metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    # 4. Generate Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=150)
    set_dark_neon_plot_style(ax, fig)
    ax.scatter(y_test, y_pred, color='#00f0ff', alpha=0.85, edgecolors='#38bdf8', s=55, label='Testing Samples (20%)')
    
    # Ideal 45-degree diagonal line in neon pink
    min_val = min(y_test.min(), y_pred.min()) - 2
    max_val = max(y_test.max(), y_pred.max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], color='#ff007f', linestyle='--', linewidth=2.2, label='Ideal Fit (y = x)')
    
    ax.set_title('Actual vs. Predicted Exam Scores (Test Set)', fontsize=12, fontweight='bold', pad=12, color='#f8fafc')
    ax.set_xlabel('Actual Exam Score (True y)', fontsize=10, fontweight='600', color='#00ff9d')
    ax.set_ylabel('Predicted Exam Score (Predicted y)', fontsize=10, fontweight='600', color='#00f0ff')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    leg = ax.legend(loc='upper left', frameon=True, facecolor='#1e293b', edgecolor='#334155')
    for text in leg.get_texts():
        text.set_color('#f8fafc')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PLOTS_DIR, 'actual_vs_predicted.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # 5. Generate Residuals Plot (Errors vs Predicted)
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    set_dark_neon_plot_style(ax, fig)
    ax.scatter(y_pred, residuals, color='#00ff9d', alpha=0.85, edgecolors='#10b981', s=50)
    ax.axhline(0, color='#ff007f', linestyle='--', linewidth=2.0, label='Zero Error Baseline')
    ax.set_title('Residuals Analysis (Errors vs. Predicted Score)', fontsize=12, fontweight='bold', pad=12, color='#f8fafc')
    ax.set_xlabel('Predicted Exam Score', fontsize=10, fontweight='600', color='#00f0ff')
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=10, fontweight='600', color='#ff007f')
    leg = ax.legend(loc='upper right', frameon=True, facecolor='#1e293b', edgecolor='#334155')
    for text in leg.get_texts():
        text.set_color('#f8fafc')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PLOTS_DIR, 'residual_plot.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    # Make sure EDA plots are also up to date
    generate_eda_plots()

    return {
        'test_samples': int(len(y_test)),
        'train_samples': int(len(y_train)),
        'metrics': {
            'mae': round(mae, 4),
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'r2_score': round(r2, 4)
        },
        'plots': {
            'actual_vs_predicted': '/static/plots/actual_vs_predicted.png',
            'residual_plot': '/static/plots/residual_plot.png',
            'study_vs_exam': '/static/plots/study_vs_exam.png',
            'attendance_vs_exam': '/static/plots/attendance_vs_exam.png',
            'previous_vs_exam': '/static/plots/previous_vs_exam.png',
            'sleep_vs_exam': '/static/plots/sleep_vs_exam.png',
            'correlation_heatmap': '/static/plots/correlation_heatmap.png'
        }
    }


if __name__ == '__main__':
    metrics = evaluate_model_performance()
    print("Dark Neon Evaluation Plots generated successfully!")
    print("Metrics:", metrics['metrics'])
