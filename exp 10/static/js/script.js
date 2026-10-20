/**
 * Study Time -> Exam Score Prediction | Frontend Controller
 * Handles asynchronous API communication, UI updates, prediction processing, and history rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    initDashboard();
    setupNavigationHighlighting();
});

function initDashboard() {
    loadSummary();
    loadDatasetOverview();
    loadPreprocessing();
    loadTrainingDetails();
    loadEvaluation();
    fetchPredictionHistory();
}

/* ==========================================================================
   Navigation Scroll Active State
   ========================================================================== */
function setupNavigationHighlighting() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        const scrollPosition = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

/* ==========================================================================
   Summary API Loader
   ========================================================================== */
async function loadSummary() {
    try {
        const response = await fetch('/api/summary');
        const res = await response.json();
        if (res.success) {
            document.getElementById('sum-records').textContent = res.dataset_records;
            document.getElementById('sum-features').textContent = res.features_count;
            document.getElementById('sum-model').textContent = res.model_name;
            document.getElementById('sum-split').textContent = res.train_test_split;
            document.getElementById('sum-target').textContent = res.target_variable;
            document.getElementById('sum-status').textContent = res.model_status;
        }
    } catch (err) {
        console.error('Error loading summary:', err);
    }
}

/* ==========================================================================
   Step 1: Dataset Overview Loader
   ========================================================================== */
async function loadDatasetOverview() {
    try {
        const response = await fetch('/api/dataset');
        const json = await response.json();
        if (json.success) {
            const data = json.data;
            document.getElementById('dataset-total-rows').textContent = data.total_records;
            document.getElementById('dataset-shape').textContent = `(${data.shape.rows}, ${data.shape.columns})`;

            const tbody = document.getElementById('dataset-tbody');
            tbody.innerHTML = '';

            data.sample_data.forEach((row, idx) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${idx + 1}</strong></td>
                    <td>${Number(row.Study_Hours).toFixed(1)}</td>
                    <td>${Number(row.Attendance).toFixed(1)}%</td>
                    <td>${Number(row.Previous_Score).toFixed(1)}</td>
                    <td>${Number(row.Sleep_Hours).toFixed(1)}</td>
                    <td class="target-col">${Number(row.Exam_Score).toFixed(1)}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error('Error loading dataset overview:', err);
    }
}

/* ==========================================================================
   Step 2: Preprocessing Loader
   ========================================================================== */
async function loadPreprocessing() {
    try {
        const response = await fetch('/api/preprocess');
        const json = await response.json();
        if (json.success) {
            const summary = json.data;
            document.getElementById('missing-before-val').textContent = `${summary.total_missing_before} Missing`;
            document.getElementById('missing-after-val').textContent = `${summary.total_missing_after} Missing`;
            document.getElementById('x-shape').textContent = `(${summary.total_rows}, 4)`;
            document.getElementById('y-shape').textContent = `(${summary.total_rows},)`;

            const statsTbody = document.getElementById('stats-tbody');
            statsTbody.innerHTML = '';

            if (summary.statistics) {
                summary.statistics.forEach(statRow => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${statRow.index}</strong></td>
                        <td>${statRow.Study_Hours !== undefined ? statRow.Study_Hours : '--'}</td>
                        <td>${statRow.Attendance !== undefined ? statRow.Attendance : '--'}</td>
                        <td>${statRow.Previous_Score !== undefined ? statRow.Previous_Score : '--'}</td>
                        <td>${statRow.Sleep_Hours !== undefined ? statRow.Sleep_Hours : '--'}</td>
                        <td class="target-col">${statRow.Exam_Score !== undefined ? statRow.Exam_Score : '--'}</td>
                    `;
                    statsTbody.appendChild(tr);
                });
            }
        }
    } catch (err) {
        console.error('Error loading preprocessing:', err);
    }
}

/* ==========================================================================
   Step 4: Model Training Loader / Retrain
   ========================================================================== */
async function loadTrainingDetails() {
    try {
        const response = await fetch('/api/train');
        const json = await response.json();
        if (json.success) {
            renderTrainingDetails(json.training);
        }
    } catch (err) {
        console.error('Error loading training details:', err);
    }
}

async function retrainModel() {
    const btn = document.getElementById('btn-retrain');
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-icon">&#x21bb;</span> Retraining...`;

    try {
        const response = await fetch('/api/train', { method: 'POST' });
        const json = await response.json();
        if (json.success) {
            renderTrainingDetails(json.training);
            // Refresh evaluations and plots
            await loadEvaluation();
            showTemporaryNotification("Model retrained and evaluation metrics updated successfully!");
        }
    } catch (err) {
        console.error('Error retraining model:', err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="btn-icon">&#x21bb;</span> Retrain Model`;
    }
}

function renderTrainingDetails(trainData) {
    document.getElementById('train-samples').textContent = `${trainData.train_samples} (80%)`;
    document.getElementById('test-samples').textContent = `${trainData.test_samples} (20%)`;
    document.getElementById('training-status-text').innerHTML = `<strong>${trainData.status}</strong> &bull; Serialized to ${trainData.model_saved_path}`;

    // Display formatted equation
    const coef = trainData.coefficients;
    const intercept = trainData.intercept;
    const formattedEq = `Exam_Score = ${intercept} + (${coef.Study_Hours} &times; Study_Hours) + (${coef.Attendance} &times; Attendance) + (${coef.Previous_Score} &times; Previous_Score) + (${coef.Sleep_Hours} &times; Sleep_Hours)`;
    document.getElementById('model-equation-display').innerHTML = formattedEq;

    // Render parameters table
    const paramsTbody = document.getElementById('params-tbody');
    paramsTbody.innerHTML = `
        <tr>
            <td><strong>&beta;<sub>0</sub> (Intercept)</strong></td>
            <td>Base Constant</td>
            <td><strong>${intercept}</strong></td>
            <td>Baseline score when all input metrics are zero</td>
        </tr>
        <tr>
            <td><strong>&beta;<sub>1</sub></strong></td>
            <td>Study_Hours</td>
            <td><strong>${coef.Study_Hours >= 0 ? '+' : ''}${coef.Study_Hours}</strong></td>
            <td>Every +1 hour of daily study increases score by ~${coef.Study_Hours} marks</td>
        </tr>
        <tr>
            <td><strong>&beta;<sub>2</sub></strong></td>
            <td>Attendance</td>
            <td><strong>${coef.Attendance >= 0 ? '+' : ''}${coef.Attendance}</strong></td>
            <td>Every +1% class attendance increases score by ~${coef.Attendance} marks</td>
        </tr>
        <tr>
            <td><strong>&beta;<sub>3</sub></strong></td>
            <td>Previous_Score</td>
            <td><strong>${coef.Previous_Score >= 0 ? '+' : ''}${coef.Previous_Score}</strong></td>
            <td>Every +1 mark in prior exams contributes ~${coef.Previous_Score} marks</td>
        </tr>
        <tr>
            <td><strong>&beta;<sub>4</sub></strong></td>
            <td>Sleep_Hours</td>
            <td><strong>${coef.Sleep_Hours >= 0 ? '+' : ''}${coef.Sleep_Hours}</strong></td>
            <td>Every +1 hour of adequate nightly sleep adds ~${coef.Sleep_Hours} marks</td>
        </tr>
    `;
}

/* ==========================================================================
   Step 5: Prediction Handler
   ========================================================================== */
function applyPreset(study, attendance, previous, sleep) {
    document.getElementById('input-study').value = study;
    document.getElementById('input-attendance').value = attendance;
    document.getElementById('input-previous').value = previous;
    document.getElementById('input-sleep').value = sleep;

    // Trigger prediction automatically on preset click
    document.getElementById('btn-predict').click();
}

async function handlePrediction(event) {
    event.preventDefault();
    const errorDiv = document.getElementById('prediction-error');
    errorDiv.style.display = 'none';

    const studyHours = parseFloat(document.getElementById('input-study').value);
    const attendance = parseFloat(document.getElementById('input-attendance').value);
    const previousScore = parseFloat(document.getElementById('input-previous').value);
    const sleepHours = parseFloat(document.getElementById('input-sleep').value);

    // Client-side validation
    if (isNaN(studyHours) || studyHours < 0 || studyHours > 24) {
        showError("Study Hours must be between 0 and 24 hours.");
        return;
    }
    if (isNaN(attendance) || attendance < 0 || attendance > 100) {
        showError("Attendance must be between 0% and 100%.");
        return;
    }
    if (isNaN(previousScore) || previousScore < 0 || previousScore > 100) {
        showError("Previous Score must be between 0 and 100 marks.");
        return;
    }
    if (isNaN(sleepHours) || sleepHours < 0 || sleepHours > 24) {
        showError("Sleep Hours must be between 0 and 24 hours.");
        return;
    }

    const btn = document.getElementById('btn-predict');
    btn.disabled = true;
    btn.innerHTML = `<span class="btn-icon">&#x25B6;</span> Calculating Prediction...`;

    try {
        const payload = {
            study_hours: studyHours,
            attendance: attendance,
            previous_score: previousScore,
            sleep_hours: sleepHours
        };

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const res = await response.json();
        if (res.success) {
            const numScore = Number(res.predicted_score);
            const scoreStr = numScore.toFixed(2);
            document.getElementById('predicted-score-value').textContent = scoreStr;
            
            // Update meter & grade badge
            const percentage = Math.min(Math.max(numScore, 0), 100);
            document.getElementById('score-bar-fill').style.width = `${percentage}%`;
            updateGradeBadge(numScore);

            // Update status details
            document.getElementById('res-status-text').textContent = `Expected Exam Score calculated from trained Linear Regression model.`;
            document.getElementById('res-db-log').textContent = `Logged into SQLite Database (Record ID: #${res.db_id})`;
            document.getElementById('res-timestamp').textContent = res.timestamp;

            // Refresh prediction history table
            fetchPredictionHistory();
        } else {
            showError(res.error || "Failed to generate prediction.");
        }
    } catch (err) {
        showError("Server communication error. Please ensure the Flask app is running.");
        console.error('Prediction error:', err);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="btn-icon">&#x25B6;</span> Predict Exam Score`;
    }
}

function updateGradeBadge(score) {
    const badge = document.getElementById('grade-badge');
    const fill = document.getElementById('score-bar-fill');
    if (!badge) return;

    if (score < 40) {
        badge.className = 'perf-badge perf-fail';
        badge.textContent = `🔴 Fail / Critical (${score.toFixed(2)} Marks - Below 40)`;
        if (fill) fill.style.background = 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)';
    } else if (score < 60) {
        badge.className = 'perf-badge perf-pass';
        badge.textContent = `🟠 Average Pass (${score.toFixed(2)} Marks)`;
        if (fill) fill.style.background = 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)';
    } else if (score < 80) {
        badge.className = 'perf-badge perf-good';
        badge.textContent = `🔵 First Class / Good (${score.toFixed(2)} Marks)`;
        if (fill) fill.style.background = 'linear-gradient(90deg, #3b82f6 0%, #2563eb 100%)';
    } else {
        badge.className = 'perf-badge perf-excellent';
        badge.textContent = `🟢 Distinction / High Achiever (${score.toFixed(2)} Marks)`;
        if (fill) fill.style.background = 'linear-gradient(90deg, #10b981 0%, #059669 100%)';
    }
}

function showError(msg) {
    const errorDiv = document.getElementById('prediction-error');
    errorDiv.textContent = msg;
    errorDiv.style.display = 'block';
}

/* ==========================================================================
   Step 6: Model Evaluation Loader
   ========================================================================== */
async function loadEvaluation() {
    try {
        const response = await fetch('/api/evaluate');
        const json = await response.json();
        if (json.success) {
            const m = json.evaluation.metrics;
            document.getElementById('metric-mae').textContent = Number(m.mae).toFixed(4);
            document.getElementById('metric-mse').textContent = Number(m.mse).toFixed(4);
            document.getElementById('metric-rmse').textContent = Number(m.rmse).toFixed(4);
            document.getElementById('metric-r2').textContent = Number(m.r2_score).toFixed(4);

            // Add timestamp query parameter to bypass image caching when refreshed
            const cacheBust = `?t=${new Date().getTime()}`;
            const plots = json.evaluation.plots;
            if (plots.study_vs_exam) document.getElementById('plot-study-exam').src = plots.study_vs_exam + cacheBust;
            if (plots.attendance_vs_exam) document.getElementById('plot-attendance-exam').src = plots.attendance_vs_exam + cacheBust;
            if (plots.previous_vs_exam) document.getElementById('plot-previous-exam').src = plots.previous_vs_exam + cacheBust;
            if (plots.sleep_vs_exam) document.getElementById('plot-sleep-exam').src = plots.sleep_vs_exam + cacheBust;
            if (plots.correlation_heatmap) document.getElementById('plot-correlation-heatmap').src = plots.correlation_heatmap + cacheBust;
            if (plots.actual_vs_predicted) document.getElementById('plot-actual-pred').src = plots.actual_vs_predicted + cacheBust;
            if (plots.residual_plot) document.getElementById('plot-residuals').src = plots.residual_plot + cacheBust;
        }
    } catch (err) {
        console.error('Error loading evaluation:', err);
    }
}

/* ==========================================================================
   Step 7: SQLite Prediction History Loader
   ========================================================================== */
async function fetchPredictionHistory() {
    try {
        const response = await fetch('/api/history');
        const json = await response.json();
        if (json.success) {
            const history = json.history;
            document.getElementById('history-count').textContent = history.length;
            const tbody = document.getElementById('history-tbody');
            tbody.innerHTML = '';

            if (history.length === 0) {
                tbody.innerHTML = `<tr><td colspan="7" class="text-center" style="color: #64748b; padding: 1.5rem;">No predictions recorded yet. Use the prediction form in Step 5 to generate predictions.</td></tr>`;
                return;
            }

            history.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${item.id}</strong></td>
                    <td>${Number(item.study_hours).toFixed(1)} hrs</td>
                    <td>${Number(item.attendance).toFixed(1)}%</td>
                    <td>${Number(item.previous_score).toFixed(1)}</td>
                    <td>${Number(item.sleep_hours).toFixed(1)} hrs</td>
                    <td class="target-col"><strong>${Number(item.predicted_score).toFixed(2)}</strong></td>
                    <td><small style="color: #64748b;">${item.timestamp}</small></td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error('Error fetching history:', err);
    }
}

async function clearPredictionHistory() {
    if (!confirm("Are you sure you want to clear the entire SQLite prediction history?")) {
        return;
    }

    try {
        const response = await fetch('/api/clear-history', { method: 'POST' });
        const json = await response.json();
        if (json.success) {
            fetchPredictionHistory();
            showTemporaryNotification("Prediction history cleared successfully.");
        }
    } catch (err) {
        console.error('Error clearing history:', err);
    }
}

function showTemporaryNotification(msg) {
    const note = document.createElement('div');
    note.style.position = 'fixed';
    note.style.bottom = '20px';
    note.style.right = '20px';
    note.style.background = '#1e293b';
    note.style.color = '#ffffff';
    note.style.padding = '0.75rem 1.25rem';
    note.style.borderRadius = '8px';
    note.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    note.style.zIndex = '9999';
    note.style.fontSize = '0.9rem';
    note.style.fontWeight = '600';
    note.textContent = msg;

    document.body.appendChild(note);
    setTimeout(() => {
        note.remove();
    }, 3000);
}
