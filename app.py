"""
Intracranial Aneurysm Growth Prediction System
Modern Medical Dashboard Style - Clean, Hierarchical, Low Cognitive Load
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ============ Page Config ============
st.set_page_config(
    page_title="Intracranial Aneurysm Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ Config ============
MODEL_DIR = Path("modeldata")
SCALER_DIR = Path("scalerdata")
SELECTED_FEATURES = ['HD2ND_Ratio', 'Inflow_Angle', 'Oscillatory_Shear_Index_Mean', 'Pressure_Mean']

ALL_SCALER_FEATURES = [
    'Age', 'Diameter', 'Width', 'Height', 'HorizDiam', 'MaxDiam', 'D2W_Ratio',
    'Volume', 'Neck_Surface', 'Neck_Diameter', 'HD2ND_Ratio', 'Inflow_Angle',
    'Aneurysm_Angle', 'Near_Diam_D1', 'Far_Diam_D2', 'Parent_Artery_Mean_Diam',
    'Parent_Artery_Length', 'Undulation_Index', 'Size_Ratio', 'Aspect_Ratio',
    'Nonsphericity_Index', 'V2NS_Ratio', 'Low_Shear_Area_Ratio',
    'Oscillatory_Shear_Index_Mean', 'Pressure_Mean',
    'Wall_Shear_Stress_Mean', 'Energy_Loss_Ratio'
]

FEATURE_INFO = {
    'HD2ND_Ratio': {
        'label': 'Horizontal Diameter to Neck Diameter',
        'help': 'The ratio of the horizontal diameter to the neck diameter of intracranial aneurysm.',
        'icon': '📏'
    },
    'Inflow_Angle': {
        'label': 'Inflow Angle',
        'help': 'The angle between the direction of blood flow and the direction of the diameter of intracranial aneurysm. (°)',
        'icon': '📐'
    },
    'Oscillatory_Shear_Index_Mean': {
        'label': 'OSI (Mean)',
        'help': 'Hemodynamic parameter of UKnow®: mean Oscillatory Shear Index of intracranial aneurysm.',
        'icon': '〰️'
    },
    'Pressure_Mean': {
        'label': 'Pressure (Mean)',
        'help': 'Hemodynamic parameter of UKnow®： mean Pressure.',
        'icon': '⭕'
    }
}

MODEL_PARAMS = {
    'class_weight': 'balanced',
    'colsample_bytree': 0.4,
    'learning_rate': 0.02,
    'max_depth': 3,
    'min_child_samples': 20,
    'n_estimators': 250,
    'num_leaves': 4,
    'reg_alpha': 0.5,
    'reg_lambda': 1.5,
    'subsample': 0.5,
    'random_state': 123,
    'oob_score': True
}


# ============ Load Resources ============
@st.cache_resource
def load_scaler():
    scaler_path = SCALER_DIR / "scaler.joblib"
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "lgb_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        if hasattr(model, 'feature_names_in_'):
            del model.feature_names_in_
        return model
    return None


scaler = load_scaler()
model = load_model()


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ============ Medical Dashboard CSS ============
st.markdown("""
<style>
    /* ===== Global Theme - Minty Medical ===== */
    :root {
        --primary: #26A69A;
        --primary-light: #80CBC4;
        --primary-dark: #00897B;
        --accent: #4DB6AC;
        --success: #66BB6A;
        --warning: #FFA726;
        --danger: #EF5350;
        --bg-main: #F5F7FA;
        --bg-card: #FFFFFF;
        --text-primary: #2D3748;
        --text-secondary: #718096;
        --text-muted: #A0AEC0;
        --border: #E2E8F0;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 30px rgba(0,0,0,0.12);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }

    /* ===== Main Background ===== */
    .stApp {
        background-color: var(--bg-main);
    }

    /* ===== Header ===== */
    .dashboard-header {
        background: linear-gradient(135deg, #26A69A 0%, #4DB6AC 50%, #80CBC4 100%);
        padding: 28px 40px;
        border-radius: var(--radius-lg);
        margin-bottom: 28px;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
    }
    .dashboard-header::after {
        content: '';
        position: absolute;
        bottom: -60%;
        left: 20%;
        width: 200px;
        height: 200px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .dashboard-header h1 {
        color: white;
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        color: rgba(255,255,255,0.9);
        font-size: 14px;
        margin: 0;
    }

    /* ===== Cards ===== */
    .card {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        padding: 24px;
        margin-bottom: 20px;
        transition: box-shadow 0.2s ease;
    }
    .card:hover {
        box-shadow: var(--shadow-md);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 2px solid #E0F2F1;
    }
    .card-header-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #26A69A, #4DB6AC);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 18px;
    }
    .card-header-title {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* ===== KPI Value Boxes ===== */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-box {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        padding: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .kpi-box.status-good::before { background: var(--success); }
    .kpi-box.status-warning::before { background: var(--warning); }
    .kpi-box.status-danger::before { background: var(--danger); }
    .kpi-box.status-info::before { background: var(--primary); }

    .kpi-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 12px;
        color: var(--text-secondary);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ===== Feature Input Section ===== */
    .feature-group {
        background: #FAFBFC;
        border-radius: var(--radius-sm);
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #EDF2F7;
    }
    .feature-label {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .feature-icon {
        font-size: 18px;
    }
    .feature-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
    }
    .feature-help {
        font-size: 11px;
        color: var(--text-muted);
    }

    /* ===== Result Cards ===== */
    .result-card {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        padding: 28px;
        margin-bottom: 20px;
    }
    .result-positive {
        border-left: 5px solid var(--danger);
        background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
    }
    .result-negative {
        border-left: 5px solid var(--success);
        background: linear-gradient(135deg, #F0FFF4 0%, #C6F6D5 100%);
    }

    /* ===== Progress Bar ===== */
    .custom-progress {
        height: 12px;
        background: #EDF2F7;
        border-radius: 6px;
        overflow: hidden;
        margin: 12px 0;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
    .custom-progress-fill.danger {
        background: linear-gradient(90deg, #EF5350, #F44336);
    }
    .custom-progress-fill.success {
        background: linear-gradient(90deg, #66BB6A, #4CAF50);
    }

    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(135deg, #26A69A 0%, #00897B 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: var(--radius-sm);
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(38, 166, 154, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(38, 166, 154, 0.4);
    }

    /* ===== Section Headers ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
    }
    .section-line {
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, var(--primary-light), transparent);
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
    }

    /* ===== Footer ===== */
    .dashboard-footer {
        text-align: center;
        padding: 24px;
        color: var(--text-muted);
        font-size: 13px;
        margin-top: 40px;
    }
    .dashboard-footer a {
        color: var(--primary);
        text-decoration: none;
    }

    /* ===== Divider ===== */
    .divider {
        height: 1px;
        background: var(--border);
        margin: 24px 0;
    }

    /* ===== Model Info Badge ===== */
    .info-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #E0F2F1;
        color: var(--primary-dark);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }

    /* ===== Empty State ===== */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: var(--text-muted);
    }
    .empty-state-icon {
        font-size: 56px;
        margin-bottom: 16px;
        opacity: 0.6;
    }
    .empty-state-text {
        font-size: 16px;
    }

    /* ===== Parameter Table ===== */
    .param-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #F7FAFC;
    }
    .param-row:last-child {
        border-bottom: none;
    }
    .param-key {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 13px;
        color: var(--text-secondary);
    }
    .param-value {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)

# ============ Dashboard Layout ============

# Header
st.markdown("""
<div class="dashboard-header">
    <h1>🧠 Intracranial Aneurysm Growth Prediction</h1>
    <p>Clinical Decision Support System | LightGBM Model | SHAP Explainable Analysis</p>
</div>
""", unsafe_allow_html=True)

# ===== KPI Section =====
status_icon = "✅" if model else "❌"
status_text = "Ready" if model else "Error"
scaler_icon = "✅" if scaler else "❌"
scaler_text = "Loaded" if scaler else "Not Found"

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-box status-{'good' if model else 'danger'}">
        <div class="kpi-icon">🤖</div>
        <div class="kpi-value">{status_icon}</div>
        <div class="kpi-label">Model Status</div>
    </div>
    <div class="kpi-box status-info">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">4</div>
        <div class="kpi-label">Features</div>
    </div>
    <div class="kpi-box status-{'good' if scaler else 'warning'}">
        <div class="kpi-icon">⚖️</div>
        <div class="kpi-value">{scaler_icon}</div>
        <div class="kpi-label">Scaler</div>
    </div>
    <div class="kpi-box status-info">
        <div class="kpi-icon">🌲</div>
        <div class="kpi-value">250</div>
        <div class="kpi-label">Estimators</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== Main Content =====
col_input, col_result = st.columns([2, 3], gap="large")

# ===== Left Column - Input =====
with col_input:
    # Quick Guide Card
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-header-icon">📋</div>
            <div class="card-header-title">Quick Guide</div>
        </div>
        <div style="font-size:14px; color:var(--text-secondary); line-height:1.8;">
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#E0F2F1; color:#00897B; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">1</span>
                <span>Input the <strong>2 morphological and 2 hemodynamic features</strong> of the aneurysm from imaging data</span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#E0F2F1; color:#00897B; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">2</span>
                <span>Click <strong>"Start Prediction"</strong> to run the analysis</span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#E0F2F1; color:#00897B; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">3</span>
                <span>Review the <strong>risk assessment</strong> and probability scores</span>
            </div>
        </div>
        <div style="margin-top:16px; padding-top:16px; border-top:1px solid #E2E8F0;">
            <div style="font-size:12px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span>💡</span>
                <span>All values are automatically standardized before prediction</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Input Card
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-header-icon">📝</div>
            <div class="card-header-title">Feature Input</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    input_values = {}

    feature_ranges = {
        'HD2ND_Ratio': (0.00, 10.0),
        'Inflow_Angle': (0.0, 180.0),
        'Oscillatory_Shear_Index_Mean': (0.0, 1.0),
        'Pressure_Mean': (0.0, 2000.0)
    }

    feature_defaults = {
        'HD2ND_Ratio': 0.8,
        'Inflow_Angle': 60.0,
        'Oscillatory_Shear_Index_Mean': 0.3,
        'Pressure_Mean': 500.0

    }

    for feature in SELECTED_FEATURES:
        info = FEATURE_INFO[feature]
        st.markdown(f"""
        <div class="feature-group">
            <div class="feature-label">
                <span class="feature-icon">{info['icon']}</span>
                <div>
                    <div class="feature-name">{info['label']}</div>
                    <div class="feature-help">{info['help']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        min_val, max_val = feature_ranges[feature]
        default_v = feature_defaults[feature]

        val = st.number_input(
            f"{info['label']}",
            value=default_v,
            format="%.4f",
            key=feature,
            label_visibility="collapsed"
        )

        if not (min_val <= val <= max_val):
            st.error(f"⚠️ Value must be between {min_val} and {max_val}")

        input_values[feature] = val

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🔬 Start Prediction", use_container_width=True)

    if predict_clicked:
        st.session_state.prediction_made = True
        st.session_state.input_values = input_values

# ===== Right Column - Results =====
with col_result:
    if st.session_state.get('prediction_made', False):
        if model is None:
            st.error("Model not loaded. Please check model files.")
        else:
            # Prepare data
            full_input = {feat: 0.0 for feat in ALL_SCALER_FEATURES}
            for feat in SELECTED_FEATURES:
                full_input[feat] = input_values[feat]
            full_input_df = pd.DataFrame([full_input])

            if scaler is not None:
                full_scaled = scaler.transform(full_input_df)
                full_scaled_df = pd.DataFrame(full_scaled, columns=ALL_SCALER_FEATURES).round(2)
                input_scaled_df = full_scaled_df[SELECTED_FEATURES].copy()
            else:
                input_scaled_df = pd.DataFrame([input_values])[SELECTED_FEATURES].copy()

            X_input = input_scaled_df.values

            # Predict
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]

            is_growth = prediction == 1
            confidence = max(proba) * 100
            growth_prob = proba[1] * 100
            no_growth_prob = proba[0] * 100

            # Result Header
            st.markdown("""
            <div class="section-header">
                <span class="section-title">Prediction Results</span>
                <div class="section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            # Main Result Card
            if is_growth:
                st.markdown(f"""
                <div class="result-card result-positive">
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                        <span style="font-size:32px;">⚠️</span>
                        <div>
                            <div style="font-size:22px; font-weight:700; color:#C53030;">Aneurysm Growth Risk Detected</div>
                            <div style="font-size:14px; color:#9B2C2C;">Model indicates high probability of growth</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-negative">
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                        <span style="font-size:32px;">✅</span>
                        <div>
                            <div style="font-size:22px; font-weight:700; color:#276749;">No Growth Risk</div>
                            <div style="font-size:14px; color:#2F855A;">Model indicates low probability of growth</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # KPI Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="kpi-box status-{'good' if confidence > 70 else 'warning'}">
                    <div class="kpi-icon">🎯</div>
                    <div class="kpi-value">{confidence:.1f}%</div>
                    <div class="kpi-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="kpi-box status-success">
                    <div class="kpi-icon">🛡️</div>
                    <div class="kpi-value">{no_growth_prob:.1f}%</div>
                    <div class="kpi-label">Non-Growth</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="kpi-box status-{'danger' if growth_prob > 50 else 'info'}">
                    <div class="kpi-icon">⚡</div>
                    <div class="kpi-value">{growth_prob:.1f}%</div>
                    <div class="kpi-label">Growth</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability Bar
            st.markdown(f"""
            <div class="result-card" style="margin-top:20px;">
                <div style="font-size:14px; font-weight:600; color:var(--text-secondary); margin-bottom:12px;">
                    Growth Probability
                </div>
                <div class="custom-progress">
                    <div class="custom-progress-fill {'danger' if growth_prob > 50 else 'success'}"
                         style="width: {growth_prob}%;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-muted);">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ===== SHAP Analysis Section =====
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <span class="section-title">SHAP Explainability Analysis</span>
                <div class="section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            chart_type = st.selectbox(
                "Chart Type",
                ["SHAP Bar Chart", "Waterfall Plot", "Force Plot", "Custom Bar Chart"],
                key="shap_chart_type"
            )

            with st.spinner("Computing SHAP values..."):
                explainer = get_explainer(model)
                shap_values = explainer.shap_values(X_input)

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Old format: list of arrays [class0, class1, ...]
                    shap_values_display = shap_values[1]
                    if isinstance(explainer.expected_value, list):
                        expected_value = explainer.expected_value[1]
                    elif isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.size > 1:
                        expected_value = explainer.expected_value.ravel()[1]
                    else:
                        expected_value = explainer.expected_value
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    # New format: (n_samples, n_features, n_classes)
                    shap_values_display = shap_values[:, :, 1]  # positive class (growth)
                    expected_value = np.asarray(explainer.expected_value).ravel()[1]
                else:
                    # Regression or single-class
                    shap_values_display = shap_values
                    expected_value = explainer.expected_value

                # Ensure expected_value is a scalar
                expected_value = np.asarray(expected_value).ravel()[0]

                feature_display_names = [FEATURE_INFO[f]['label'] for f in SELECTED_FEATURES]

            st.markdown(f"""
            <div style="background:#E0F2F1; padding:10px 16px; border-radius:8px; margin-bottom:16px;
                        font-size:13px; color:{'#C53030' if expected_value > 0.5 else '#276749'};">
                <strong>Base Value (Expected):</strong> {expected_value:.4f}
            </div>
            """, unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(8, 5))

            if chart_type == "SHAP Bar Chart":
                shap.plots.bar(
                    shap.Explanation(
                        values=shap_values_display[0],
                        base_values=expected_value,
                        data=X_input[0],
                        feature_names=feature_display_names
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.clf()

            elif chart_type == "Waterfall Plot":
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values_display[0],
                        base_values=expected_value,
                        data=X_input[0],
                        feature_names=feature_display_names
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.clf()

            elif chart_type == "Force Plot":
                shap.plots.force(
                    base_value=expected_value,
                    shap_values=shap_values_display[0],
                    features=X_input[0],
                    feature_names=feature_display_names,
                    matplotlib=True,
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.clf()

            elif chart_type == "Custom Bar Chart":
                values = shap_values_display[0]
                colors = ['#EF5350' if v > 0 else '#66BB6A' for v in values]
                bars = ax.barh(feature_display_names, values, color=colors, edgecolor='white')
                ax.axvline(x=0, color='#2D3748', linewidth=0.8)
                ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=11, color='#718096')
                ax.set_title('Feature Impact on Prediction', fontsize=13, fontweight='bold', color='#2D3748')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors='#718096')

                for bar, val in zip(bars, values):
                    offset = 0.003
                    x_pos = bar.get_width() + offset if val >= 0 else bar.get_width() - offset
                    ha = 'left' if val >= 0 else 'right'
                    ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:+.4f}',
                            va='center', ha=ha, fontsize=10, color='#2D3748')

                st.pyplot(fig)
                plt.clf()

            # Scaled Data
            with st.expander("📋 View Scaled Input Data", expanded=False):
                st.dataframe(
                    input_scaled_df.style.format("{:.4f}").background_gradient(cmap="YlGnBu"),
                    use_container_width=True
                )

    else:
        # Empty State
        st.markdown("""
        <div class="card">
            <div class="empty-state">
                <div class="empty-state-icon">🩺</div>
                <div class="empty-state-text">
                    Enter patient features on the left panel<br>
                    then click <strong>"Start Prediction"</strong> to analyze
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="dashboard-footer">
    <div style="margin-bottom:8px;">
        <span class="info-badge">🧠 RF Model</span>
        <span class="info-badge">📊 4 Features</span>
        <span class="info-badge">🔬 Research Use Only</span>
    </div>
    <p>Intracranial Aneurysm Growth Prediction Dashboard v1.0</p>
    <p>Data reduced to 367 samples | Not for clinical diagnosis</p>
</div>
""", unsafe_allow_html=True)
