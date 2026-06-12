"""
Intracranial Aneurysm Growth Prediction System
Modern Medical Dashboard Style - Clean, Hierarchical, Low Cognitive Load
Model: SVM (RBF Kernel) | 7 Morphological & Hemodynamic Features
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
    page_title="Aneurysm Prediction Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ File Config ============
MODEL_PATH = Path("SVMappdata/svm_model.joblib")
SCALER_PATH = Path("SVMappdata/scaler.joblib")
TRAIN_DATA_PATH = Path("SVMappdata/traindataSVM.joblib")

SELECTED_FEATURES = [
    'Neck_Diam',
    'Inflow_Angle',
    'DP',
    'UI',
    'NSI',
    'OSI_Mean',
    'EL_Ratio'
]

FEATURE_INFO = {
    'Neck_Diam': {
        'label': 'Neck Diameter',
        'help': 'Morphological parameter: The maximum diameter of the neck plane of intracranial aneurysm.',
        'icon': '📏'
    },
    'Inflow_Angle': {
        'label': 'Inflow Angle',
        'help': 'Morphological parameter: The angle between the direction of blood flow and the direction of the diameter of intracranial aneurysm. (°)',
        'icon': '🧿'
    },
    'DP': {
        'label': 'Mean Diameter of Parent Artery',
        'help': 'Morphological parameter: Mean diameter of the parent artery measured 10 mm proximal and distal to the neck of intracranial arterial aneurysms.',
        'icon': '🧬'
    },
    'UI': {
        'label': 'Undulation Index',
        'help': 'Morphological parameter: Undulation index of intracranial aneurysm. (**Rajabzadeh-Oghaz, H.** et al. ***World Neurosurg*** 2018, 119 : e541–e550)',
        'icon': '🌊'
    },
    'NSI': {
        'label': 'Nonsphericity Index',
        'help': 'Morphological parameter: Nonsphericity index of intracranial aneurysm. (**Rajabzadeh-Oghaz, H.** et al. ***World Neurosurg*** 2018, 119 : e541–e550)',
        'icon': '🔘'
    },
    'OSI_Mean': {
        'label': 'Mean Oscillatory Shear Index',
        'help': 'Hemodynamic parameter: Mean oscillatory shear index of intracranial aneurysm.',
        'icon': '🌀'
    },
    'EL_Ratio': {
        'label': 'Energy Loss Ratio',
        'help': 'Hemodynamic parameter: Energy loss ratio of intracranial aneurysm.',
        'icon': '⚡'
    }
}

SVM_PARAMS = {
    'C': 0.7,
    'kernel': 'rbf',
    'gamma': 0.12,
    'degree': 2,
    'tol': 0.001,
    'class_weight': 'balanced',
    'probability': True,
    'random_state': 123,
}

# ============ Load Resources ============
@st.cache_resource
def load_scaler():
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        return scaler, "✅ Loaded existing scaler from SVMappdata"
    return _build_scaler_from_training_data()


def _build_scaler_from_training_data():
    from sklearn.preprocessing import StandardScaler
    if not TRAIN_DATA_PATH.exists():
        return None, "❌ No scaler available — traindataSVM.csv not found"

    try:
        df = joblib.load(TRAIN_DATA_PATH)
        scaler = StandardScaler()
        scaler.fit(df[SELECTED_FEATURES])
        scaler.feature_names_in_ = np.array(SELECTED_FEATURES)
        return scaler, "✅ Built new scaler from traindataSVM.csv (7 features)"
    except Exception as e:
        return None, f"❌ Failed to build scaler: {e}"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        if hasattr(model, 'feature_names_in_'):
            del model.feature_names_in_
        return model
    return None


@st.cache_resource
def get_background_data():
    if not TRAIN_DATA_PATH.exists():
        return None

    try:
        df = joblib.load(TRAIN_DATA_PATH)
        full_data = df[SELECTED_FEATURES].fillna(0).values

        if len(full_data) > 30:
            return shap.kmeans(full_data, 30)
        return full_data
    except Exception as e:
        st.error(f"❌ Failed to load background data: {e}")
        return None


@st.cache_resource
def get_explainer():
    if model is None:
        return None, "SVM model not loaded"

    bg = background
    if bg is None:
        bg = get_background_data()
    if bg is None:
        return None, "No background data available — traindataSVM.csv missing"

    if hasattr(bg, 'data'):
        bg_array = bg.data
    else:
        bg_array = bg

    try:
        explainer = shap.KernelExplainer(model.predict_proba, bg_array)
        return explainer, None
    except Exception as e:
        return None, f"KernelExplainer creation failed: {e}"


# ============ Initialize ============
scaler, scaler_msg = load_scaler()
model = load_model()
background = get_background_data()

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
        background: linear-gradient(135deg, #7C4DFF 0%, #651FFF 50%, #448AFF 100%);
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
        border-bottom: 2px solid #EDE7F6;
    }
    .card-header-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #7C4DFF, #651FFF);
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
    .kpi-box.status-info::before { background: #7C4DFF; }
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
    }
    .feature-icon {
        font-size: 18px;
    }
    .feature-name {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-primary);
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
        background: linear-gradient(135deg, #7C4DFF 0%, #651FFF 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: var(--radius-sm);
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(124, 77, 255, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 77, 255, 0.4);
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
        background: linear-gradient(90deg, #B388FF, transparent);
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
        color: #7C4DFF;
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
        background: #EDE7F6;
        color: #651FFF;
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
    <p>Clinical Decision Support System | SVM (RBF Kernel) | SHAP Explainable Analysis</p>
</div>
""", unsafe_allow_html=True)

# ===== KPI Section =====
status_icon = "✅" if model else "❌"
scaler_status_icon = "✅" if scaler else "⚠️"
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-box status-{'good' if model else 'danger'}">
        <div class="kpi-icon">🤖</div>
        <div class="kpi-value">{status_icon}</div>
        <div class="kpi-label">SVM Model Status</div>
    </div>
    <div class="kpi-box status-info">
        <div class="kpi-icon">📊</div>
        <div class="kpi-value">7</div>
        <div class="kpi-label">Features</div>
    </div>
    <div class="kpi-box status-{'good' if scaler else 'warning'}">
        <div class="kpi-icon">⚖️</div>
        <div class="kpi-value">{scaler_status_icon}</div>
        <div class="kpi-label">Scaler</div>
    </div>
    <div class="kpi-box status-info">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-value">RBF</div>
        <div class="kpi-label">SVM Kernel</div>
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
                <span style="background:#EDE7F6; color:#651FFF; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">1</span>
                <span>Input the <strong>7 morphological & hemodynamic features</strong> of the aneurysm</span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#EDE7F6; color:#651FFF; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">2</span>
                <span>Click <strong>"Start Prediction"</strong> to run the analysis</span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#EDE7F6; color:#651FFF; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">3</span>
                <span>Review the <strong>risk assessment</strong> and SHAP explanation</span>
            </div>
        </div>
        <div style="margin-top:16px; padding-top:16px; border-top:1px solid #E2E8F0;">
            <div style="font-size:12px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span>💡</span>
                <span>Values are standardized before prediction. SVM with probability calibration.</span>
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

    feature_ranges = {
        'Neck_Diam': (0.10, 10.00),
        'Inflow_Angle': (5.00, 170.00),
        'DP': (1.00, 6.00),
        'UI': (0.04, 0.40),
        'NSI': (0.10, 0.50),
        'OSI_Mean': (0.001, 0.08),
        'EL_Ratio': (0.02, 4.00)
    }

    feature_defaults = {
        'Neck_Diam': 2.17,
        'Inflow_Angle': 91.62,
        'DP': 2.21,
        'UI': 0.11,
        'NSI': 0.11,
        'OSI_Mean': 0.01,
        'EL_Ratio': 0.46
    }

    input_values = {}
    for feature in SELECTED_FEATURES:
        info = FEATURE_INFO[feature]
        st.markdown(f"""
        <div class="feature-group">
            <div class="feature-label">
                <span class="feature-icon">{info['icon']}</span>
                <div>
                    <div class="feature-name">{info['label']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        help_text = info['help']
        st.markdown(
            f'<span style="font-size:12px; color:#A0AEC0;">{help_text}</span>',
            unsafe_allow_html=True
        )

        min_val, max_val = feature_ranges[feature]
        default_v = feature_defaults[feature]
        input_values[feature] = st.number_input(
            f"{info['label']}",
            value = default_v,
            format = "%.4f",
            key = feature,
            label_visibility = "collapsed"
        )

        # Real-time validation
        if (
                input_values[feature] < min_val
                or input_values[feature] > max_val
        ):
            st.warning(
                f"⚠️ {info['label']} is outside the recommended range "
                f"({min_val:.4f}–{max_val:.4f}). "
                f"Please verify the entered value."
            )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🔬 Start Prediction", use_container_width=True)
    if predict_clicked:
        # Input Validation
        validation_errors = []

        for feature in SELECTED_FEATURES:

            value = input_values[feature]
            min_val, max_val = feature_ranges[feature]

            if value < min_val or value > max_val:
                validation_errors.append(
                    f"{FEATURE_INFO[feature]['label']}: "
                    f"expected range {min_val:.4f}–{max_val:.4f}, "
                    f"received {value:.4f}"
                )

        # Out off Range
        if validation_errors:

            st.error(
                """
                Input validation failed.

                One or more parameters are outside the validated operating range of the prediction model.
                Predictions generated using out-of-range values may be unreliable.

                Please review the following entries:
                """
            )

            for err in validation_errors:
                st.markdown(f"• {err}")

            st.stop()
        # Continue Prediction
        st.session_state.prediction_made = True
        st.session_state.input_values = input_values

    # ---- SVM Model Parameters Card ----
    with st.expander("⚙️ SVM Model Parameters", expanded=False):
        st.markdown("""
        <div style="font-size:13px; color:var(--text-secondary); margin-bottom:12px;">
            Best SVM hyperparameters from grid search optimization:
        </div>
        """, unsafe_allow_html=True)
        for key, val in SVM_PARAMS.items():
            st.markdown(f"""
            <div class="param-row">
                <span class="param-key">{key}</span>
                <span class="param-value">{val}</span>
            </div>
            """, unsafe_allow_html=True)

# ===== Right Column - Results =====
with col_result:
    if st.session_state.get('prediction_made', False):
        if model is None:
            st.error("❌ SVM model not found. Please ensure `svm_model.joblib` is in the SVMappdata folder.")
        else:
            # ---- Prepare input data ----
            input_array = np.array([[input_values[f] for f in SELECTED_FEATURES]])
            input_df = pd.DataFrame(input_array, columns=SELECTED_FEATURES)

            # ---- Apply scaler ----
            if scaler is not None:
                input_scaled = scaler.transform(input_df)
                input_scaled_df = pd.DataFrame(input_scaled, columns=SELECTED_FEATURES).round(4)
            else:
                st.info("ℹ️ No scaler available, using raw input values.")
                input_scaled_df = input_df.copy()
                input_scaled = input_array

            X_input = input_scaled_df.values

            # st.write("Raw Input:")
            # st.write(input_df)
            #
            # st.write("Scaled Input:")
            # st.write(input_scaled_df)
            #
            # st.write("Scaler Mean:")
            # st.write(scaler.mean_)
            #
            # st.write("Scaler Scale:")
            # st.write(scaler.scale_)

            # ---- Predict ----
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]
            #is_growth = prediction == 1
            #confidence = max(proba) * 100
            growth_prob = proba[1] * 100
            no_growth_prob = proba[0] * 100

            # st.write("growth_prob:")
            # st.write(growth_prob)

            #confidence = abs(growth_prob - 50) * 2
            is_growth = growth_prob >= 50

            # st.write("classes =", model.classes_)
            # st.write("proba =", proba)
            # st.write("prediction =", prediction)
            # st.write("X_input =", X_input)
            # st.write("decision =", model.decision_function(X_input))
            #
            # scaler, scaler_msg = load_scaler()
            # st.write("scaler =", scaler)
            # st.write("scaler_msg =", scaler_msg)

            def get_risk_category(prob):
                if prob < 0.2:
                    return "Low"
                elif prob < 0.5:
                    return "Moderate"
                elif prob < 0.8:
                    return "High"
                else:
                    return "Very High"

            raw_growth_prob = proba[1]
            risk_category = get_risk_category(raw_growth_prob)

            # ---- Result Header ----
            st.markdown("""
            <div class="section-header">
                <span class="section-title">Prediction Results</span>
                <div class="section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            # ---- Main Result Card ----
            if is_growth:
                st.markdown(f"""
                <div class="result-card result-positive">
                    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                        <span style="font-size:32px;">⚠️</span>
                        <div>
                            <div style="font-size:22px; font-weight:700; color:#C53030;">Aneurysm Growth Risk Detected</div>
                            <div style="font-size:14px; color:#9B2C2C;">SVM model indicates high probability of aneurysm growth</div>
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
                            <div style="font-size:22px; font-weight:700; color:#276749;">Low Growth Risk</div>
                            <div style="font-size:14px; color:#2F855A;">SVM model indicates low probability of aneurysm growth</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ---- KPI Metrics ----
            m1, m2, m3 = st.columns(3)
            with m1:
                if risk_category == "Low":
                    status_cls = "good"
                elif risk_category == "Moderate":
                    status_cls = "info"
                elif risk_category == "High":
                    status_cls = "warning"
                else:  # Very High
                    status_cls = "danger"

                st.markdown(f"""
                <div class="kpi-box status-{status_cls}">
                    <div class="kpi-icon">🏷️</div>
                    <div class="kpi-value">{risk_category}</div>
                    <div class="kpi-label">Risk Category</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="kpi-box status-success">
                    <div class="kpi-icon">🛡️</div>
                    <div class="kpi-value">{no_growth_prob:.1f}%</div>
                    <div class="kpi-label">Stable</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="kpi-box status-{'danger' if growth_prob > 50 else 'info'}">
                    <div class="kpi-icon">🎯</div>
                    <div class="kpi-value">{growth_prob:.1f}%</div>
                    <div class="kpi-label">Growth Risk</div>
                </div>
                """, unsafe_allow_html=True)

            # ---- Probability Bar ----
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

            # ===== SHAP Explainability Analysis =====
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-header">
                <span class="section-title">Explainability Analysis</span>
                <div class="section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            explainer, explainer_error = get_explainer()
            use_shap = explainer is not None
            if not use_shap:
                st.warning(f"⚠️ SHAP not available: {explainer_error}. Using permutation importance instead.")

            chart_type = st.selectbox(
                "Chart Type",
                ["SHAP Bar Chart", "Waterfall Plot", "Force Plot", "Custom Bar Chart"] if use_shap
                else ["Permutation Importance", "Custom Bar Chart"],
                key="shap_chart_type"
            )

            feature_display_names = [FEATURE_INFO[f]['label'] for f in SELECTED_FEATURES]

            # ========== SHAP-based explanation ==========
            if use_shap:
                with st.spinner("🔄 Computing SHAP values (KernelExplainer)..."):
                    try:
                        shap_values_raw = explainer.shap_values(X_input, nsamples=200)

                        if isinstance(shap_values_raw, list):
                            shap_arr = shap_values_raw[1]
                            shap_vals_1d = shap_arr[0] if shap_arr.ndim == 2 else shap_arr
                        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
                            shap_vals_1d = shap_values_raw[0, :, 1]
                        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2:
                            shap_vals_1d = shap_values_raw[0]
                        else:
                            shap_vals_1d = np.asarray(shap_values_raw).ravel()[:len(SELECTED_FEATURES)]

                        if isinstance(explainer.expected_value, list):
                            base_val = explainer.expected_value[1]
                        elif isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.size > 1:
                            base_val = explainer.expected_value.ravel()[1]
                        else:
                            base_val = explainer.expected_value
                        base_val = float(np.asarray(base_val).ravel()[0])

                        st.markdown(f"""
                        <div style="background:#EDE7F6; padding:10px 16px; border-radius:8px; margin-bottom:16px;
                                    font-size:13px; color:{'#C53030' if base_val > 0.5 else '#276749'};">
                            <strong>Base Value (Expected):</strong> {base_val:.4f}
                        </div>
                        """, unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        if chart_type == "SHAP Bar Chart":
                            shap.plots.bar(shap.Explanation(
                                values=shap_vals_1d, base_values=base_val,
                                data=X_input[0], feature_names=feature_display_names
                            ), show=False)
                            st.pyplot(fig);
                            plt.clf()
                        elif chart_type == "Waterfall Plot":
                            shap.plots.waterfall(shap.Explanation(
                                values=shap_vals_1d, base_values=base_val,
                                data=X_input[0], feature_names=feature_display_names
                            ), show=False)
                            st.pyplot(fig);
                            plt.clf()
                        elif chart_type == "Force Plot":
                            shap.plots.force(
                                base_value=base_val, shap_values=shap_vals_1d,
                                features=X_input[0], feature_names=feature_display_names,
                                matplotlib=True, show=False
                            )
                            st.pyplot(plt.gcf());
                            plt.clf()
                        elif chart_type == "Custom Bar Chart":
                            colors = ['#EF5350' if v > 0 else '#66BB6A' for v in shap_vals_1d]
                            bars = ax.barh(feature_display_names, shap_vals_1d, color=colors, edgecolor='white')
                            ax.axvline(x=0, color='#2D3748', linewidth=0.8)
                            ax.set_xlabel('SHAP Value', fontsize=11, color='#718096')
                            ax.set_title('Feature Impact on Prediction', fontsize=13, fontweight='bold',
                                         color='#2D3748')
                            for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
                            ax.tick_params(colors='#718096')
                            for bar, val in zip(bars, shap_vals_1d):
                                offset = 0.003
                                x_pos = bar.get_width() + offset if val >= 0 else bar.get_width() - offset
                                ha = 'left' if val >= 0 else 'right'
                                ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f'{val:+.4f}',
                                        va='center', ha=ha, fontsize=10, color='#2D3748')
                            st.pyplot(fig);
                            plt.clf()
                    except Exception as e:
                        st.error(f"SHAP analysis error: {e}")

            # ========== Permutation Importance fallback ==========
            else:
                with st.spinner("🔄 Computing Permutation Importance..."):
                    try:
                        from sklearn.inspection import permutation_importance

                        if TRAIN_DATA_PATH.exists():
                            train_df = joblib.load(TRAIN_DATA_PATH)
                            X_train = train_df[SELECTED_FEATURES].fillna(0).values
                            y_train = train_df['Outcome'].values

                            # 标准化训练数据
                            if scaler is not None:
                                X_train = scaler.transform(X_train)

                            r = permutation_importance(
                                model, X_train, y_train,
                                n_repeats=10, random_state=123, scoring='accuracy'
                            )
                        else:
                            r = None
                            st.warning("Training data unavailable for permutation importance.")

                        if r is not None and chart_type == "Permutation Importance":
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sorted_idx = r.importances_mean.argsort()
                            ax.barh(
                                [feature_display_names[i] for i in sorted_idx],
                                r.importances_mean[sorted_idx],
                                xerr=r.importances_std[sorted_idx],
                                color='#7C4DFF', edgecolor='white'
                            )
                            ax.set_xlabel('Mean Accuracy Decrease', fontsize=11, color='#718096')
                            ax.set_title('Permutation Feature Importance', fontsize=13, fontweight='bold',
                                         color='#2D3748')
                            for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
                            ax.tick_params(colors='#718096')
                            st.pyplot(fig);
                            plt.clf()
                        elif r is not None and chart_type == "Custom Bar Chart":
                            importances = r.importances_mean
                            fig, ax = plt.subplots(figsize=(8, 5))
                            colors = ['#EF5350' if v > 0 else '#7C4DFF' for v in importances]
                            bars = ax.barh(feature_display_names, importances, color=colors, edgecolor='white')
                            ax.set_xlabel('Mean Accuracy Decrease', fontsize=11, color='#718096')
                            ax.set_title('Permutation Feature Importance', fontsize=13, fontweight='bold',
                                         color='#2D3748')
                            for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
                            ax.tick_params(colors='#718096')
                            for bar, val in zip(bars, importances):
                                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                                        f'{val:.4f}', va='center', fontsize=10, color='#2D3748')
                            st.pyplot(fig);
                            plt.clf()
                    except Exception as e:
                        st.error(f"Permutation importance error: {e}")
                        st.info("Ensure `traindataSVM.csv` is available with the 'Outcome' column.")

            # ---- Scaled Data ----
            with st.expander("📋 View Scaled Input Data", expanded=False):
                st.dataframe(
                    input_scaled_df.style.format("{:.4f}").background_gradient(cmap="YlGnBu"),
                    use_container_width=True
                )
            # ---- Raw Input ----
            with st.expander("📝 View Raw Input Values", expanded=False):
                st.dataframe(
                    input_df.style.format("{:.4f}"),
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

# ===== Model Info Sidebar =====
with st.sidebar:
    st.markdown("### 🧠 Model Information")
    st.markdown(f"""
    - **Model Type:** Support Vector Machine (SVM)
    - **Kernel:** RBF (γ={SVM_PARAMS['gamma']})
    - **C:** {SVM_PARAMS['C']}
    - **Class Weight:** {SVM_PARAMS['class_weight']}
    - **Features:** {len(SELECTED_FEATURES)}
    - **Probability:** Enabled
    ---
    ### 📊 Feature List
    """)
    for feat in SELECTED_FEATURES:
        info = FEATURE_INFO[feat]
        st.markdown(f"- {info['icon']} **{info['label']}**")
    st.markdown("---")
    scaler_status_color = "green" if scaler else "orange"
    st.markdown(f"**Scaler:** :{scaler_status_color}[{scaler_msg}]")
    st.markdown(f"**Model:** :{'green' if model else 'red'}[{'Loaded' if model else 'Not Found'}]")
    st.markdown("---")
    st.markdown("### ⚙️ SVM Parameters")
    for key, val in SVM_PARAMS.items():
        st.markdown(f"- `{key}` = `{val}`")

# Footer
st.markdown("""
<div class="dashboard-footer">
    <div style="margin-bottom:8px;">
        <span class="info-badge">🧠 SVM · RBF Kernel</span>
        <span class="info-badge">📊 7 Features</span>
        <span class="info-badge">🔬 Research Use Only</span>
    </div>
    <p>Intracranial Aneurysm Growth Prediction Dashboard v2.0</p>
    <p>SVM Model · SHAP KernelExplainer · Not for clinical diagnosis</p>
</div>
""", unsafe_allow_html=True)
