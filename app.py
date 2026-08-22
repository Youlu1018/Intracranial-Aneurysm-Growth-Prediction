"""
Intracranial Aneurysm Growth Prediction System
Modern Medical Dashboard Style - Clean, Hierarchical, Low Cognitive Load
Model: SVM (RBF Kernel) | Features loaded from the final SVM deployment manifest
"""
import io
import hashlib
import json
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import warnings
from cryptography.fernet import Fernet, InvalidToken

warnings.filterwarnings('ignore')

# ============ Page Config ============
st.set_page_config(
    page_title="Aneurysm Growth Research Prototype",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ File Config ============
APP_DIR = Path(__file__).resolve().parent
MODEL_MANIFEST_PATH = APP_DIR / "SVMappdata" / "svm_deployment_manifest.json"
ENCRYPTION_KEY_NAME = "MODEL_ENCRYPTION_KEY"
LOCAL_KEY_PATH = APP_DIR / ".svm_deployment_key_DO_NOT_UPLOAD.txt"


def _load_deployment_manifest():
    """Load the non-sensitive schema exported with the final SVM model."""
    if not MODEL_MANIFEST_PATH.exists():
        raise RuntimeError(f"Deployment manifest not found: {MODEL_MANIFEST_PATH}")
    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    required = {
        "bundle_file",
        "schema_version",
        "model",
        "selected_features",
        "feature_defaults_training_median",
        "feature_ranges_complete_cohort",
        "model_parameters",
        "bundle_sha256",
        "bundle_bytes",
    }
    missing = required.difference(manifest)
    if missing:
        raise RuntimeError(f"Deployment manifest is incomplete: {sorted(missing)}")
    if manifest["model"] != "SVM":
        raise RuntimeError("This application accepts the final SVM deployment only")
    features = manifest["selected_features"]
    if not features or len(features) != len(set(features)):
        raise RuntimeError("Deployment manifest has empty or duplicated SVM features")
    if set(features) != set(manifest["feature_defaults_training_median"]):
        raise RuntimeError("SVM feature defaults do not match selected_features")
    if set(features) != set(manifest["feature_ranges_complete_cohort"]):
        raise RuntimeError("SVM feature ranges do not match selected_features")
    return manifest


def _load_encryption_key():
    """Load a deployment secret without ever displaying or logging it."""
    key = os.environ.get(ENCRYPTION_KEY_NAME)
    if not key:
        try:
            key = st.secrets.get(ENCRYPTION_KEY_NAME)
        except Exception:
            key = None
    # Local-only convenience requested for this delivery. The file is ignored
    # by Git and is never expected to exist in Streamlit Community Cloud.
    if not key and LOCAL_KEY_PATH.exists():
        match = re.search(
            rf"^{ENCRYPTION_KEY_NAME}\s*=\s*['\"]?([A-Za-z0-9_-]{{43}}=)['\"]?\s*$",
            LOCAL_KEY_PATH.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        key = match.group(1) if match else None
    return key


DEPLOYMENT_MANIFEST = _load_deployment_manifest()
MODEL_BUNDLE_PATH = APP_DIR / "SVMappdata" / DEPLOYMENT_MANIFEST["bundle_file"]
REQUIRED_BUNDLE_KEYS = {
    "model",
    "scaler",
    "selected_features",
    "shap_background",
    "permutation_importance_mean",
    "permutation_importance_std",
}

SELECTED_FEATURES = list(DEPLOYMENT_MANIFEST["selected_features"])
BINARY_FEATURES = {"Alcohol_Use", "Tobbaco_Use"}.intersection(SELECTED_FEATURES)

FEATURE_INFO_CATALOG = {
    'Alcohol_Use': {
        'label': 'Alcohol Use',
        'help': 'Clinical variable: history of alcohol use (No = 0, Yes = 1).',
        'icon': '🍷'
    },
    'Tobbaco_Use': {
        'label': 'Tobacco Use',
        'help': 'Clinical variable: history of tobacco use (No = 0, Yes = 1).',
        'icon': '🚭'
    },
    'Diam': {
        'label': 'Aneurysm Diameter',
        'help': 'Morphological parameter: aneurysm diameter (mm).',
        'icon': '📐'
    },
    'MaxDiam': {
        'label': 'Maximum Diameter',
        'help': 'Morphological parameter: Maximum diameter of the intracranial aneurysm (mm).',
        'icon': '📏'
    },
    'Volume': {
        'label': 'Aneurysm Volume',
        'help': 'Morphological parameter: aneurysm volume (mm³).',
        'icon': '🧊'
    },
    'UI': {
        'label': 'Undulation Index',
        'help': 'Morphological parameter: Undulation index of the intracranial aneurysm.',
        'icon': '〰️'
    },
    'NSI': {
        'label': 'Nonsphericity Index',
        'help': 'Morphological parameter: Nonsphericity index of the intracranial aneurysm.',
        'icon': '🔷'
    }
}
FEATURE_INFO = {
    feature: FEATURE_INFO_CATALOG.get(
        feature,
        {
            "label": feature.replace("_", " "),
            "help": f"Input feature exported by the final SVM pipeline: {feature}.",
            "icon": "📊",
        },
    )
    for feature in SELECTED_FEATURES
}

SVM_PARAMS = dict(DEPLOYMENT_MANIFEST["model_parameters"])

# Observed bounds in the complete 784-case study cohort. These values are
# validation limits for model use; they are not used to refit the model.
FEATURE_RANGES = {
    feature: tuple(DEPLOYMENT_MANIFEST["feature_ranges_complete_cohort"][feature])
    for feature in SELECTED_FEATURES
}

# Training-set medians provide representative, non-identifying defaults.
FEATURE_DEFAULTS = {
    feature: float(DEPLOYMENT_MANIFEST["feature_defaults_training_median"][feature])
    for feature in SELECTED_FEATURES
}

# ============ Load Resources ============
@st.cache_resource
def _load_deployment_bundle():
    """Decrypt and validate the deployment-only model bundle in memory."""
    if not MODEL_BUNDLE_PATH.exists():
        return None, f"Encrypted deployment bundle not found: {MODEL_BUNDLE_PATH.name}"

    encrypted_bytes = MODEL_BUNDLE_PATH.read_bytes()
    if len(encrypted_bytes) != int(DEPLOYMENT_MANIFEST["bundle_bytes"]):
        return None, "Encrypted model bundle size does not match the deployment manifest"
    if hashlib.sha256(encrypted_bytes).hexdigest() != DEPLOYMENT_MANIFEST["bundle_sha256"]:
        return None, "Encrypted model bundle failed the SHA-256 integrity check"

    key = _load_encryption_key()
    if not key:
        return None, f"Missing deployment secret: {ENCRYPTION_KEY_NAME}"

    try:
        decrypted = Fernet(str(key).encode("utf-8")).decrypt(encrypted_bytes)
        bundle = joblib.load(io.BytesIO(decrypted))
        missing = REQUIRED_BUNDLE_KEYS.difference(bundle)
        if missing:
            return None, f"Encrypted bundle is incomplete: {sorted(missing)}"
        if bundle.get("selected_features") != SELECTED_FEATURES:
            return None, "Encrypted bundle feature order does not match the application"
        if bundle.get("schema_version") != DEPLOYMENT_MANIFEST["schema_version"]:
            return None, "Encrypted bundle schema version does not match the manifest"
        if bundle.get("model_parameters") != SVM_PARAMS:
            return None, "Encrypted bundle model parameters do not match the manifest"
        scaler_features = getattr(bundle["scaler"], "feature_names_in_", None)
        if scaler_features is not None and list(scaler_features) != SELECTED_FEATURES:
            return None, "Encrypted scaler feature order does not match the manifest"
        background = np.asarray(bundle["shap_background"], dtype=float)
        if background.ndim != 2 or background.shape[1] != len(SELECTED_FEATURES):
            return None, "Encrypted bundle has an invalid SHAP background shape"
        return bundle, None
    except (InvalidToken, ValueError):
        return None, "Invalid deployment secret or corrupted encrypted model bundle"
    except Exception as exc:
        return None, f"Failed to load encrypted deployment bundle: {exc}"


@st.cache_resource
def load_scaler():
    bundle, error = _load_deployment_bundle()
    if bundle is not None:
        return bundle["scaler"], "✅ Loaded training-set scaler from encrypted bundle"
    return None, f"❌ {error}"


def _build_scaler_from_training_data():
    # Kept to preserve the original app structure. Refitting is intentionally
    # disabled because deployment must reuse the training-only scaler.
    return None, "❌ Scaler refitting is disabled in the privacy-preserving deployment"

@st.cache_resource
def load_model():
    bundle, _ = _load_deployment_bundle()
    if bundle is not None:
        return bundle["model"]
    return None


@st.cache_resource
def get_background_data():
    bundle, _ = _load_deployment_bundle()
    if bundle is None:
        return None
    return np.asarray(bundle["shap_background"], dtype=float)


@st.cache_resource
def get_explainer():
    if model is None:
        return None, "SVM model not loaded"

    bg = background
    if bg is None:
        bg = get_background_data()
    if bg is None:
        return None, "No privacy-preserving SHAP background available"

    if isinstance(bg, np.ndarray):
        bg_array = bg
    elif hasattr(bg, 'data'):
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
st.session_state.setdefault("prediction_made", False)
st.session_state.setdefault("input_values", None)

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
    <h1>🧠 Intracranial Aneurysm Imaging-Detected Growth</h1>
    <p>Exploratory Research Prototype | Locked SVM (RBF Kernel) | SHAP Model Explanation</p>
</div>
""", unsafe_allow_html=True)

st.error(
    "RESEARCH USE ONLY — Not for clinical diagnosis, treatment selection, or patient-management decisions. "
    "This single-center, internally validated prototype has not undergone independent multicenter or "
    "prospective validation."
)
st.info(
    "Scope: conservatively managed saccular unruptured intracranial aneurysms matching the study eligibility "
    "criteria. The output estimates the study's binary outcome of imaging-detected growth during available "
    "follow-up (median 375 days); it is not a validated 12-month, 24-month, or lifetime risk estimate."
)

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
        <div class="kpi-value">{len(SELECTED_FEATURES)}</div>
        <div class="kpi-label">Features</div>
    </div>
    <div class="kpi-box status-{'good' if scaler else 'warning'}">
        <div class="kpi-icon">⚖️</div>
        <div class="kpi-value">{scaler_status_icon}</div>
        <div class="kpi-label">Scaler</div>
    </div>
    <div class="kpi-box status-info">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-value">{str(SVM_PARAMS['kernel']).upper()}</div>
        <div class="kpi-label">SVM Kernel</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== Main Content =====
col_input, col_result = st.columns([2, 3], gap="large")

# ===== Left Column - Input =====
with col_input:
    # Quick Guide Card
    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <div class="card-header-icon">📋</div>
            <div class="card-header-title">Quick Guide</div>
        </div>
        <div style="font-size:14px; color:var(--text-secondary); line-height:1.8;">
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#EDE7F6; color:#651FFF; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">1</span>
                <span>Input the <strong>{len(SELECTED_FEATURES)} features selected by the final SVM pipeline</strong></span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#EDE7F6; color:#651FFF; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">2</span>
                <span>Click <strong>"Generate research output"</strong> to run the locked model</span>
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; margin-bottom:12px;">
                <span style="background:#EDE7F6; color:#651FFF; width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;">3</span>
                <span>Review the <strong>risk assessment</strong> and SHAP explanation</span>
            </div>
        </div>
        <div style="margin-top:16px; padding-top:16px; border-top:1px solid #E2E8F0;">
            <div style="font-size:12px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span>💡</span>
                <span>Continuous values are standardized with the locked training-derived transformer; binary values remain 0/1.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Batch all feature edits so the displayed result changes only after an
    # explicit submission.
    input_form = st.form("prediction_form", border=False)

    # Feature Input Card
    input_form.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-header-icon">📝</div>
            <div class="card-header-title">Feature Input</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    feature_ranges = FEATURE_RANGES
    feature_defaults = FEATURE_DEFAULTS

    input_values = {}
    for feature in SELECTED_FEATURES:
        info = FEATURE_INFO[feature]
        input_form.markdown(f"""
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
        input_form.markdown(
            f'<span style="font-size:12px; color:#A0AEC0;">{help_text}</span>',
            unsafe_allow_html=True
        )

        min_val, max_val = feature_ranges[feature]
        default_v = feature_defaults[feature]
        input_form.caption(
            f"Observed complete-cohort input range: {min_val:.4f} to {max_val:.4f}. "
            "The prototype does not extrapolate beyond this range."
        )
        if feature in BINARY_FEATURES:
            binary_labels = ["No (0)", "Yes (1)"]
            selected_binary = input_form.selectbox(
                f"{info['label']}",
                binary_labels,
                index=int(round(default_v)),
                key=feature,
                label_visibility="collapsed",
            )
            input_values[feature] = float(binary_labels.index(selected_binary))
        else:
            input_values[feature] = input_form.number_input(
                f"{info['label']}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=default_v,
                format="%.4f",
                key=feature,
                label_visibility="collapsed",
            )

        # Real-time validation
        if (
                input_values[feature] < min_val
                or input_values[feature] > max_val
        ):
            input_form.warning(
                f"⚠️ {info['label']} is outside the recommended range "
                f"({min_val:.4f}–{max_val:.4f}). "
                f"Please verify the entered value."
            )

    input_form.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    predict_clicked = input_form.form_submit_button(
        "Generate research output",
        icon=":material/science:",
        type="primary",
        width="stretch",
    )
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

                One or more parameters are outside the observed range of the complete study cohort.
                The research prototype does not generate extrapolated outputs.

                Please review the following entries:
                """
            )

            for err in validation_errors:
                st.markdown(f"• {err}")

            st.stop()
        # Continue Prediction
        st.session_state.prediction_made = True
        st.session_state.input_values = dict(input_values)

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
            st.error(f"❌ SVM model unavailable. {scaler_msg}")
        else:
            # ---- Prepare input data ----
            submitted_inputs = st.session_state.input_values
            input_array = np.array([[submitted_inputs[f] for f in SELECTED_FEATURES]])
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
            proba = model.predict_proba(X_input)[0]
            growth_prob = proba[1] * 100
            no_growth_prob = proba[0] * 100

            # ---- Result Header ----
            st.markdown("""
            <div class="section-header">
                <span class="section-title">Prediction Results</span>
                <div class="section-line"></div>
            </div>
            """, unsafe_allow_html=True)

            st.warning(
                "This output is a model-estimated probability for research demonstration only. "
                "No clinical risk categories or action thresholds have been validated."
            )

            # ---- Model Output Metrics ----
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="kpi-box status-info">
                    <div class="kpi-icon">🎯</div>
                    <div class="kpi-value">{growth_prob:.1f}%</div>
                    <div class="kpi-label">Imaging-detected growth output</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="kpi-box status-good">
                    <div class="kpi-icon">🛡️</div>
                    <div class="kpi-value">{no_growth_prob:.1f}%</div>
                    <div class="kpi-label">Without detected growth output</div>
                </div>
                """, unsafe_allow_html=True)

            # ---- Probability Bar ----
            st.markdown(f"""
            <div class="result-card" style="margin-top:20px;">
                <div style="font-size:14px; font-weight:600; color:var(--text-secondary); margin-bottom:12px;">
                    Model-estimated probability of imaging-detected growth
                </div>
                <div class="custom-progress">
                    <div class="custom-progress-fill {'danger' if growth_prob >= 50 else 'success'}"
                         style="width: {growth_prob}%;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-muted);">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                "Outcome definition in the study: ≥1 mm increase in maximum diameter, >0.5 mm increase in two "
                "perpendicular diameters, or >10% relative increase in volume under the same imaging modality. "
                "The estimate must not be interpreted as causal or used to determine surveillance or treatment."
            )

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
                        bundle, _ = _load_deployment_bundle()
                        importances_mean = np.asarray(bundle["permutation_importance_mean"])
                        importances_std = np.asarray(bundle["permutation_importance_std"])

                        if chart_type == "Permutation Importance":
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sorted_idx = importances_mean.argsort()
                            ax.barh(
                                [feature_display_names[i] for i in sorted_idx],
                                importances_mean[sorted_idx],
                                xerr=importances_std[sorted_idx],
                                color='#7C4DFF', edgecolor='white'
                            )
                            ax.set_xlabel('Mean Accuracy Decrease', fontsize=11, color='#718096')
                            ax.set_title('Permutation Feature Importance', fontsize=13, fontweight='bold',
                                         color='#2D3748')
                            for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
                            ax.tick_params(colors='#718096')
                            st.pyplot(fig);
                            plt.clf()
                        elif chart_type == "Custom Bar Chart":
                            importances = importances_mean
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
                        st.info("The encrypted deployment bundle does not contain fallback importance values.")

            # ---- Scaled Data ----
            with st.expander("📋 View Scaled Input Data", expanded=False):
                st.dataframe(
                    input_scaled_df.style.format("{:.4f}").background_gradient(cmap="YlGnBu"),
                    width="stretch"
                )
            # ---- Raw Input ----
            with st.expander("📝 View Raw Input Values", expanded=False):
                st.dataframe(
                    input_df.style.format("{:.4f}"),
                    width="stretch"
                )
    else:
        # Empty State
        st.markdown("""
        <div class="card">
            <div class="empty-state">
                <div class="empty-state-icon">🩺</div>
                <div class="empty-state-text">
                    Enter patient features on the left panel<br>
                    then click <strong>"Generate research output"</strong>
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
st.markdown(f"""
<div class="dashboard-footer">
    <div style="margin-bottom:8px;">
        <span class="info-badge">🧠 SVM · RBF Kernel</span>
        <span class="info-badge">📊 {len(SELECTED_FEATURES)} Features</span>
        <span class="info-badge">🔬 RESEARCH USE ONLY</span>
    </div>
    <p>Intracranial Aneurysm Growth Prediction Dashboard v2.0</p>
    <p>Not for clinical diagnosis, treatment selection, surveillance planning, or patient-management decisions</p>
    <p>Single-center internal validation only · Independent multicenter and prospective validation required</p>
</div>
""", unsafe_allow_html=True)
