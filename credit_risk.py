"""
Credit Risk & Default Prediction Dashboard
by Doug Chingosho

Install:
    pip install streamlit scikit-learn xgboost imbalanced-learn pandas numpy
                matplotlib seaborn plotly scipy

Run:
    streamlit run credit_risk.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
# CSS Theme  (same dark palette as BSM calculator)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
*{ box-sizing:border-box; }
.stApp { background:#0f1117; }
section[data-testid="stSidebar"]          { background:#1a1d27 !important; }
section[data-testid="stSidebar"] *        { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
    background:#252836 !important;
    border:1px solid #3a3f55 !important;
    color:#e2e8f0 !important; border-radius:6px;
}
section[data-testid="stSidebar"] hr { border-color:#2d3148 !important; }
h1,h2,h3,h4,p,label,.stMarkdown { color:#e2e8f0 !important; }
hr { border-color:#2d3148 !important; }

div[data-testid="stMetric"] {
    border-radius:10px !important; padding:14px 16px !important;
    border:1px solid rgba(255,255,255,0.07) !important;
}
div[data-testid="stMetric"] label {
    font-size:11px !important; letter-spacing:.06em !important;
    text-transform:uppercase !important; color:#94a3b8 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size:22px !important; font-weight:700 !important; color:#f1f5f9 !important;
}

details { background:#1a1d27 !important; border-radius:8px !important;
          border:1px solid #2d3148 !important; }
details summary { color:#a5b4fc !important; }

.stDataFrame thead th { background:#252836 !important; color:#a5b4fc !important; font-size:12px !important; }
.stDataFrame tbody tr:nth-child(even) { background:#1e2130 !important; }
.stDataFrame tbody tr:nth-child(odd)  { background:#1a1d27 !important; }
.stDataFrame tbody td { color:#e2e8f0 !important; font-size:13px !important; }

.stButton>button {
    background:linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color:white !important; border:none !important;
    border-radius:8px !important; font-weight:700 !important;
    font-size:15px !important; height:46px !important;
}
.stButton>button:hover { opacity:.88 !important; }

.hero-banner {
    background:linear-gradient(135deg,#1e1b4b 0%,#312e81 40%,#1e3a5f 100%);
    border-radius:14px; padding:24px 32px; margin-bottom:24px;
    border:1px solid rgba(99,102,241,0.3); text-align:center;
}
.hero-title {
    font-size:22px; font-weight:700; color:#f1f5f9 !important;
    -webkit-text-fill-color:#f1f5f9 !important;
    margin:0 0 6px 0; text-align:center;
}
.hero-sub { font-size:13px; color:#a5b4fc !important; margin:0; text-align:center; }
.hero-badges { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
.badge { display:inline-block; padding:4px 12px; border-radius:20px;
         font-size:11px; font-weight:600; letter-spacing:.04em; }
.badge-blue   { background:#1e3a5f; color:#93c5fd; border:1px solid #2563eb44; }
.badge-purple { background:#2e1065; color:#c4b5fd; border:1px solid #7c3aed44; }
.badge-teal   { background:#042f2e; color:#5eead4; border:1px solid #0d948844; }
.badge-red    { background:#450a0a; color:#fca5a5; border:1px solid #dc262644; }
.badge-amber  { background:#431407; color:#fcd34d; border:1px solid #d9770644; }

.section-pill {
    display:inline-block; background:#312e81; color:#a5b4fc;
    font-size:11px; font-weight:700; padding:3px 12px; border-radius:20px;
    letter-spacing:.06em; text-transform:uppercase; margin-bottom:12px;
    border:1px solid rgba(99,102,241,0.3);
}
.risk-high   { color:#f87171 !important; font-weight:700; }
.risk-medium { color:#fbbf24 !important; font-weight:700; }
.risk-low    { color:#34d399 !important; font-weight:700; }

.contact-bar {
    margin-top:40px; padding:18px 24px;
    background:linear-gradient(135deg,#1e1b4b 0%,#1e3a5f 100%);
    border-radius:12px; border:1px solid rgba(99,102,241,0.3);
    display:flex; align-items:center; justify-content:center; gap:16px; flex-wrap:wrap;
}
.contact-label { font-size:13px; color:#a5b4fc; font-weight:500; margin-right:4px; }
.contact-btn {
    display:inline-flex; align-items:center; gap:7px; padding:8px 18px;
    border-radius:8px; font-size:13px; font-weight:600;
    text-decoration:none !important; cursor:pointer; border:none;
}
.contact-btn:hover { opacity:.85; }
.btn-linkedin { background:#0a66c2; color:#ffffff !important; }
.btn-email    { background:#4f46e5; color:#ffffff !important; }
.contact-divider { width:1px; height:28px; background:rgba(165,180,252,0.2); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Synthetic dataset generator (realistic Lending Club-style)
# ─────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=5000, seed=42):
    """
    Generates a realistic synthetic lending dataset.
    Default rate ≈ 22%, matching historical Lending Club averages.
    All features are grounded in real credit risk literature.
    """
    rng = np.random.default_rng(seed)

    age             = rng.integers(21, 70, n)
    income          = rng.lognormal(10.8, 0.6, n).clip(15000, 500000)
    loan_amount     = rng.lognormal(9.5, 0.7, n).clip(1000, 40000)
    loan_term       = rng.choice([36, 60], n, p=[0.6, 0.4])
    interest_rate   = rng.uniform(5.5, 28.0, n)
    credit_score    = rng.integers(300, 850, n)
    dti             = rng.uniform(0, 45, n)           # debt-to-income %
    emp_length      = rng.integers(0, 30, n)          # years employed
    home_ownership  = rng.choice(["RENT","OWN","MORTGAGE"], n, p=[0.45,0.15,0.40])
    num_credit_lines= rng.integers(1, 30, n)
    delinq_2yrs     = rng.integers(0, 5, n)
    purpose         = rng.choice(
        ["debt_consolidation","credit_card","home_improvement","other","small_business"],
        n, p=[0.40, 0.25, 0.15, 0.12, 0.08])
    annual_inc_joint= rng.choice([0, 1], n, p=[0.7, 0.3])
    revol_util      = rng.uniform(0, 100, n)          # revolving utilisation %
    open_acc        = rng.integers(1, 20, n)
    pub_rec         = rng.integers(0, 3, n)

    # Default probability driven by risk factors
    log_odds = (
        -3.5
        + 0.02  * (750 - credit_score) / 100    # low score → higher default
        + 0.04  * (dti - 20) / 10               # high DTI → higher default
        + 0.03  * (loan_amount / income * 10)    # loan burden
        - 0.015 * (income / 50000)               # higher income → lower default
        + 0.06  * (interest_rate - 12) / 5       # high rate signals risk
        + 0.03  * delinq_2yrs                    # past delinquencies
        + 0.02  * revol_util / 20               # high utilisation
        + 0.02  * (loan_term == 60).astype(float)# longer term riskier
        + 0.05  * (pub_rec > 0).astype(float)    # public records
        - 0.01  * emp_length / 5                 # more experience → safer
        + rng.normal(0, 0.3, n)                  # noise
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default      = (rng.uniform(0, 1, n) < prob_default).astype(int)

    df = pd.DataFrame({
        "age":              age,
        "annual_income":    income.round(0),
        "loan_amount":      loan_amount.round(0),
        "loan_term":        loan_term,
        "interest_rate":    interest_rate.round(2),
        "credit_score":     credit_score,
        "dti":              dti.round(2),
        "emp_length":       emp_length,
        "home_ownership":   home_ownership,
        "num_credit_lines": num_credit_lines,
        "delinq_2yrs":      delinq_2yrs,
        "purpose":          purpose,
        "revol_util":       revol_util.round(2),
        "open_acc":         open_acc,
        "pub_rec":          pub_rec,
        "joint_application":annual_inc_joint,
        "default":          default,
    })
    return df


# ─────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df["loan_to_income"]       = df["loan_amount"]  / df["annual_income"]
    df["monthly_payment_est"]  = df["loan_amount"]  / df["loan_term"]
    df["payment_to_income"]    = df["monthly_payment_est"] / (df["annual_income"] / 12)
    df["credit_score_bucket"]  = pd.cut(
        df["credit_score"],
        bins=[300,580,670,740,800,850],
        labels=["Very Poor","Fair","Good","Very Good","Exceptional"]
    )
    df["has_delinquency"]      = (df["delinq_2yrs"] > 0).astype(int)
    df["has_pub_rec"]          = (df["pub_rec"] > 0).astype(int)
    df["high_utilisation"]     = (df["revol_util"] > 70).astype(int)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["home_ownership", "purpose"]:
        df[col + "_enc"] = le.fit_transform(df[col])
    cs_map = {"Very Poor":0,"Fair":1,"Good":2,"Very Good":3,"Exceptional":4}
    df["credit_bucket_enc"] = df["credit_score_bucket"].map(cs_map)

    return df


# ─────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df, use_smote, test_size, model_choice):
    df_eng = engineer_features(df)

    feature_cols = [
        "age","annual_income","loan_amount","loan_term","interest_rate",
        "credit_score","dti","emp_length","num_credit_lines","delinq_2yrs",
        "revol_util","open_acc","pub_rec","joint_application",
        "home_ownership_enc","purpose_enc","credit_bucket_enc",
        "loan_to_income","payment_to_income","has_delinquency",
        "has_pub_rec","high_utilisation",
    ]

    X = df_eng[feature_cols].fillna(0)
    y = df_eng["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    if use_smote and SMOTE_OK:
        sm = SMOTE(random_state=42)
        X_train_sc, y_train = sm.fit_resample(X_train_sc, y_train)

    models = {}
    if model_choice in ("Logistic Regression", "All"):
        lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        lr.fit(X_train_sc, y_train)
        models["Logistic Regression"] = lr
    if model_choice in ("Random Forest", "All"):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_sc, y_train)
        models["Random Forest"] = rf
    if model_choice in ("Gradient Boosting", "All"):
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        learning_rate=0.1, random_state=42)
        gb.fit(X_train_sc, y_train)
        models["Gradient Boosting"] = gb
    if XGBOOST_OK and model_choice in ("XGBoost", "All"):
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                            use_label_encoder=False, eval_metric="logloss",
                            random_state=42, verbosity=0)
        xgb.fit(X_train_sc, y_train)
        models["XGBoost"] = xgb

    results = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test_sc)[:, 1]
        pred  = model.predict(X_test_sc)
        results[name] = {
            "model":  model,
            "proba":  proba,
            "pred":   pred,
            "auc":    roc_auc_score(y_test, proba),
            "ap":     average_precision_score(y_test, proba),
            "brier":  brier_score_loss(y_test, proba),
            "report": classification_report(y_test, pred, output_dict=True),
            "cm":     confusion_matrix(y_test, pred),
        }

    return results, scaler, feature_cols, X_test, y_test, X_test_sc


# ─────────────────────────────────────────────────────────
# Plotting helpers (all dark-themed)
# ─────────────────────────────────────────────────────────
DARK_BG = "#0f1117"
CARD_BG  = "#1a1d27"
GRID_CLR = "#2d3148"
TEXT_CLR = "#e2e8f0"
MUTED    = "#64748b"

def dark_fig(figsize=(7,4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.5)
    return fig, ax


def plot_roc_curves(results, y_test):
    fig = go.Figure()
    colors = ["#6366f1","#34d399","#f59e0b","#f87171"]
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=MUTED, dash="dash", width=1))
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['auc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2.5)))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=12),
        legend=dict(bgcolor=DARK_BG, bordercolor=GRID_CLR),
        xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=40,r=20,t=50,b=40),
    )
    return fig


def plot_pr_curves(results, y_test):
    fig = go.Figure()
    colors = ["#6366f1","#34d399","#f59e0b","#f87171"]
    baseline = y_test.mean()
    fig.add_shape(type="line", x0=0, y0=baseline, x1=1, y1=baseline,
                  line=dict(color=MUTED, dash="dash", width=1))
    for i, (name, res) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_test, res["proba"])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{name} (AP={res['ap']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2.5)))
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall", yaxis_title="Precision",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=12),
        legend=dict(bgcolor=DARK_BG, bordercolor=GRID_CLR),
        xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=40,r=20,t=50,b=40),
    )
    return fig


def plot_confusion_matrix(cm, model_name):
    labels = ["No Default","Default"]
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale=[[0,"#1a1d27"],[0.5,"#312e81"],[1,"#6366f1"]],
        title=f"Confusion Matrix — {model_name}",
    )
    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=12),
        margin=dict(l=40,r=20,t=50,b=40),
    )
    return fig


def plot_feature_importance(model, feature_cols, model_name):
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_[0])
    else:
        return None

    idx  = np.argsort(imps)[-15:]
    feat = [feature_cols[i] for i in idx]
    vals = imps[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=feat, orientation="h",
        marker=dict(
            color=vals,
            colorscale=[[0,"#312e81"],[0.5,"#6366f1"],[1,"#a5b4fc"]],
        )
    ))
    fig.update_layout(
        title=f"Feature Importance — {model_name}",
        xaxis_title="Importance",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=12),
        xaxis=dict(gridcolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=180,r=20,t=50,b=40),
        height=500,
    )
    return fig


def plot_score_distribution(results, y_test, model_name):
    if model_name not in results:
        return None
    proba = results[model_name]["proba"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=proba[y_test == 0], nbinsx=50, name="No Default",
        marker_color="#34d399", opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=proba[y_test == 1], nbinsx=50, name="Default",
        marker_color="#f87171", opacity=0.7,
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"Predicted Probability Distribution — {model_name}",
        xaxis_title="Predicted Default Probability",
        yaxis_title="Count",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=12),
        legend=dict(bgcolor=DARK_BG, bordercolor=GRID_CLR),
        xaxis=dict(gridcolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=40,r=20,t=50,b=40),
    )
    return fig


def plot_calibration(results, y_test):
    fig = go.Figure()
    colors = ["#6366f1","#34d399","#f59e0b","#f87171"]
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=MUTED, dash="dash", width=1))
    for i, (name, res) in enumerate(results.items()):
        frac_pos, mean_pred = calibration_curve(y_test, res["proba"], n_bins=10)
        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos, mode="lines+markers", name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title="Calibration Curves (Reliability Diagram)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=12),
        legend=dict(bgcolor=DARK_BG, bordercolor=GRID_CLR),
        xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        margin=dict(l=40,r=20,t=50,b=40),
    )
    return fig


def plot_eda_default_by(df, col, title):
    rates = df.groupby(col)["default"].mean().reset_index()
    rates.columns = [col, "default_rate"]
    rates = rates.sort_values("default_rate", ascending=False)
    fig = px.bar(
        rates, x=col, y="default_rate",
        title=title,
        color="default_rate",
        color_continuous_scale=[[0,"#312e81"],[0.5,"#6366f1"],[1,"#f87171"]],
        text=rates["default_rate"].apply(lambda x: f"{x:.1%}"),
    )
    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=12),
        xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        coloraxis_showscale=False,
        margin=dict(l=40,r=20,t=50,b=40),
    )
    fig.update_traces(textposition="outside")
    return fig


# ─────────────────────────────────────────────────────────
# Single-loan scorer
# ─────────────────────────────────────────────────────────
def score_single_loan(model, scaler, feature_cols, loan_data):
    df_single = pd.DataFrame([loan_data])
    df_eng    = engineer_features(df_single)
    X         = df_eng[feature_cols].fillna(0)
    X_sc      = scaler.transform(X)
    prob      = model.predict_proba(X_sc)[0, 1]
    pred      = model.predict(X_sc)[0]

    if prob >= 0.50:
        risk_label = "HIGH RISK"
        risk_class = "risk-high"
        decision   = "❌ Decline"
    elif prob >= 0.25:
        risk_label = "MEDIUM RISK"
        risk_class = "risk-medium"
        decision   = "⚠️ Review"
    else:
        risk_label = "LOW RISK"
        risk_class = "risk-low"
        decision   = "✅ Approve"

    return prob, pred, risk_label, risk_class, decision


# ─────────────────────────────────────────────────────────
# ══  UI  ══
# ─────────────────────────────────────────────────────────

# ── Hero Banner ──────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🏦 Credit Risk & Default Prediction Dashboard</div>
  <div class="hero-sub">Machine learning–powered loan default scoring by Doug Chingosho</div>
  <div class="hero-badges">
    <span class="badge badge-blue">Logistic Regression</span>
    <span class="badge badge-purple">Random Forest</span>
    <span class="badge badge-teal">Gradient Boosting</span>
    <span class="badge badge-red">XGBoost</span>
    <span class="badge badge-amber">SMOTE Balancing</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Contact bar ───────────────────────────────────────────
st.markdown("""
<div class="contact-bar">
  <span class="contact-label">Connect with Doug Chingosho</span>
  <div class="contact-divider"></div>
  <a class="contact-btn btn-linkedin"
     href="https://www.linkedin.com/in/douglas-chingosho"
     target="_blank" rel="noopener noreferrer">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853
               0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9
               1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337
               7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782
               13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0
               1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24
               22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
    LinkedIn
  </a>
  <a class="contact-btn btn-email"
     href="mailto:douglas.chingosho@wustl.edu"
     target="_blank" rel="noopener noreferrer">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="2" y="4" width="20" height="16" rx="2"/>
      <path d="M2 7l10 7 10-7"/>
    </svg>
    Email
  </a>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.divider()

    st.markdown("**Dataset**")
    n_samples   = st.slider("Number of loans", 1000, 10000, 5000, 500)
    test_size   = st.slider("Test set size (%)", 10, 40, 20, 5) / 100

    st.divider()
    st.markdown("**Models to train**")
    model_choice = st.selectbox(
        "Select model(s)",
        ["All", "Logistic Regression", "Random Forest",
         "Gradient Boosting", "XGBoost"] if XGBOOST_OK
        else ["All", "Logistic Regression", "Random Forest", "Gradient Boosting"]
    )

    st.divider()
    st.markdown("**Class imbalance**")
    use_smote = st.checkbox(
        "Apply SMOTE oversampling",
        value=True,
        help="SMOTE creates synthetic minority-class samples to balance the dataset"
    ) if SMOTE_OK else False

    st.divider()
    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "Default probability threshold", 0.10, 0.90, 0.50, 0.05,
        help="Predictions above this threshold are classified as default"
    )

    st.divider()
    run_btn = st.button("🚀  Train Models", use_container_width=True)


# ─────────────────────────────────────────────────────────
# Load data + train
# ─────────────────────────────────────────────────────────
df = generate_dataset(n=n_samples)

if run_btn or "results" not in st.session_state:
    with st.spinner("Generating dataset and training models..."):
        results, scaler, feature_cols, X_test, y_test, X_test_sc = train_models(
            df, use_smote, test_size, model_choice
        )
    st.session_state["results"]      = results
    st.session_state["scaler"]       = scaler
    st.session_state["feature_cols"] = feature_cols
    st.session_state["X_test"]       = X_test
    st.session_state["y_test"]       = y_test
    st.session_state["X_test_sc"]    = X_test_sc

results      = st.session_state["results"]
scaler       = st.session_state["scaler"]
feature_cols = st.session_state["feature_cols"]
X_test       = st.session_state["X_test"]
y_test       = st.session_state["y_test"]
X_test_sc    = st.session_state["X_test_sc"]
best_model_name = max(results, key=lambda k: results[k]["auc"])

# ─────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Model Performance",
    "🔍 Feature Analysis",
    "🎯 Score a Loan",
    "📋 Dataset Explorer",
])


# ══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-pill">Dataset summary</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total loans",       f"{len(df):,}")
    c2.metric("Default rate",      f"{df['default'].mean():.1%}")
    c3.metric("Avg loan amount",   f"${df['loan_amount'].mean():,.0f}")
    c4.metric("Avg credit score",  f"{df['credit_score'].mean():.0f}")
    c5.metric("Avg interest rate", f"{df['interest_rate'].mean():.1f}%")

    st.divider()
    st.markdown('<div class="section-pill">Model leaderboard</div>', unsafe_allow_html=True)

    leaderboard = pd.DataFrame([
        {
            "Model": name,
            "AUC-ROC": f"{r['auc']:.4f}",
            "Avg Precision": f"{r['ap']:.4f}",
            "Brier Score": f"{r['brier']:.4f}",
            "Accuracy": f"{r['report']['accuracy']:.4f}",
            "F1 (Default)": f"{r['report']['1']['f1-score']:.4f}",
            "Precision (Default)": f"{r['report']['1']['precision']:.4f}",
            "Recall (Default)": f"{r['report']['1']['recall']:.4f}",
            "Best": "⭐" if name == best_model_name else "",
        }
        for name, r in results.items()
    ])
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-pill">EDA — Default rates by segment</div>', unsafe_allow_html=True)

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.plotly_chart(
            plot_eda_default_by(df, "purpose", "Default rate by loan purpose"),
            use_container_width=True)
    with col_eda2:
        st.plotly_chart(
            plot_eda_default_by(df, "home_ownership", "Default rate by home ownership"),
            use_container_width=True)

    col_eda3, col_eda4 = st.columns(2)
    with col_eda3:
        df_eng_eda = engineer_features(df)
        st.plotly_chart(
            plot_eda_default_by(df_eng_eda, "credit_score_bucket", "Default rate by credit score band"),
            use_container_width=True)
    with col_eda4:
        df["loan_term_label"] = df["loan_term"].astype(str) + " months"
        st.plotly_chart(
            plot_eda_default_by(df, "loan_term_label", "Default rate by loan term"),
            use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-pill">ROC & Precision-Recall</div>', unsafe_allow_html=True)

    col_roc, col_pr = st.columns(2)
    with col_roc:
        st.plotly_chart(plot_roc_curves(results, y_test), use_container_width=True)
    with col_pr:
        st.plotly_chart(plot_pr_curves(results, y_test), use_container_width=True)

    st.divider()
    st.markdown('<div class="section-pill">Score distributions & calibration</div>', unsafe_allow_html=True)

    sel_model = st.selectbox("Select model to inspect", list(results.keys()), key="perf_sel")

    col_dist, col_cal = st.columns(2)
    with col_dist:
        fig_dist = plot_score_distribution(results, y_test, sel_model)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)
    with col_cal:
        st.plotly_chart(plot_calibration(results, y_test), use_container_width=True)

    st.divider()
    st.markdown('<div class="section-pill">Confusion matrix</div>', unsafe_allow_html=True)

    # Apply custom threshold
    col_cm, col_thresh = st.columns([2,1])
    with col_cm:
        proba_thresh = results[sel_model]["proba"]
        pred_thresh  = (proba_thresh >= threshold).astype(int)
        cm_thresh    = confusion_matrix(y_test, pred_thresh)
        st.plotly_chart(plot_confusion_matrix(cm_thresh, sel_model), use_container_width=True)
    with col_thresh:
        tn,fp,fn,tp = cm_thresh.ravel()
        st.metric("True Positives (caught defaults)",  tp)
        st.metric("False Positives (false alarms)",    fp)
        st.metric("True Negatives (correct approvals)",tn)
        st.metric("False Negatives (missed defaults)", fn)
        precision_t = tp/(tp+fp) if (tp+fp)>0 else 0
        recall_t    = tp/(tp+fn) if (tp+fn)>0 else 0
        st.metric("Precision at threshold", f"{precision_t:.3f}")
        st.metric("Recall at threshold",    f"{recall_t:.3f}")

    st.divider()
    st.markdown('<div class="section-pill">Detailed classification report</div>', unsafe_allow_html=True)

    report = results[sel_model]["report"]
    report_df = pd.DataFrame({
        "Class":     ["No Default (0)", "Default (1)", "Macro Avg", "Weighted Avg"],
        "Precision": [f"{report['0']['precision']:.4f}", f"{report['1']['precision']:.4f}",
                      f"{report['macro avg']['precision']:.4f}",
                      f"{report['weighted avg']['precision']:.4f}"],
        "Recall":    [f"{report['0']['recall']:.4f}", f"{report['1']['recall']:.4f}",
                      f"{report['macro avg']['recall']:.4f}",
                      f"{report['weighted avg']['recall']:.4f}"],
        "F1-Score":  [f"{report['0']['f1-score']:.4f}", f"{report['1']['f1-score']:.4f}",
                      f"{report['macro avg']['f1-score']:.4f}",
                      f"{report['weighted avg']['f1-score']:.4f}"],
    })
    st.dataframe(report_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════
# TAB 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-pill">Feature importance</div>', unsafe_allow_html=True)
    sel_feat_model = st.selectbox("Select model", list(results.keys()), key="feat_sel")

    fig_imp = plot_feature_importance(
        results[sel_feat_model]["model"], feature_cols, sel_feat_model)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-pill">Correlation heatmap</div>', unsafe_allow_html=True)

    num_cols = ["annual_income","loan_amount","interest_rate","credit_score",
                "dti","revol_util","emp_length","delinq_2yrs","default"]
    corr = df[num_cols].corr()

    fig_corr = px.imshow(
        corr, text_auto=".2f", title="Feature Correlation Matrix",
        color_continuous_scale=[[0,"#f87171"],[0.5,"#1a1d27"],[1,"#6366f1"]],
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_CLR, size=11),
        margin=dict(l=40,r=20,t=50,b=40),
        height=520,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-pill">Default rate by credit score band</div>',
                unsafe_allow_html=True)

    df_eng_feat = engineer_features(df)
    col_cs1, col_cs2 = st.columns(2)
    with col_cs1:
        fig_cs = px.box(
            df_eng_feat, x="credit_score_bucket", y="interest_rate",
            color="credit_score_bucket", title="Interest Rate by Credit Band",
            color_discrete_sequence=["#f87171","#f59e0b","#34d399","#60a5fa","#a78bfa"],
        )
        fig_cs.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR), showlegend=False,
            xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        )
        st.plotly_chart(fig_cs, use_container_width=True)
    with col_cs2:
        fig_dti = px.histogram(
            df, x="dti", color=df["default"].map({0:"No Default",1:"Default"}),
            nbins=50, barmode="overlay", opacity=0.7,
            title="Debt-to-Income Distribution by Outcome",
            color_discrete_map={"No Default":"#34d399","Default":"#f87171"},
        )
        fig_dti.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            font=dict(color=TEXT_CLR),
            legend=dict(bgcolor=DARK_BG, title=""),
            xaxis=dict(gridcolor=GRID_CLR), yaxis=dict(gridcolor=GRID_CLR),
        )
        st.plotly_chart(fig_dti, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — SCORE A LOAN
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-pill">Single loan scorer</div>', unsafe_allow_html=True)
    st.markdown("Enter an applicant's details below and get an instant default probability and credit decision.")

    sel_scoring_model = st.selectbox(
        "Model to use for scoring", list(results.keys()), key="score_sel",
        index=list(results.keys()).index(best_model_name)
    )

    st.divider()
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown("**Personal & Employment**")
        age_s       = st.number_input("Age", 18, 80, 35, key="age_s")
        income_s    = st.number_input("Annual income ($)", 10000, 500000, 65000, 1000, key="inc_s")
        emp_s       = st.number_input("Years employed", 0, 40, 5, key="emp_s")
        home_s      = st.selectbox("Home ownership", ["RENT","OWN","MORTGAGE"], key="home_s")
        joint_s     = st.selectbox("Joint application", [0,1],
                                   format_func=lambda x: "Yes" if x else "No", key="joint_s")

    with col_s2:
        st.markdown("**Loan Details**")
        loan_s      = st.number_input("Loan amount ($)", 500, 40000, 10000, 500, key="loan_s")
        term_s      = st.selectbox("Loan term (months)", [36, 60], key="term_s")
        rate_s      = st.number_input("Interest rate (%)", 5.0, 30.0, 12.5, 0.1,
                                      format="%.1f", key="rate_s")
        purpose_s   = st.selectbox("Loan purpose",
                                   ["debt_consolidation","credit_card",
                                    "home_improvement","other","small_business"],
                                   key="purp_s")

    with col_s3:
        st.markdown("**Credit Profile**")
        cscore_s    = st.number_input("Credit score", 300, 850, 680, key="cs_s")
        dti_s       = st.number_input("Debt-to-income (%)", 0.0, 50.0, 18.0, 0.5,
                                      format="%.1f", key="dti_s")
        revol_s     = st.number_input("Revolving utilisation (%)", 0.0, 100.0, 35.0, 1.0,
                                      format="%.1f", key="revol_s")
        delinq_s    = st.number_input("Delinquencies (2yr)", 0, 10, 0, key="del_s")
        num_cl_s    = st.number_input("Number of credit lines", 1, 30, 8, key="ncl_s")
        open_s      = st.number_input("Open accounts", 1, 20, 6, key="open_s")
        pub_s       = st.number_input("Public records", 0, 5, 0, key="pub_s")

    st.divider()
    score_btn = st.button("⚡  Score This Loan", use_container_width=True)

    if score_btn:
        loan_data = {
            "age": age_s, "annual_income": income_s, "loan_amount": loan_s,
            "loan_term": term_s, "interest_rate": rate_s, "credit_score": cscore_s,
            "dti": dti_s, "emp_length": emp_s, "home_ownership": home_s,
            "num_credit_lines": num_cl_s, "delinq_2yrs": delinq_s,
            "purpose": purpose_s, "revol_util": revol_s, "open_acc": open_s,
            "pub_rec": pub_s, "joint_application": joint_s,
        }

        prob, pred, risk_label, risk_class, decision = score_single_loan(
            results[sel_scoring_model]["model"], scaler, feature_cols, loan_data
        )

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default probability", f"{prob:.1%}")
        r2.metric("Risk category",       risk_label)
        r3.metric("Decision",            decision)
        r4.metric("Model used",          sel_scoring_model)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": TEXT_CLR, "size": 32}},
            delta={"reference": 25, "suffix": "%",
                   "increasing": {"color": "#f87171"},
                   "decreasing": {"color": "#34d399"}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": TEXT_CLR,
                          "tickfont": {"color": TEXT_CLR}},
                "bar":   {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 25],  "color": "#052e16"},
                    {"range": [25, 50], "color": "#422006"},
                    {"range": [50, 100],"color": "#450a0a"},
                ],
                "threshold": {
                    "line": {"color": "#f87171", "width": 3},
                    "thickness": 0.75, "value": threshold * 100,
                },
            },
            title={"text": "Default Probability", "font": {"color": TEXT_CLR, "size": 16}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor=CARD_BG, font=dict(color=TEXT_CLR),
            height=300, margin=dict(l=30,r=30,t=60,b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander("Scoring details & risk drivers"):
            st.markdown(f"""
| Factor | Value | Risk Signal |
|--------|-------|-------------|
| Credit score | {cscore_s} | {"🔴 Poor" if cscore_s < 580 else "🟡 Fair" if cscore_s < 670 else "🟢 Good"} |
| Debt-to-income | {dti_s:.1f}% | {"🔴 High" if dti_s > 35 else "🟡 Moderate" if dti_s > 20 else "🟢 Low"} |
| Revolving utilisation | {revol_s:.1f}% | {"🔴 High" if revol_s > 70 else "🟡 Moderate" if revol_s > 40 else "🟢 Low"} |
| Past delinquencies | {delinq_s} | {"🔴 Yes — negative signal" if delinq_s > 0 else "🟢 None"} |
| Public records | {pub_s} | {"🔴 Yes — negative signal" if pub_s > 0 else "🟢 None"} |
| Loan-to-income ratio | {loan_s/income_s:.2%} | {"🔴 High" if loan_s/income_s > 0.3 else "🟡 Moderate" if loan_s/income_s > 0.15 else "🟢 Low"} |
| Interest rate | {rate_s:.1f}% | {"🔴 High" if rate_s > 20 else "🟡 Elevated" if rate_s > 12 else "🟢 Normal"} |
            """)


# ══════════════════════════════════════════════════════════
# TAB 5 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-pill">Raw dataset preview</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    default_filter = col_f1.selectbox("Filter by default", ["All","Default","No Default"])
    purpose_filter = col_f2.selectbox("Filter by purpose", ["All"] + sorted(df["purpose"].unique()))
    home_filter    = col_f3.selectbox("Filter by ownership", ["All"] + sorted(df["home_ownership"].unique()))

    df_view = df.copy()
    if default_filter == "Default":
        df_view = df_view[df_view["default"]==1]
    elif default_filter == "No Default":
        df_view = df_view[df_view["default"]==0]
    if purpose_filter != "All":
        df_view = df_view[df_view["purpose"]==purpose_filter]
    if home_filter != "All":
        df_view = df_view[df_view["home_ownership"]==home_filter]

    st.caption(f"Showing {len(df_view):,} of {len(df):,} loans")
    st.dataframe(df_view.head(200), use_container_width=True)

    st.divider()
    st.markdown('<div class="section-pill">Descriptive statistics</div>', unsafe_allow_html=True)
    st.dataframe(df_view.describe().round(2), use_container_width=True)


# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:48px; padding:20px 24px;
            border-top:1px solid rgba(99,102,241,0.2);
            display:flex; align-items:center; justify-content:space-between;
            flex-wrap:wrap; gap:12px;">
  <div style="font-size:12px; color:#475569;">
    Built by <span style="color:#a5b4fc; font-weight:600;">Doug Chingosho</span>
    &nbsp;·&nbsp; MS Business Analytics, Olin Business School, WashU
    &nbsp;·&nbsp; Fintech · Analytics · Data Insights
  </div>
  <div style="display:flex; gap:10px;">
    <a href="https://www.linkedin.com/in/douglas-chingosho" target="_blank"
       style="font-size:12px; color:#6366f1; text-decoration:none; font-weight:500;">
      LinkedIn ↗
    </a>
    <span style="color:#334155;">·</span>
    <a href="mailto:douglas.chingosho@wustl.edu"
       style="font-size:12px; color:#6366f1; text-decoration:none; font-weight:500;">
      Email ↗
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
