# 💳 Credit Risk & Loan Default Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doug-credit-risk-default-prediction.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive machine learning dashboard for predicting loan default risk using ensemble methods. Built with Python and Streamlit, it provides real-time probability scores, model interpretability via SHAP values, and risk segmentation across four categories.

🔗 **Live App:** [doug-credit-risk-default-prediction.streamlit.app](https://doug-credit-risk-default-prediction.streamlit.app/)

---

## 📸 Preview

> Interactive dashboard with real-time predictions, SHAP feature importance charts, ROC curves, and risk segmentation panels.

---

## 🎯 Features

- **Ensemble Modeling** — XGBoost, Random Forest, and Logistic Regression with weighted voting
- **Real-Time Predictions** — Instant default probability score for any borrower profile
- **SHAP Interpretability** — Feature importance explanations for every individual prediction
- **Risk Segmentation** — Borrowers classified as Low / Medium / High / Very High risk
- **Class Imbalance Handling** — SMOTE oversampling for balanced training
- **Interactive Visualizations** — ROC curves, confusion matrix, feature distributions
- **Model Comparison** — Side-by-side performance metrics across all three models

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87% |
| ROC-AUC | 0.91 |
| Precision | 83% |
| Recall | 79% |
| F1-Score | 0.81 |
| Dataset Size | 250,000+ loan applications |

---

## 🔝 Top Predictive Features

| Feature | Importance |
|---------|-----------|
| Credit Score (FICO) | 25% |
| Debt-to-Income Ratio | 18% |
| Employment Length | 12% |
| Loan Amount | 10% |
| Credit Inquiries (Last 6 Months) | 8% |
| Payment History | 7% |
| Credit Utilization | 6% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Web Framework | Streamlit |
| ML Models | XGBoost, Scikit-learn (Random Forest, Logistic Regression) |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Interpretability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualizations | Plotly, Matplotlib, Seaborn |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Doug1192/credit-risk-dashboard.git
cd credit-risk-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run credit_risk.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
credit-risk-dashboard/
│
├── credit_risk.py          # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── data/
│   └── loan_data.csv       # Lending Club loan dataset
│
├── models/
│   └── trained_models.pkl  # Pre-trained model artifacts
│
└── utils/
    ├── preprocessing.py    # Feature engineering pipeline
    └── evaluation.py       # Model evaluation utilities
```

---

## 💡 How It Works

```python
# Core prediction pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# 1. Feature Engineering — 50+ features including:
#    - Credit history, DTI ratio, employment length
#    - Loan amount, interest rate, installment
#    - Delinquency history, public records

# 2. Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. Train ensemble models
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
rf_model  = RandomForestClassifier(n_estimators=100, max_depth=10)
lr_model  = LogisticRegression(C=1.0, max_iter=1000)

# 4. Weighted voting for final prediction
# 5. SHAP values for individual explainability
```

---

## 📈 Business Impact

- Identifies **79% of defaults** before they occur, reducing bad debt exposure
- Enables approval of creditworthy borderline applicants who would otherwise be declined
- Reduces loan processing time from days to **under one minute**
- SHAP explanations ensure **regulatory compliance** and model transparency
- Risk tiers allow lenders to price loans appropriately by risk segment

---

## 🎓 Academic Context

Built as part of the **MS Financial Technology Analytics** program at Washington University in St. Louis, Olin Business School. Demonstrates applied machine learning in a real-world financial services context, combining statistical rigor with production-ready deployment.

---

## 👤 Author

**Douglas Tawanda Chingosho**
MS Business Analytics — Financial Technology Analytics Track
Washington University in St. Louis, Olin Business School

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/douglas-tawanda-chingosho)
[![GitHub](https://img.shields.io/badge/GitHub-doug1192-black)](https://github.com/doug1192)
[![Portfolio](https://img.shields.io/badge/Portfolio-dougchingosho.com-blue)](https://dougchingosho.com)
[![Email](https://img.shields.io/badge/Email-douglasc%40wustl.edu-red)](mailto:douglasc@wustl.edu)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
