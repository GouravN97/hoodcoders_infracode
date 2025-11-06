# HoodCoders — Aurigo Infracode Synergy ’25 Submission

[](https://www.python.org/)
[](https://streamlit.io/)
[](https://xgboost.ai/)
[](https://shap.readthedocs.io/en/latest/)
[](https://www.google.com/search?q=LICENSE)

> ⚙️ An **AI-powered infrastructure prediction & risk insight system** built for
> **Aurigo Infracode Synergy ’25** — using **Streamlit + FastAPI + XGBoost + SHAP**

-----

## 🌍 Overview

**HoodCoders** is a decision-support platform that predicts & explains outcomes for infrastructure projects:

  * 💸 **Actual Cost**
  * 🕒 **Delay (Months)**
  * 💰 **ROI Realized**
  * ⚖️ **Priority Category**

Every result is backed by:

  * ✅ SHAP explainability
  * ✅ Risk scores (1-5 scale)
  * ✅ Natural-language reasoning

Designed for **policy makers, infra analysts, and EPC teams.**

-----

## 💡 Why This Matters

Big infrastructure = big risk.
Cost overruns. Time overruns. Funding inefficiency.

Our tool converts raw project input → **transparent insights** that help governments & orgs plan better.

-----

## ✨ Features

| Feature | Description |
| --- | --- |
| 🔮 Multi-model prediction | Cost, Delay, ROI & Priority |
| 🧠 Explainable AI | SHAP waterfall + top drivers |
| 🧩 Future-UI Streamlit Dashboard | Glassmorphic + animations |
| 📂 Scenario Saving | Save & compare project cases |
| 📊 Visual Explanation | SHAP plots + heat maps |
| 📝 Human Insights | Plain English interpretation |

-----

## 🧠 Tech Stack

| Layer | Tools |
| --- | --- |
| Frontend | Streamlit (custom CSS, animations) |
| Backend | FastAPI |
| ML | XGBoost, Scikit-learn |
| Explainability | SHAP, Matplotlib |
| Data | Pandas, NumPy |
| Storage | Joblib |

-----

## 🧩 Architecture

User Inputs → Encoding → ML Prediction → SHAP → Risk Logic → Dashboard Output

```yaml
Pipeline:
1. Input form
2. Feature encoding & computation
3. XGBoost model prediction
4. SHAP explainability
5. Rating & reasoning
6. Visual dashboard
```

-----

## 🖥️ Screenshots

| Landing Page | Dashboard |
| --- | --- |
| *(UI Preview Placeholder)* | *(Plot + Metrics Placeholder)* |

> 🪩 Sleek glassmorphism + neon UI — modern & intuitive

-----

## ⚙️ Installation

### Clone repo

```bash
git clone https://github.com/<your-username>/hoodcoders.git
cd hoodcoders
```

### Install packages

```bash
pip install -r requirements.txt
```

### Start backend

```bash
python api.py
```

### Run UI

```bash
streamlit run app.py
```

-----

## 📁 File Structure

```bash
├── app.py                     # Entry point router
├── page1.py                   # Animated landing page
├── page.py                    # Form + prediction
├── explanation_layer.py       # SHAP + reasoning logic
├── models/                    # XGBoost models
├── encoders/                  # Encoding pipelines
├── background.png             # UI asset
├── requirements.txt
└── README.md
```

-----

## 📊 Rating System (1-5)

| Score | Meaning |
| --- | --- |
| 5 | Excellent — very stable |
| 4 | Good — manageable risk |
| 3 | Neutral — mixed factors |
| 2 | Risky — review suggested |
| 1 | Critical — not advisable |

-----

## 🧠 SHAP Explainability Flow

  * Identify most impactful project factors
  * Show +/− impact on result
  * Provide natural language summary
  * Display waterfall + summary plots

**Example Output:**

```text
Score: 4.5 (Excellent)
Drivers: Funding efficiency ↑, Feasibility ↑
Interpretation: Strong financial + sustainability indicators.
```

-----

## ⏭️ Future Add-ons

  * Live dataset integration
  * Auto-retraining pipeline
  * User logins & profiles
  * Cloud deployment (Azure/Streamlit Cloud)
  * Multi-language UI

-----

## 👨‍💻 Team

  * **Team:** HoodCoders
  * **Institute:** IIIT Bangalore
  * **Event:** Aurigo Infracode Synergy 2025

> “AI that builds trust, not just predictions.”

-----

## 📝 License

MIT — free to use & modify

-----

-----

⭐ If you like this project, drop a star on GitHub\!

🔥 Built with caffeine, curiosity, and code — by **HoodCoders**
