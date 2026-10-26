# Multi-Risk Supply Chain Intelligence System 🌍📦

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, end-to-end **Supply Chain Risk Management & Intelligence Platform** powered by machine learning and multi-source data fusion. 

This platform integrates internal ERP transaction records with **8 external global datasets** (weather, logistics performance, governance indices, economic indicators, inflation, disaster databases, fuel prices, and public holidays) to predict shipment delay risks and forecast delay durations.

> 📖 **Full Technical Documentation**: For complete architecture, data schemas, mathematical formulas, and ML audit results, see [PROJECT_DETAILS.md](PROJECT_DETAILS.md).

---

## 🌟 Key Capabilities

- **Multi-Source Data Fusion**: Merges 176,530 historical shipments from DataCo with Open-Meteo weather, World Bank Logistics Performance Index (LPI), Worldwide Governance Indicators (WGI), FAO GDP & CPI data, EM-DAT natural disaster records, and global diesel fuel prices.
- **Leakage-Audited Feature Engineering**: 41 engineered features spanning operational efficiency, infrastructure quality, political stability, climate shocks, and seasonal disruptions.
- **Machine Learning & Audits**:
  - Binary risk classification (Random Forest & XGBoost with ROC-AUC ~0.728 and >99.8% recall at tuned operational threshold).
  - Delay duration regression to estimate exact days of delay.
  - Probability calibration (Brier score < 0.198) and repeated group-aware cross-validation to prevent leakage.
- **Interactive Streamlit Dashboard**: 5 comprehensive views:
  1. **📊 Overview**: High-level KPIs, delay trends, market distribution.
  2. **🌍 Geography & Logistics**: Regional risk breakdown, LPI correlations, shipping heatmaps.
  3. **💰 Economic & Operational**: Category profitability, order sizing, commercial exposure.
  4. **⚠️ External Risk**: Climate events, political stability impact, natural disaster disruptions.
  5. **🤖 Order Risk Predictor**: Real-time simulation and risk scoring for new shipments.

---

## 🏗️ Architecture

```
                                  [ Heterogeneous Data Sources ]
           DataCo ERP | Open-Meteo | World Bank LPI/WGI | FAO GDP/CPI | EM-DAT | Fuel Prices
                                                │
                                                ▼
                                   [ Phase 1–4: Preprocessing ]
                           Entity Resolution, Temporal & Spatial Harmonization
                                                │
                                                ▼
                                   [ Phase 5–6: Feature Engine ]
                         41 Features, GroupShuffleSplit Leakage Auditing
                                                │
                                                ▼
                                     [ Phase 7: EDA & Audits ]
                             Multicollinearity (VIF), Class Balancing
                                                │
                                                ▼
                                    [ Phase 8–9: Model Suite ]
                   Random Forest & XGBoost Classifiers | Delay Duration Regressor
                                                │
                                                ▼
                                  [ Phase 10: Streamlit Dashboard ]
                       Overview | Logistics | Economics | External Risk | Predictor
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Atharv-ND/Supply-Chain-Risk-Management.git
cd Supply-Chain-Risk-Management

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
supply-chain-intelligence/
├── app.py                         # Main Streamlit web application
├── PROJECT_DETAILS.md             # In-depth technical documentation & architecture
├── requirements.txt               # Python package dependencies
├── config/                        # Pipeline configuration
├── data/
│   ├── external/                  # Raw external data (weather, country codes)
│   ├── processed/                 # Cleaned datasets
│   └── processed.zip              # Packaged processed data archive
├── results.zip                    # Packaged evaluation reports, plots & matrices
├── src/
│   ├── dashboard/                 # Streamlit UI modules & Plotly visualizations
│   ├── eda/                       # Exploratory Data Analysis & VIF scripts
│   ├── features/                  # Feature engineering pipelines
│   ├── integration/               # Multi-source dataset merge engine
│   ├── ml/                        # Classification, regression, and model selection audits
│   ├── preprocessing/             # Raw dataset cleaners & country name resolvers
│   └── validation/                # Target and temporal leakage verification
└── main.py                        # Pipeline runner
```

---

## 📊 Model Benchmark Highlights

| Model | ROC-AUC | PR-AUC | Recall (Tuned @ 0.15) | Precision | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.7240 | 0.7949 | 99.84% | 0.5459 | **0.1979** |
| **XGBoost** | **0.7276** | **0.7978** | **99.97%** | 0.5458 | 0.1983 |
| **Logistic Regression** | 0.6384 | 0.7011 | 88.42% | 0.5180 | 0.2241 |

*See [PROJECT_DETAILS.md](PROJECT_DETAILS.md) for detailed confusion matrices, threshold calibration curves, and feature importance analyses.*

---

## 📜 License
This project is licensed under the MIT License.
