# Supply Chain Intelligence

This project provides a comprehensive ML backend and core logic for supply chain intelligence, risk management, and explainability, along with a front-end dashboard built with Streamlit.

## Directory Structure

```
supply-chain-intelligence/
│
├── data/                        ← Data assets
│   ├── raw/                     ← The original, immutable data dump
│   ├── external/                ← Data from third party sources
│   └── processed/               ← The final, canonical data sets for modeling
│
├── src/                         ← ML BACKEND / CORE LOGIC
│   ├── preprocessing/           ← Scripts to turn raw data into features
│   ├── integration/             ← Data integration pipelines
│   ├── features/                ← Scripts to turn processed data into features
│   ├── models/                  ← ML Models architecture and logic
│   ├── explainability/          ← Model explainability and interpretability modules
│   └── risk_engine/             ← Supply chain risk analysis engine
│
├── models/                      ← SAVED TRAINED MODELS
│
├── dashboard/                   ← FRONTEND / USER INTERFACE
│
├── results/                     ← OUTPUTS / EVALUATION
│
├── config/                      ← CONFIGURATION
│
├── main.py                      ← MAIN PIPELINE ENTRY POINT
└── requirements.txt             ← Project dependencies
```

## Running the Dashboard

To run the Streamlit front-end dashboard:
```bash
streamlit run dashboard/app.py
```
