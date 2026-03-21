# Calgary 311 service request router

## Problem statement

Calgary receives over 1.7 million 311 service requests annually covering road maintenance, waste collection, bylaw complaints, and more. Manually routing each request to the correct department introduces delays and misrouting. This project uses machine learning on 500,000+ historical requests to automatically predict the responsible department, enabling faster and more accurate routing.

## Approach

- Fetched 311 service request data from Calgary Open Data (dataset `iahh-g8bj`)
- Parsed timestamps, computed resolution times, and extracted temporal features
- Engineered community-level aggregates and service type frequency encoding
- Trained Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers
- Evaluated with accuracy, weighted F1, and macro F1 across 15 department classes

## Key results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| Accuracy | ~0.80 |
| Weighted F1 | ~0.79 |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
project_11_311_service_request_router/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    └── model.py
```
