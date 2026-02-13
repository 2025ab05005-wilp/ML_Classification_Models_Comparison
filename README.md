# Machine Learning Assignment 2

## Problem statement
Build and compare six classification models to predict obesity levels using the provided dataset, and present evaluation metrics in a Streamlit app.
Dataset : https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

## Dataset description
- Source: UCI Machine Learning Repository – Estimation of obesity levels based on eating habits and physical condition
- Size: 2111 rows, 17 columns (16 features + 1 target)
- Target column: `NObeyesdad` (multi-class)
- Features: mix of numeric and categorical variables

## Models used (metrics on 80/20 split)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8676 | 0.9814 | 0.8643 | 0.8653 | 0.8641 | 0.8456 |
| Decision Tree | 0.9362 | 0.9623 | 0.9358 | 0.9351 | 0.9351 | 0.9256 |
| KNN | 0.8322 | 0.9617 | 0.8270 | 0.8292 | 0.8150 | 0.8073 |
| Naive Bayes | 0.5390 | 0.8469 | 0.5502 | 0.5372 | 0.4827 | 0.4861 |
| Random Forest (Ensemble) | 0.9314 | 0.9942 | 0.9325 | 0.9310 | 0.9308 | 0.9203 |
| XGBoost (Ensemble) | 0.9740 | 0.9991 | 0.9737 | 0.9743 | 0.9739 | 0.9697 |

## Observations
| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong baseline with high AUC, but lower than tree ensembles. |
| Decision Tree | Good accuracy, but slightly less stable than ensemble methods. |
| KNN | Moderate performance; sensitive to feature scaling. |
| Naive Bayes | Lowest scores; class-conditional independence is too simplistic here. |
| Random Forest (Ensemble) | Consistently strong across metrics. |
| XGBoost (Ensemble) | Best overall performance across all metrics. |
