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
| Logistic Regression | 0.8747 | 0.9834 | 0.8707 | 0.8724 | 0.8708 | 0.8539 |
| Decision Tree | 0.9149 | 0.9488 | 0.9157 | 0.9118 | 0.9133 | 0.9007 |
| KNN | 0.8251 | 0.9650 | 0.8235 | 0.8187 | 0.8131 | 0.7981 |
| Naive Bayes | 0.5083 | 0.8372 | 0.5207 | 0.5002 | 0.4511 | 0.4471 |
| Random Forest (Ensemble) | 0.9433 | 0.9946 | 0.9461 | 0.9420 | 0.9430 | 0.9341 |
| XGBoost (Ensemble) | 0.9598 | 0.9977 | 0.9600 | 0.9582 | 0.9587 | 0.9532 |

## Observations
| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strong baseline with high AUC, but lower than tree ensembles. |
| Decision Tree | Good accuracy, but slightly less stable than ensemble methods. |
| KNN | Moderate performance; sensitive to feature scaling. |
| Naive Bayes | Lowest scores; class-conditional independence is too simplistic here. |
| Random Forest (Ensemble) | Consistently strong across metrics. |
| XGBoost (Ensemble) | Best overall performance across all metrics. |
