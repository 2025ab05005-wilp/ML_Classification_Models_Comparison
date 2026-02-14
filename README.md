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
| Logistic Regression | Achieves 0.8676 accuracy, ~7.1% higher than KNN (0.8322) and significantly outperforms Naive Bayes (0.5390). However, it underperforms ensemble methods—XGBoost is 10.6% higher and Random Forest is 6.4% higher. Despite limitations, strong baseline with AUC of 0.9814 indicates excellent class separation capability. |
| Decision Tree | Delivers 0.9362 accuracy, competitive with Random Forest (0.9314, -0.5%) but more stable than single decision trees typically are. Slightly lower AUC (0.9623) compared to Random Forest (0.9942) and XGBoost (0.9991). Generally reliable for this multi-classification problem despite potential overfitting concerns. |
| KNN | Achieves only 0.8322 accuracy, significantly below tree-based and ensemble models. Outperforms only Naive Bayes by 3.1%. Despite reasonable AUC of 0.9617, the low accuracy suggests KNN struggles with local density variations in this high-dimensional obesity feature space. Feature scaling sensitivity limits effectiveness. |
| Naive Bayes | Severely underperforms with 0.5390 accuracy, lagging all other models—33.5% behind XGBoost and 39.7% behind Decision Tree. While AUC of 0.8469 is respectable, the low accuracy indicates strong assumption violations (conditional independence). Not suitable for this multi-class obesity classification despite computational simplicity. |
| Random Forest (Ensemble) | Achieves competitive 0.9314 accuracy, within 0.5% of Decision Tree and 1.4% below XGBoost. Strong AUC of 0.9942 demonstrates excellent class separation. Balanced performance across precision, recall, and F1 makes it a robust alternative when interpretability is required despite slightly lower accuracy than XGBoost. |
| XGBoost (Ensemble) | Outstanding performance with 0.9740 accuracy—6.4% above Random Forest and 7.8% above Decision Tree. Highest AUC of 0.9991 and best-balanced metrics across all evaluation criteria. Superior gradient boosting optimization outweighs the black-box nature, making it the recommended model for production use. |

## Folder Structure
```
ML_Classification_Models_Comparison/
├── app.py                          # Streamlit application for visualization
├── data.csv                        # Dataset with obesity measurements and labels
├── requirements.txt                # Python package dependencies
├── README.md                       # Project documentation (this file)
├── .gitignore                      # Git ignore rules
└── model/
    ├── train_models.py             # Training script with all ML utility functions
    ├── logistic_regression.joblib  # Trained Logistic Regression model
    ├── decision_tree.joblib        # Trained Decision Tree model
    ├── knn.joblib                  # Trained KNN model
    ├── naive_bayes.joblib          # Trained Naive Bayes model
    ├── random_forest.joblib        # Trained Random Forest model
    ├── xgboost.joblib              # Trained XGBoost model
    ├── label_encoder.joblib        # Label encoder for target variable
    ├── metrics.csv                 # Model performance metrics
    └── validation_data.csv         # Validation dataset
```

## How to run the Model training python script

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup
1. Clone or download the project to your local machine
2. Navigate to the project directory:
   ```bash
   cd ML_Classification_Models_Comparison
   ```

3. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the training script
Execute the training script to train all models:
```bash
python model/train_models.py
```

This script will:
- Load the dataset from `data.csv`
- Split the data into train (74%), test (20%), and validation (6%) sets
- Train all 6 classification models
- Evaluate models on the test set
- Save trained models to the `model/` directory
- Generate `metrics.csv` with performance metrics
- Save the label encoder and validation data

The output will display a table with metrics for all models.

## How to run the Streamlit app locally

### Prerequisites
- Same as above (Python 3.8+ and dependencies installed)
- Trained models in the `model/` directory (run `train_models.py` first if models don't exist)

### Running the Streamlit app
1. Ensure you're in the project directory and the virtual environment is activated
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. The app will start and typically open in your default browser at `http://localhost:8501`

### Features of the Streamlit app
- **Model Overview**: View model performance metrics in an interactive table
- **Model Selection**: Select individual models to examine detailed metrics
- **Confusion Matrix**: Visualize model predictions vs actual values
- **Classification Report**: View precision, recall, F1-score per obesity class
- **Feature Analysis**: Explore feature distributions and their relationships with obesity levels

### Navigating the app
- Use the sidebar to select different models
- Interact with visualizations (zoom, pan, etc.)
- Hover over charts for detailed information
