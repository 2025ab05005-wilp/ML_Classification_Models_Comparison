import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

TARGET_COL = "NObeyesdad"
RANDOM_STATE = 42
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


def load_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_dataset(df: pd.DataFrame, test_size: float = 0.2) -> DatasetSplit:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_models(num_classes: int) -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=1),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            n_estimators=250,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def build_pipelines(X_train: pd.DataFrame, num_classes: int) -> Dict[str, Pipeline]:
    models = build_models(num_classes)
    pipelines = {}
    for name, model in models.items():
        preprocessor = build_preprocessor(X_train)
        pipelines[name] = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
    return pipelines


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    num_classes: int,
) -> Dict[str, float]:
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    auc = float("nan")
    if y_prob is not None:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc = roc_auc_score(
                y_true_bin,
                y_prob,
                multi_class="ovr",
                average="macro",
            )

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
    }


def evaluate_models(
    models: Dict[str, Pipeline],
    split: DatasetSplit,
) -> pd.DataFrame:
    num_classes = len(split.label_encoder.classes_)
    rows = []

    for name, model in models.items():
        y_pred = model.predict(split.X_test)
        y_prob = model.predict_proba(split.X_test)
        metrics = compute_metrics(split.y_test, y_pred, y_prob, num_classes)
        rows.append({"Model": name, **metrics})

    return pd.DataFrame(rows)


def train_models(
    df: pd.DataFrame,
    model_dir: str,
    test_size: float = 0.2,
) -> Tuple[Dict[str, Pipeline], DatasetSplit, pd.DataFrame]:
    split = split_dataset(df, test_size=test_size)
    num_classes = len(split.label_encoder.classes_)

    models = build_pipelines(split.X_train, num_classes)

    for model in models.values():
        model.fit(split.X_train, split.y_train)

    metrics_table = evaluate_models(models, split)

    os.makedirs(model_dir, exist_ok=True)
    for name, model in models.items():
        filename = MODEL_FILES.get(name)
        if filename is None:
            continue
        joblib.dump(model, os.path.join(model_dir, filename))

    joblib.dump(split.label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
    metrics_table.to_csv(os.path.join(model_dir, "metrics.csv"), index=False)

    return models, split, metrics_table


def load_models(model_dir: str) -> Dict[str, Pipeline]:
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def load_metrics(model_dir: str) -> pd.DataFrame | None:
    path = os.path.join(model_dir, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_label_encoder(model_dir: str) -> LabelEncoder:
    return joblib.load(os.path.join(model_dir, "label_encoder.joblib"))


def evaluate_single_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    num_classes = len(label_encoder.classes_)
    metrics = compute_metrics(y_test, y_pred, y_prob, num_classes)
    metrics_df = pd.DataFrame([metrics])

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    return metrics_df, cm_df, report
