from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from ml_utils import (
    DatasetSplit,
    MODEL_FILES,
    RANDOM_STATE,
    TARGET_COL,
    evaluate_models,
    evaluate_single_model,
    load_dataset,
    load_label_encoder,
    load_metrics,
    load_models,
    train_models,
)

ROOT_DIR = Path(__file__).parent
DATA_PATH = ROOT_DIR / "data.csv"
MODEL_DIR = ROOT_DIR / "model"


def models_ready() -> bool:
    required = [MODEL_DIR / filename for filename in MODEL_FILES.values()]
    required.append(MODEL_DIR / "label_encoder.joblib")
    return all(path.exists() for path in required)


@st.cache_resource
def prepare_artifacts():
    df = load_dataset(str(DATA_PATH))

    if models_ready():
        models = load_models(str(MODEL_DIR))
        label_encoder = load_label_encoder(str(MODEL_DIR))
        metrics_table = load_metrics(str(MODEL_DIR))

        if metrics_table is None:
            split = build_split(df, label_encoder)
            metrics_table = evaluate_models(models, split)
        else:
            split = build_split(df, label_encoder)
    else:
        models, split, metrics_table = train_models(df, str(MODEL_DIR))
        label_encoder = split.label_encoder

    return df, models, split, metrics_table, label_encoder


def build_split(df: pd.DataFrame, label_encoder) -> DatasetSplit:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
    )


st.set_page_config(page_title="Obesity Level Classification", layout="wide")

st.title("Obesity Level Classification Models")

with st.spinner("Loading models and data..."):
    df, models, split, metrics_table, label_encoder = prepare_artifacts()

st.markdown("**Dataset:** Estimation of obesity levels based on eating habits and physical condition")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
feature_columns = df.drop(columns=[TARGET_COL]).columns.tolist()

st.subheader("Model Comparison Metrics")
metrics_display = metrics_table.copy()
metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metrics_display[metric_cols] = metrics_display[metric_cols].round(4)
st.dataframe(metrics_display, use_container_width=True, hide_index=True)

st.subheader("Model Evaluation (Default Test Split)")
model_name = st.selectbox("Select a model", list(models.keys()))

selected_model = models[model_name]
metrics_df, cm_df, report = evaluate_single_model(
    selected_model,
    split.X_test,
    split.y_test,
    label_encoder,
)

st.markdown("**Metrics**")
st.dataframe(metrics_df.round(4), use_container_width=True, hide_index=True)

st.markdown("**Confusion Matrix**")
st.dataframe(cm_df, use_container_width=True)

st.markdown("**Classification Report**")
st.text(report)

st.subheader("Upload Test CSV")
uploaded_file = st.file_uploader(
    "Upload a CSV file with the same columns as the training data.",
    type=["csv"],
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    missing_cols = [col for col in feature_columns if col not in test_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
    elif TARGET_COL in test_df.columns:
        X_test = test_df[feature_columns]
        y_test_raw = test_df[TARGET_COL]
        y_test = label_encoder.transform(y_test_raw)

        upload_metrics_df, upload_cm_df, upload_report = evaluate_single_model(
            selected_model,
            X_test,
            y_test,
            label_encoder,
        )

        st.markdown("**Metrics (Uploaded Test Data)**")
        st.dataframe(upload_metrics_df.round(4), use_container_width=True, hide_index=True)

        st.markdown("**Confusion Matrix (Uploaded Test Data)**")
        st.dataframe(upload_cm_df, use_container_width=True)

        st.markdown("**Classification Report (Uploaded Test Data)**")
        st.text(upload_report)
    else:
        X_test = test_df[feature_columns]
        predictions = selected_model.predict(X_test)
        labels = label_encoder.inverse_transform(predictions)

        preview = X_test.copy()
        preview["Prediction"] = labels

        st.markdown("Target column not found. Showing predictions only.")
        st.dataframe(preview.head(10), use_container_width=True)
