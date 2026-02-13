from pathlib import Path
import io

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split

from model.train_models import (
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
    """Build dataset split matching the training configuration"""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split off validation set (6%)
    X_temp, X_val, y_temp_raw, y_val_raw = train_test_split(
        X,
        y,
        test_size=0.06,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Then split remaining into train/test
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X_temp,
        y_temp_raw,
        test_size=0.2 / 0.94,  # Adjusted to maintain ~20% of total
        random_state=RANDOM_STATE,
        stratify=y_temp_raw,
    )

    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    y_val = label_encoder.transform(y_val_raw)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        X_val=X_val,
        y_train=y_train,
        y_test=y_test,
        y_val=y_val,
        label_encoder=label_encoder,
    )


st.set_page_config(page_title="Obesity Level Classification", layout="wide")

st.title("🎯 Obesity Level Classification Models")

with st.spinner("Loading models and data..."):
    df, models, split, metrics_table, label_encoder = prepare_artifacts()

# ============================================================================
# SIDEBAR - Controls, Filters, and Actions
# ============================================================================
st.sidebar.header("⚙️ Controls & Options")

# Dataset info
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
**Total Rows:** {df.shape[0]}  
**Features:** {df.shape[1] - 1}  
**Target Classes:** {len(label_encoder.classes_)}
""")

# Model selection
st.sidebar.subheader("🤖 Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a model",
    list(models.keys()),
    help="Select the ML model for detailed evaluation"
)

# Data source selection
st.sidebar.subheader("🔍 Data Source")
data_source = st.sidebar.radio(
    "Evaluate on:",
    ["Test Set", "Validation Set", "Upload Custom Data"],
    help="Choose which dataset to use for model evaluation"
)

# Download options
st.sidebar.subheader("⬇️ Download Data")
col1, col2 = st.sidebar.columns(2)

# Download test data
test_data = pd.DataFrame(split.X_test)
test_data[TARGET_COL] = label_encoder.inverse_transform(split.y_test)
test_csv = test_data.to_csv(index=False).encode('utf-8')
col1.download_button(
    label="📥 Test Set",
    data=test_csv,
    file_name="test_data.csv",
    mime="text/csv",
)

# Download validation data
val_data = pd.DataFrame(split.X_val)
val_data[TARGET_COL] = label_encoder.inverse_transform(split.y_val)
val_csv = val_data.to_csv(index=False).encode('utf-8')
col2.download_button(
    label="📥 Val Set",
    data=val_csv,
    file_name="validation_data.csv",
    mime="text/csv",
)

# Upload option
uploaded_file = None
if data_source == "Upload Custom Data":
    st.sidebar.subheader("⬆️ Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Upload a CSV with the same features as training data"
    )

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • ML Classification Dashboard")

# ============================================================================
# MAIN AREA - Metrics, Visualizations, and Results
# ============================================================================

# Baseline metrics display
st.header("📈 Baseline Model Performance")
st.markdown("**Comparison of all trained models on the test set**")

metrics_display = metrics_table.copy()
metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
metrics_display[metric_cols] = metrics_display[metric_cols].round(4)

# Highlight best model for each metric
def highlight_max(s):
    if s.name in metric_cols:
        is_max = s == s.max()
        return ['background-color: lightgreen' if v else '' for v in is_max]
    return ['' for _ in s]

styled_metrics = metrics_display.style.apply(highlight_max, axis=0)
st.dataframe(styled_metrics, width='stretch', hide_index=True)

# Visualization of baseline metrics
st.subheader("📊 Baseline Metrics Comparison")
col1, col2 = st.columns(2)

with col1:
    # Accuracy comparison
    fig_acc = px.bar(
        metrics_display,
        x='Model',
        y='Accuracy',
        title='Model Accuracy Comparison',
        text='Accuracy',
        color='Accuracy',
        color_continuous_scale='Blues'
    )
    fig_acc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_acc.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_acc, width='stretch')

with col2:
    # F1 Score comparison
    fig_f1 = px.bar(
        metrics_display,
        x='Model',
        y='F1',
        title='Model F1 Score Comparison',
        text='F1',
        color='F1',
        color_continuous_scale='Greens'
    )
    fig_f1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_f1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_f1, width='stretch')

st.markdown("---")

# ============================================================================
# MODEL-SPECIFIC EVALUATION
# ============================================================================

st.header(f"🔬 Detailed Evaluation: {model_name}")

# Determine which data to evaluate on
selected_model = models[model_name]
X_eval, y_eval = split.X_test, split.y_test
eval_label = "Test Set"

if data_source == "Validation Set":
    X_eval, y_eval = split.X_val, split.y_val
    eval_label = "Validation Set"
elif data_source == "Upload Custom Data" and uploaded_file is not None:
    eval_label = "Uploaded Data"

if data_source == "Upload Custom Data" and uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    feature_columns = df.drop(columns=[TARGET_COL]).columns.tolist()
    
    missing_cols = [col for col in feature_columns if col not in test_df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
    elif TARGET_COL in test_df.columns:
        X_eval = test_df[feature_columns]
        y_eval = label_encoder.transform(test_df[TARGET_COL])
    else:
        st.warning("⚠️ Target column not found in uploaded data. Predictions only mode.")
        X_eval = test_df[feature_columns]
        y_eval = None

# Perform evaluation
if y_eval is not None:
    metrics_df, cm_df, report = evaluate_single_model(
        selected_model,
        X_eval,
        y_eval,
        label_encoder,
    )

    # Display metrics
    st.subheader(f"📊 Performance Metrics ({eval_label})")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics_dict = metrics_df.iloc[0].to_dict()
    col1.metric("Accuracy", f"{metrics_dict['Accuracy']:.4f}")
    col2.metric("AUC", f"{metrics_dict['AUC']:.4f}")
    col3.metric("Precision", f"{metrics_dict['Precision']:.4f}")
    col4.metric("Recall", f"{metrics_dict['Recall']:.4f}")
    col5.metric("F1 Score", f"{metrics_dict['F1']:.4f}")
    col6.metric("MCC", f"{metrics_dict['MCC']:.4f}")

    st.markdown("---")

    # Confusion Matrix and Correlation side by side
    st.subheader("📉 Confusion Matrix & Feature Correlations")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion Matrix**")
        # Create interactive heatmap
        fig_cm = ff.create_annotated_heatmap(
            z=cm_df.values,
            x=cm_df.columns.tolist(),
            y=cm_df.index.tolist(),
            colorscale='Blues',
            showscale=True
        )
        fig_cm.update_layout(
            title=dict(
                text="Confusion Matrix Heatmap",
                y=0.98,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=14)
            ),
            height=500,
            margin=dict(t=60, b=80, l=80, r=20)
        )
        fig_cm.update_xaxes(
            title_text="Predicted",
            title_font=dict(size=12),
            side='bottom'
        )
        fig_cm.update_yaxes(
            title_text="Actual",
            title_font=dict(size=12)
        )
        st.plotly_chart(fig_cm, width='stretch')
        
        # Also show as dataframe
        with st.expander("View Confusion Matrix Table"):
            st.dataframe(cm_df, width='stretch')

    with col2:
        st.markdown("**Feature Correlation Matrix**")
        # Calculate correlation matrix for numeric features
        numeric_features = df.select_dtypes(include=['number']).columns.tolist()
        if TARGET_COL in numeric_features:
            numeric_features.remove(TARGET_COL)
        
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr()
            
            # Use plotly express for better control over annotations
            num_features = len(numeric_features)
            
            # Create heatmap with or without annotations based on size
            if num_features <= 8:
                # Show annotations for smaller matrices
                fig_corr = ff.create_annotated_heatmap(
                    z=corr_matrix.values.round(2),
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale='RdBu',
                    showscale=True,
                    zmid=0,
                    annotation_text=corr_matrix.values.round(2),
                )
                # Update annotation font size
                for annotation in fig_corr['layout']['annotations']:
                    annotation['font'] = dict(size=9)
            else:
                # Use heatmap without annotations for larger matrices
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,
                    aspect="auto",
                    text_auto=False  # No annotations for readability
                )
            
            fig_corr.update_layout(
                title="Feature Correlation Heatmap",
                height=500,
                xaxis={'side': 'bottom'},
                font=dict(size=10)
            )
            fig_corr.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_corr, width='stretch')
            
            with st.expander("View Correlation Table"):
                st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), 
                           width='stretch')
        else:
            st.info("Not enough numeric features for correlation analysis")

    st.markdown("---")

    # Classification Report
    st.subheader("📋 Detailed Classification Report")
    with st.expander("View Full Report", expanded=False):
        st.text(report)
        
else:
    # Prediction only mode (no target in uploaded data)
    if data_source == "Upload Custom Data":
        predictions = selected_model.predict(X_eval)
        pred_proba = selected_model.predict_proba(X_eval)
        labels = label_encoder.inverse_transform(predictions)

        preview = X_eval.copy()
        preview["Predicted_Class"] = labels
        
        # Add probability columns
        for idx, class_name in enumerate(label_encoder.classes_):
            preview[f"Prob_{class_name}"] = pred_proba[:, idx]

        st.subheader("🔮 Predictions")
        st.markdown(f"**Showing predictions for {len(preview)} samples**")
        st.dataframe(preview, width='stretch', hide_index=True)
        
        # Download predictions
        pred_csv = preview.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions",
            data=pred_csv,
            file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
