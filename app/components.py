import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px

def metric_cards(mae: float, rmse: float, r2: float):
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("R²", f"{r2:.4f}")

def sawtooth_chart(df: pd.DataFrame, scenario_id, prediction_col: str = None):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time_hours"],
        y=df["target_Rf"],
        mode="lines",
        name="Actual Rf",
        line=dict(width=1.5),
    ))

    if prediction_col and prediction_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time_hours"],
            y=df[prediction_col],
            mode="lines",
            name="Predicted Rf",
            line=dict(width=1.5, dash="dot"),
        ))

    cip_times = df.loc[df["is_cleaning"] == 1, "time_hours"].values
    for t in cip_times:
        fig.add_vline(
            x=t,
            line_dash="dash",
            annotation_text="CIP" if t == cip_times[0] else "",
            annotation_position="top",
        )

    fig.update_layout(
        title=f"Fouling Resistance — Scenario {scenario_id}",
        xaxis_title="Operational Time (hours)",
        yaxis_title="Fouling Resistance Rf (standardised)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

def feature_importance_chart(model, feature_names: list):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    features_sorted = [feature_names[i] for i in sorted_idx]
    values_sorted = importances[sorted_idx]

    fig = go.Figure(go.Bar(
        x=values_sorted,
        y=features_sorted,
        orientation="h",
    ))
    fig.update_layout(
        title="XGBoost Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=320,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

def prediction_gauge(predicted_rf: float, threshold: float = 1.5):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_rf,
        delta={"reference": threshold, "valueformat": ".4f"},
        gauge={
            "axis": {"range": [-1, 3]},
            "steps": [
                {"range": [-1, 1.0]},
                {"range": [1.0, 1.5]},
                {"range": [1.5, 3.0]},
            ],
            "threshold": {
                "line": {"width": 2},
                "thickness": 0.75,
                "value": threshold,
            },
        },
        title={"text": "Predicted Rf (standardised)"},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def scatter_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, sample_n: int = 5000):
    idx = np.random.choice(len(y_true), size=min(sample_n, len(y_true)), replace=False)
    fig = px.scatter(
        x=y_true[idx], y=y_pred[idx],
        labels={"x": "Actual Rf", "y": "Predicted Rf"},
        opacity=0.4,
        title="Predicted vs Actual Rf (test scenarios)",
    )
    lim = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
    fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                  line=dict(dash="dash", width=1))
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)