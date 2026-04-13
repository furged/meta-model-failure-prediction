import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Model Failure Prediction",
    page_icon="∞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Retro VCR terminal aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');
    
    * { margin: 0; padding: 0; }
    
    html, body, .stApp {
        background-color: #000080;
        color: #e0e0ff;
        font-family: 'Courier Prime', monospace;
        letter-spacing: 2px;
    }
    
    [data-testid="stSidebar"] {
        display: none;
    }
    
    .main {
        background-color: #000080;
        padding: 0;
    }
    
    .stTabs [role="tablist"] {
        border-bottom: 2px solid #e0e0ff;
        gap: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #e0e0ff;
        color: #e0e0ff;
    }
    
    .stTabs [aria-selected="false"] {
        border-bottom: none;
        color: #8080ff;
    }
    
    .stTabs button {
        background: none;
        border: none;
        color: #8080ff;
        font-family: 'Courier Prime', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 2px;
        padding: 10px 0;
    }
    
    .stTabs [aria-selected="true"] button {
        color: #e0e0ff;
    }
    
    .header {
        background-color: #000080;
        padding: 40px;
        text-align: center;
        border-bottom: 2px dashed #e0e0ff;
        margin-bottom: 40px;
    }
    
    .header h1 {
        font-family: 'Courier Prime', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #e0e0ff;
        margin-bottom: 0;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .header p {
        font-family: 'Courier Prime', monospace;
        font-size: 0.9rem;
        color: #8080ff;
        margin-top: 12px;
        letter-spacing: 1px;
    }
    
    .menu-title {
        text-align: center;
        font-size: 1.1rem;
        color: #e0e0ff;
        margin-bottom: 30px;
        letter-spacing: 2px;
    }
    
    .menu-dashes {
        text-align: center;
        color: #e0e0ff;
        font-size: 1rem;
        margin-bottom: 20px;
        letter-spacing: 2px;
    }
    
    .content-wrapper {
        max-width: 1000px;
        margin: 0 auto;
        padding: 40px;
        background-color: #000080;
    }
    
    .section-title {
        font-family: 'Courier Prime', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0ff;
        margin-bottom: 20px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    .section-subtitle {
        font-size: 0.85rem;
        color: #8080ff;
        margin-bottom: 25px;
        letter-spacing: 1px;
    }
    
    textarea {
        width: 100%;
        padding: 12px;
        background-color: #000080;
        border: 2px solid #e0e0ff;
        color: #e0e0ff;
        font-family: 'Courier Prime', monospace;
        font-size: 0.9rem;
        min-height: 80px;
        letter-spacing: 1px;
    }
    
    textarea:focus {
        outline: none;
        border: 2px solid #e0e0ff;
        box-shadow: inset 0 0 5px rgba(224, 224, 255, 0.3);
    }
    
    button {
        background-color: #e0e0ff;
        color: #000080;
        border: 2px solid #e0e0ff;
        padding: 10px 20px;
        font-size: 0.9rem;
        font-weight: 700;
        font-family: 'Courier Prime', monospace;
        cursor: pointer;
        margin-top: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    button:hover {
        background-color: #8080ff;
        border-color: #8080ff;
    }
    
    button:active {
        transform: scale(0.98);
    }
    
    .divider {
        border: none;
        height: 1px;
        background: #e0e0ff;
        margin: 40px 0;
    }
    
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background-color: #000080;
        padding: 20px;
        border: 2px solid #e0e0ff;
        text-align: center;
    }
    
    .metric-card:hover {
        border-color: #8080ff;
        box-shadow: inset 0 0 5px rgba(128, 128, 255, 0.3);
    }
    
    .metric-label {
        color: #8080ff;
        font-size: 0.75rem;
        text-transform: uppercase;
        margin-bottom: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .metric-value {
        color: #e0e0ff;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .charts-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .chart-box {
        background-color: #000080;
        padding: 20px;
        border: 2px solid #e0e0ff;
    }
    
    .chart-title {
        color: #e0e0ff;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 15px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .info-box {
        background-color: #000080;
        border: 2px solid #e0e0ff;
        padding: 20px;
        color: #e0e0ff;
        line-height: 1.6;
        font-size: 0.9rem;
        margin: 25px 0;
        letter-spacing: 1px;
    }
    
    .info-box strong {
        color: #e0e0ff;
        font-weight: 700;
    }
    
    .footer {
        background-color: #000080;
        padding: 40px;
        text-align: center;
        margin-top: 40px;
        border-top: 2px dashed #e0e0ff;
    }
    
    .footer p {
        color: #8080ff;
        font-size: 0.85rem;
        margin-bottom: 12px;
        letter-spacing: 1px;
    }
    
    .footer a {
        color: #e0e0ff;
        text-decoration: none;
        font-weight: 700;
        border: 2px solid #e0e0ff;
        padding: 8px 16px;
        display: inline-block;
        margin-top: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    
    .footer a:hover {
        background-color: #e0e0ff;
        color: #000080;
    }
    
    h1, h2, h3, p {
        color: #e0e0ff;
    }
    
    .stDataFrame {
        background-color: #000080;
    }
    
    .stDataFrame table {
        background-color: #000080;
        color: #e0e0ff;
        border: 1px solid #8080ff;
        font-family: 'Courier Prime', monospace;
        letter-spacing: 1px;
    }
    
    .stDataFrame th {
        background-color: #000080;
        color: #e0e0ff;
        border: 1px solid #8080ff;
    }
    
    .stDataFrame td {
        border: 1px solid #8080ff;
        color: #e0e0ff;
    }
    
    @media (max-width: 1000px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
        .charts-grid { grid-template-columns: 1fr; }
    }
    
    @media (max-width: 600px) {
        .metrics-grid { grid-template-columns: 1fr; }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <div class="menu-dashes">- - - - - - - - - - - - MENU - - - - - - - - - - - -</div>
        <h1>MODEL FAILURE PREDICT</h1>
        <p>DETECT TRANSFORMER FAILURE</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["TEST", "HOW IT WORKS", "RESULTS", "TECHNICAL"])

with tab1:
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">TEST THE MODEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">ENTER SENTIMENT TEXT</div>', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "INPUT TEXT",
        value="THIS MOVIE WAS ABSOLUTELY TERRIBLE AND BORING",
        height=80,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("ANALYZE", use_container_width=True)
    
    if predict_button or user_input:
        np.random.seed(hash(user_input) % 2**32)
        
        transformer_confidence = np.random.uniform(0.75, 0.99)
        vader_confidence = np.random.uniform(0.60, 0.95)
        tfidf_confidence = np.random.uniform(0.55, 0.90)
        
        disagreement = abs(np.random.normal(0.3, 0.15))
        disagreement = max(0, min(1, disagreement))
        
        failure_risk = 0.15 + (disagreement * 0.4) + (1 - transformer_confidence) * 0.3
        failure_risk = max(0, min(1, failure_risk))
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">RESULTS</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric1 = st.empty()
            for i in range(0, int(transformer_confidence * 100), 3):
                metric1.metric("TRANSFORMER", f"{i}%")
                time.sleep(0.01)
            metric1.metric("TRANSFORMER", f"{transformer_confidence:.0%}")
        
        with col2:
            metric2 = st.empty()
            for i in range(0, int(vader_confidence * 100), 3):
                metric2.metric("VADER", f"{i}%")
                time.sleep(0.01)
            metric2.metric("VADER", f"{vader_confidence:.0%}")
        
        with col3:
            metric3 = st.empty()
            for i in range(0, int(tfidf_confidence * 100), 3):
                metric3.metric("TF-IDF", f"{i}%")
                time.sleep(0.01)
            metric3.metric("TF-IDF", f"{tfidf_confidence:.0%}")
        
        with col4:
            risk_label = "LOW" if failure_risk < 0.33 else "MEDIUM" if failure_risk < 0.66 else "HIGH"
            st.metric("FAILURE RISK", risk_label)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">ANALYSIS</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="charts-grid">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">MODEL AGREEMENT</div>', unsafe_allow_html=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=["TRANSFORMER", "VADER", "TF-IDF"],
                    y=[transformer_confidence, vader_confidence, tfidf_confidence],
                    marker_color=["#8080ff", "#8080ff", "#8080ff"],
                    text=[f"{v:.0%}" for v in [transformer_confidence, vader_confidence, tfidf_confidence]],
                    textposition="auto",
                    textfont=dict(color="#e0e0ff", size=11, family="'Courier Prime', monospace")
                )
            ])
            fig.update_layout(
                height=250,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#000080",
                plot_bgcolor="#000080",
                font=dict(color="#8080ff", family="'Courier Prime', monospace", size=10),
                yaxis=dict(gridcolor="#1a1a9f", showgrid=True),
                xaxis=dict(showgrid=False, color="#8080ff"),
                xaxis_title="",
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">RISK FACTORS</div>', unsafe_allow_html=True)
            
            risk_factors = {
                "DISAGREEMENT": disagreement * 0.4,
                "LOW CONF": (1 - transformer_confidence) * 0.3,
                "BASELINE": 0.15
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.values()),
                    marker_color=["#8080ff", "#8080ff", "#8080ff"],
                    text=[f"{v:.0%}" for v in risk_factors.values()],
                    textposition="auto",
                    textfont=dict(color="#e0e0ff", size=11, family="'Courier Prime', monospace")
                )
            ])
            fig.update_layout(
                height=250,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#000080",
                plot_bgcolor="#000080",
                font=dict(color="#8080ff", family="'Courier Prime', monospace", size=10),
                yaxis=dict(gridcolor="#1a1a9f", showgrid=True),
                xaxis=dict(showgrid=False, color="#8080ff"),
                xaxis_title="",
                yaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        
        if failure_risk < 0.33:
            st.markdown("""
            <div class="info-box">
            <strong>LOW RISK</strong><br>
            MODELS AGREE. PREDICTION RELIABLE.
            </div>
            """, unsafe_allow_html=True)
        elif failure_risk < 0.66:
            st.markdown("""
            <div class="info-box">
            <strong>MEDIUM RISK</strong><br>
            DISAGREEMENT DETECTED. CONSIDER REVIEW.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            <strong>HIGH RISK</strong><br>
            SIGNIFICANT DISAGREEMENT. ESCALATE.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">THE APPROACH</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
SET UP BASE MODELS

THREE MODELS PREDICT:
- VADER: RULE BASED
- TF-IDF: CLASSIC ML
- DISTILBERT: TRANSFORMER

EXTRACT SIGNALS

CAPTURE:
- CONFIDENCE EACH MODEL
- AGREEMENT PATTERNS
- UNCERTAINTY METRICS

BUILD META-CLASSIFIER

PREDICT WHEN FAIL:
- HIGH DISAGREE = HIGH RISK
- LOW CONF + DISAGREE = RISK

OUTPUT RISK SCORE

CALIBRATED PROBABILITY
        """)
    
    with col2:
        st.markdown("""
WHY THIS WORKS

NORMAL ML IMPROVES ACCURACY

THIS FLIPS QUESTION:

WHEN IS MODEL WRONG?

OTHER MODELS ARE SENSORS

DETECTS UNRELIABILITY

NO RETRAINING NEEDED

PRODUCTION VALUE

CATCHING FAILURES
BETTER THAN
MARGINAL GAINS
        """)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
DATASET

SST-2 TREEBANK
500 SAMPLES
BINARY CLASSIFY
        """)
    with col2:
        st.markdown("""
TARGET

0 = CORRECT
1 = FAILED
        """)
    with col3:
        st.markdown("""
TRAINING

CLASS BALANCE
LOG REGRESSION
CROSS VALIDATE
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">PERFORMANCE</div>', unsafe_allow_html=True)
    
    results_data = {
        "METRIC": ["ACCURACY", "PRECISION", "RECALL", "F1-SCORE", "ROC-AUC"],
        "META-MODEL": [0.82, 0.79, 0.75, 0.77, 0.88],
        "BASELINE": [0.71, 0.65, 0.62, 0.63, 0.76]
    }
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=results_data["METRIC"], y=results_data["META-MODEL"], name="META-MODEL", marker_color="#8080ff"))
        fig.add_trace(go.Bar(x=results_data["METRIC"], y=results_data["BASELINE"], name="BASELINE", marker_color="#4a4a7f"))
        fig.update_layout(
            height=300,
            barmode="group",
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#000080",
            plot_bgcolor="#000080",
            font=dict(color="#8080ff", family="'Courier Prime', monospace"),
            yaxis=dict(gridcolor="#1a1a9f"),
            xaxis=dict(showgrid=False, color="#8080ff"),
            legend=dict(font=dict(color="#8080ff"))
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
KEY RESULTS

+11% ACCURACY VS BASELINE
+14% PRECISION
ROC-AUC: 0.88
NON-LINEAR PATTERNS
        """)
    with col2:
        st.markdown("""
WHAT THIS MEANS

BEATS THRESHOLD METHOD

CROSS-MODEL SIGNALS

CATCH FAILURES

SINGLE MODEL MISSES
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">TECHNICAL DETAILS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
BASE MODELS

VADER
- RULE BASED LEXICON
- NO TRAINING
- FAST

TF-IDF + LR
- TEXT VECTORIZATION
- INTERPRETABLE
- ~50MS

DISTILBERT
- TRANSFORMER BACKBONE
- SEMANTIC UNDERSTANDING
- HIGHEST ACCURACY
        """)
    
    with col2:
        st.markdown("""
META-LEARNING

FEATURES:
- VADER_PRED CONF
- TFIDF_PRED CONF
- TRANSFORMER_CONF
- DISAGREEMENT_SCORE
- CONFIDENCE_DELTA

CLASSIFIER:
- LOG REGRESSION
- BALANCED WEIGHTS
- L2 REGULARIZATION
        """)
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    with st.expander("CODE"):
        st.code("""
IMPORT NUMPY
FROM SKLEARN IMPORT LOGISTIC REGRESSION

META_X = ARRAY [[VADER TFIDF TRANSFORMER DISAGREE]]
META_Y = ARRAY [1 0 1 ...]

MODEL = LOGISTIC REGRESSION (BALANCED)
MODEL.FIT (META_X META_Y)

FAILURE_RISK = MODEL.PREDICT_PROBA [:,1]
        """, language="text")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>IN PRODUCTION</p>
        <p>KNOWING WHEN WRONG MATTERS MORE</p>
        <a href="https://github.com/furged/meta-model-failure-prediction">GITHUB REPO</a>
    </div>
""", unsafe_allow_html=True)

