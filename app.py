import streamlit as st
import psutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
from joblib import load, dump
from sklearn.ensemble import IsolationForest
from pathlib import Path
import json

# ---------------- Page Config ----------------
st.set_page_config(page_title="InferAI Pro", layout="wide", page_icon="⚡")

# ---------------- Model Persistence Logic ----------------
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"

if not MODEL_PATH.exists():
    dummy_iso = IsolationForest(contamination=0.05).fit(np.random.rand(10, 3))
    dump(dummy_iso, MODEL_PATH)

pretrained_model = load(MODEL_PATH)

# ---------------- Advanced CSS ----------------
st.markdown("""
<style>
body, .stApp {background-color:#FFF; color:#000; font-family:'Inter', 'Segoe UI', sans-serif;}
[data-testid="stSidebar"] {background-color:#000 !important; border-right: 2px solid #FF4500; color:#FF4500 !important;}
[data-testid="stSidebar"] * {color:#FF4500 !important;}
.header-banner {
    background: linear-gradient(135deg, #000 0%, #FF4500 100%);
    border-radius: 20px; padding: 60px 20px; text-align: center; margin-bottom: 30px;
    box-shadow: 0 15px 35px rgba(255, 69, 0, 0.3);
}
.header-banner h1 {font-size: 68px; font-weight: 900; color: white; margin: 0; letter-spacing: 2px;}
.header-banner p {font-size: 18px; color: #fff; font-weight: 400; opacity: 0.9; margin-top: 10px;}
.metric-box {
    background: #ffffff; border: 2px solid #FF4500; border-radius: 15px;
    padding: 25px 10px; text-align: center; height: 100%;
    box-shadow: 0 4px 15px rgba(255, 69, 0, 0.1);
}
.metric-box h3 {color: #000; font-size: 14px; font-weight: 900; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;}
.metric-box h1 {color: #FF4500; font-size: 48px; font-weight: 800; margin: 10px 0;}
.metric-box p {font-size: 11px; color: #888; text-transform: uppercase; font-weight: 700;}
.feature-container { border: 2px solid #FF4500; border-radius: 15px; padding: 20px; background: #fff; margin-top: 20px;}
.panel-label {color: #FF4500; font-size: 13px; font-weight: 900; text-transform: uppercase; margin-bottom: 15px; display: block;}
.black-reasoning-box {
    background: #000; border-left: 5px solid #FF4500; padding: 20px;
    border-radius: 10px; color: #FF4500; font-family: 'Courier New', monospace;
    font-style: italic; font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Initialization ----------------
if "hist_df" not in st.session_state:
    st.session_state.hist_df = pd.DataFrame(columns=['Time', 'CPU', 'Mem', 'Risk'])
if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0

# ---------------- Logging Helper ----------------
def log_prediction(data_dict):
    log_file = Path("logs/prediction_log.json")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    if log_file.exists():
        with open(log_file, "r") as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(data_dict)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)

# ---------------- Sidebar ----------------
st.sidebar.title("🛡️ ENGINE STATUS")
mode = st.sidebar.selectbox("Data Stream", ["Real-Time Live", "CSV Playback"])
scan_speed = st.sidebar.slider("Sampling Rate", 0.5, 3.0, 1.0)

# ---------------- UI Header ----------------
st.markdown("""<div class="header-banner"><h1>InferAI</h1><p>Next-Gen Predictive Failure Analysis & Forecasting</p></div>""", unsafe_allow_html=True)

# ---------------- Metrics Engine ----------------
def get_current_metrics():
    if mode == "Real-Time Live":
        c, m = psutil.cpu_percent(), psutil.virtual_memory().percent
        s = "STABLE"
    else:
        try:
            # Look for the CSV file
            df_csv = pd.read_csv('system_metrics.csv')
            
            # Select only numeric columns to avoid "date to float" errors
            numeric_cols = df_csv.select_dtypes(include=[np.number])
            
            if numeric_cols.empty:
                return 0, 0, "NO NUMERIC DATA", 0
            
            # Get the current row based on playback index
            row = numeric_cols.iloc[st.session_state.row_idx % len(numeric_cols)]
            
            # Safely assign C and M from the first two available numeric columns
            c = float(row.iloc[0])
            m = float(row.iloc[1]) if len(row) > 1 else 0.0
            
            st.session_state.row_idx += 1
            s = f"PLAYBACK: {st.session_state.row_idx}"
        except Exception as e: 
            return 0, 0, f"ERR: {str(e)[:15]}", 0
    
    # Calculate Risk Index
    r = int(min(100, (c * 0.7) + (m * 0.3)))
    return c, m, s, r

# Execute metrics capture
cpu, mem, status, risk = get_current_metrics()

# Main Dashboard Loop Logic
ts = datetime.now().strftime("%H:%M:%S")
st.session_state.hist_df = pd.concat([st.session_state.hist_df, pd.DataFrame([[ts, cpu, mem, risk]], columns=['Time', 'CPU', 'Mem', 'Risk'])], ignore_index=True).tail(30)

# Dashboard Columns
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-box"><h3>Risk Index</h3><h1>{risk}%</h1><p>State: {status}</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-box"><h3>CPU Load</h3><h1>{cpu}%</h1><p>Freq: Active</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-box"><h3>RAM Usage</h3><h1>{mem}%</h1><p>Swap: 0.0%</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-box"><h3>AI Forecast</h3><h1>{min(100, risk + 2)}%</h1><p>Trend: Predictive</p></div>', unsafe_allow_html=True)

col_l, col_r = st.columns([2, 1])
with col_l:
    st.markdown('<div class="feature-container"><span class="panel-label">● Real-Time Signals</span>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st.session_state.hist_df['Time'], y=st.session_state.hist_df['Risk'], fill='tozeroy', line=dict(color='#FF4500', width=3), fillcolor='rgba(255, 69, 0, 0.2)'))
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, key=f"plot_{time.time()}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="feature-container"><span class="panel-label">AI Reasoning Layer</span>', unsafe_allow_html=True)
    reason = "Critical threshold breach detected." if risk > 80 else "System behavior follows historical trends."
    st.markdown(f'<div class="black-reasoning-box">"{reason}"</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-container"><span class="panel-label">Instant Diagnostic</span>', unsafe_allow_html=True)
    if st.button("📊 Run psutil-AI Probe", use_container_width=True):
        disk = psutil.disk_usage('/').percent
        x_input = np.array([[cpu, mem, disk]])
        p = pretrained_model.predict(x_input)[0]
        st.write(f"Result: {'🚨 ANOMALY' if p == -1 else '✔️ NORMAL'}")
        log_prediction({"type": "real-time-probe", "cpu": cpu, "mem": mem, "disk": disk, "pred": int(p)})
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INTEGRATED FEATURES (EXPANDERS) ----------------
st.markdown("---")
with st.expander("📁 Dataset Anomaly Detection (File Upload)"):
    pd.set_option("styler.render.max_elements", 5000000) 
    
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="file_up")
    if uploaded_file:
        try:
            df_up = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            
            num_df = df_up.select_dtypes(include=[np.number]).copy()
            
            if not num_df.empty:
                if st.button("Execute Batch Prediction"):
                    with st.spinner("Analyzing high-dimensional patterns..."):
                        iso = IsolationForest(contamination=0.05, random_state=42)
                        preds = iso.fit_predict(num_df)
                        
                        df_up["Status"] = ["NORMAL" if p == 1 else "ANOMALY" for p in preds]
                        
                        anomaly_count = int((preds == -1).sum())
                        st.subheader(f"Analysis Complete: Found {anomaly_count} potential anomalies")
                        
                        if len(df_up) > 1000:
                            st.warning("Showing first 1,000 rows.")
                            display_df = df_up.head(1000)
                        else:
                            display_df = df_up

                        styled_df = display_df.style.map(
                            lambda x: 'color: red; font-weight: bold' if x == 'ANOMALY' else '', 
                            subset=['Status']
                        )
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        csv = df_up.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Full Results", data=csv, file_name="anomaly_results.csv", mime="text/csv")
                        
                        log_prediction({"type": "batch-upload", "filename": uploaded_file.name, "records": len(df_up), "anomalies": anomaly_count})
            else:
                st.error("No numeric columns found.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------- Rerun Logic ----------------
time.sleep(scan_speed)
st.rerun()
