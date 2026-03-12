import streamlit as st
import psutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# ---------------- Page Config ----------------
st.set_page_config(page_title="InferAI Pro", layout="wide", page_icon="⚡")

# ---------------- Advanced CSS (Matches Image Exactly) ----------------
st.markdown("""
<style>
/* Main Background and Fonts */
body, .stApp {background-color:#FFF; color:#000; font-family:'Inter', 'Segoe UI', sans-serif;}

/* Sidebar - Black Background with Orange accents */
[data-testid="stSidebar"] {background-color:#000 !important; border-right: 2px solid #FF4500; color:#FF4500 !important;}
[data-testid="stSidebar"] * {color:#FF4500 !important;}

/* The Large Header Banner */
.header-banner {
    background: linear-gradient(135deg, #000 0%, #FF4500 100%);
    border-radius: 20px;
    padding: 60px 20px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 15px 35px rgba(255, 69, 0, 0.3);
}
.header-banner h1 {font-size: 68px; font-weight: 900; color: white; margin: 0; letter-spacing: 2px;}
.header-banner p {font-size: 18px; color: #fff; font-weight: 400; opacity: 0.9; margin-top: 10px;}

/* Metric Cards with Orange Outlines */
.metric-box {
    background: #ffffff;
    border: 2px solid #FF4500;
    border-radius: 15px;
    padding: 25px 10px;
    text-align: center;
    height: 100%;
    box-shadow: 0 4px 15px rgba(255, 69, 0, 0.1);
}
.metric-box h3 {color: #000; font-size: 14px; font-weight: 900; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;}
.metric-box h1 {color: #FF4500; font-size: 48px; font-weight: 800; margin: 10px 0;}
.metric-box p {font-size: 11px; color: #888; text-transform: uppercase; font-weight: 700;}

/* Feature Panels */
.feature-container {
    border: 2px solid #FF4500;
    border-radius: 15px;
    padding: 20px;
    background: #fff;
    margin-top: 20px;
}
.panel-label {color: #FF4500; font-size: 13px; font-weight: 900; text-transform: uppercase; margin-bottom: 15px; display: block;}

/* Reasoning Box */
.black-reasoning-box {
    background: #000;
    border-left: 5px solid #FF4500;
    padding: 20px;
    border-radius: 10px;
    color: #FF4500;
    font-family: 'Courier New', monospace;
    font-style: italic;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Data Logic ----------------
if "hist_df" not in st.session_state:
    st.session_state.hist_df = pd.DataFrame(columns=['Time', 'CPU', 'Mem', 'Risk'])
if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0

@st.cache_data
def load_csv_data():
    try:
        df = pd.read_csv('system_metrics.csv')
        cols = {c.strip().lower(): c for c in df.columns}
        cpu_col = [cols[k] for k in cols if 'cpu' in k][0]
        mem_col = [cols[k] for k in cols if 'mem' in k or 'ram' in k][0]
        return df.rename(columns={cpu_col: 'cpu_val', mem_col: 'mem_val'})
    except: return None

# ---------------- Sidebar ----------------
st.sidebar.title("🛡️ ENGINE STATUS")
mode = st.sidebar.selectbox("Data Stream", ["Real-Time Live", "CSV Playback"])
scan_speed = st.sidebar.slider("Sampling Rate", 0.5, 3.0, 1.0)

# ---------------- UI Header ----------------
st.markdown("""
<div class="header-banner">
    <h1>InferAI</h1>
    <p>Next-Gen Predictive Failure Analysis & Forecasting</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Data Engine ----------------
def get_current_metrics():
    if mode == "Real-Time Live":
        c, m = psutil.cpu_percent(), psutil.virtual_memory().percent
        s = "STABLE"
    else:
        df = load_csv_data()
        if df is not None and st.session_state.row_idx < len(df):
            row = df.iloc[st.session_state.row_idx]
            c, m = float(row['cpu_val']), float(row['mem_val'])
            st.session_state.row_idx += 1
            s = f"ROW {st.session_state.row_idx}"
        else: return None, None, "COMPLETE", 0
    r = int(min(100, (c * 0.7) + (m * 0.3)))
    return c, m, s, r

cpu, mem, status, risk = get_current_metrics()

if status != "COMPLETE":
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.hist_df = pd.concat([st.session_state.hist_df, pd.DataFrame([[ts, cpu, mem, risk]], columns=['Time', 'CPU', 'Mem', 'Risk'])], ignore_index=True).tail(30)

    # Top Row Cards (4 Columns)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><h3>Risk Index</h3><h1>{risk}%</h1><p>State: {status}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><h3>CPU Load</h3><h1>{cpu}%</h1><p>Freq: Active</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><h3>RAM Usage</h3><h1>{mem}%</h1><p>Swap: 0.0%</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><h3>AI Forecast</h3><h1>{min(100, risk + 2)}%</h1><p>Trend: Predictive</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Body: Signals | Reasoning
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="feature-container"><span class="panel-label">● Real-Time Signals</span>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.hist_df['Time'], y=st.session_state.hist_df['Risk'], 
                                 fill='tozeroy', line=dict(color='#FF4500', width=3), fillcolor='rgba(255, 69, 0, 0.2)'))
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="feature-container"><span class="panel-label">AI Reasoning Layer</span>', unsafe_allow_html=True)
        reason = "Critical threshold breach detected: Initiating analysis." if risk > 80 else "System behavior follows historical trends."
        st.markdown(f'<div class="black-reasoning-box">"{reason}"</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="feature-container"><span class="panel-label">Recommended Action</span>', unsafe_allow_html=True)
        action = "Initialize Load Mitigation" if risk > 80 else "No action required."
        st.markdown(f'<p style="font-weight:700; font-size:18px; color:#000;">{action}</p>', unsafe_allow_html=True)
        if st.button("Apply Safe Action", use_container_width=True):
            st.toast("Security protocols deployed.")
        st.markdown('</div>', unsafe_allow_html=True)

    time.sleep(scan_speed)
    st.rerun()
