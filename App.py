import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# ---------------- Page Config ----------------
st.set_page_config(page_title="InferAI Pro", layout="wide", page_icon="⚡")

# ---------------- Professional CSS ----------------
st.markdown("""
<style>
/* Main Background and Typography */
body, .stApp {background-color:#f8f9fa; color:#212529; font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}

/* Modernized Header */
.title-container {
    text-align:center; padding:30px 0; margin-bottom:30px;
    background: linear-gradient(135deg, #1a1a1a 0%, #FF4500 100%);
    border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.title-container .main-title {font-size:42px; font-weight:800; color:#FFF; margin:0; letter-spacing:1px;}
.title-container .subtitle {font-size:16px; color:#e0e0e0; font-weight:400; opacity:0.9;}

/* Sidebar Overhaul */
[data-testid="stSidebar"] {background-color:#000 !important; border-right: 2px solid #FF4500;}
[data-testid="stSidebar"] * {color:#FF4500 !important;}

/* Metric Cards: Added Flexbox and fixed spacing to prevent overlap */
.risk-card {
    background: #ffffff;
    border-top: 4px solid #FF4500;
    padding: 25px 15px;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    height: 100%; /* Ensures all cards in a row have same height */
}
.risk-card h3 {color: #6c757d; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;}
.risk-card h1 {color: #1a1a1a; font-size: 36px; font-weight: 800; margin: 0;}
.risk-card p {font-size: 12px; color: #FF4500; font-weight: 600; margin-top: 5px;}

/* Failure Log: Cleaner Monospace */
.failure-log {
    font-family: 'Consolas', 'Monaco', monospace; font-size: 12px; background: #1a1a1a;
    color: #00ff41; padding: 15px; border-radius: 8px; height: 320px; 
    overflow-y: auto; border: 1px solid #333; line-height: 1.6;
}

/* Section Headers */
.section-title {
    font-size: 18px; font-weight: 700; color: #1a1a1a; margin-bottom: 20px;
    border-left: 5px solid #FF4500; padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Dataset (Robust Mapper) ----------------
@st.cache_data
def load_and_map_data():
    try:
        df = pd.read_csv('system_metrics.csv')
        clean_cols = {c: c.strip().lower() for c in df.columns}
        cpu_col, mem_col = None, None
        for orig, clean in clean_cols.items():
            if 'cpu' in clean: cpu_col = orig
            if 'mem' in clean or 'ram' in clean: mem_col = orig
        
        if not cpu_col or not mem_col:
            st.error("Missing CPU/RAM columns.")
            st.stop()
        return df.rename(columns={cpu_col: 'cpu_val', mem_col: 'mem_val'})
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

data = load_and_map_data()

# ---------------- State Management ----------------
if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0
    st.session_state.log_data = []
    st.session_state.hist_df = pd.DataFrame(columns=['Time', 'CPU', 'Mem', 'Risk'])

# ---------------- UI Layout ----------------
st.markdown("""
<div class="title-container">
    <div class="main-title">INFERAI PRO</div>
    <div class="subtitle">Autonomous Predictive Monitoring & System Analytics</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Control
st.sidebar.title("🛡️ CONTROL CENTER")
speed = st.sidebar.slider("Sampling Rate (Sec)", 0.1, 2.0, 0.5)
if st.sidebar.button("RESTART ENGINE"):
    st.session_state.row_idx = 0
    st.session_state.log_data = []
    st.session_state.hist_df = pd.DataFrame(columns=['Time', 'CPU', 'Mem', 'Risk'])
    st.rerun()

# ---------------- Logic & Rendering ----------------
main_placeholder = st.empty()

if st.session_state.row_idx < len(data):
    row = data.iloc[st.session_state.row_idx]
    c_cpu = float(row['cpu_val'])
    c_mem = float(row['mem_val'])
    c_time = datetime.now().strftime("%H:%M:%S")
    risk_score = int(min(100, (c_cpu * 0.7) + (c_mem * 0.3)))

    # Update Data Structures
    new_entry = pd.DataFrame([[c_time, c_cpu, c_mem, risk_score]], columns=['Time', 'CPU', 'Mem', 'Risk'])
    st.session_state.hist_df = pd.concat([st.session_state.hist_df, new_entry], ignore_index=True).tail(30)
    st.session_state.log_data.append(f"[{c_time}] INF-SCAN: CPU {c_cpu}% | RISK {risk_score}% | STATUS: OK")
    st.session_state.row_idx += 1

    with main_placeholder.container():
        # --- TOP ROW: METRIC CARDS ---
        # Fixed spacing prevents boxes from overlapping
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="risk-card"><h3>Risk Index</h3><h1>{risk_score}%</h1><p>{"CRITICAL" if risk_score > 75 else "OPTIMAL"}</p></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="risk-card"><h3>CPU Load</h3><h1>{c_cpu}%</h1><p>ACTIVE</p></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="risk-card"><h3>RAM Usage</h3><h1>{c_mem}%</h1><p>MONITORED</p></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="risk-card"><h3>AI Forecast</h3><h1>{min(100, risk_score + 1)}%</h1><p>TRENDING</p></div>', unsafe_allow_html=True)

        st.write("---")

        # --- MIDDLE ROW: ANALYTICS & LOGS ---
        col_graph, col_log = st.columns([2, 1])
        
        with col_graph:
            st.markdown('<div class="section-title">Performance Correlation Timeline</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state.hist_df['Time'], y=st.session_state.hist_df['CPU'], 
                                     name="CPU (%)", line=dict(color='#FF4500', width=3), fill='tozeroy'))
            fig.add_trace(go.Scatter(x=st.session_state.hist_df['Time'], y=st.session_state.hist_df['Risk'], 
                                     name="Risk (%)", line=dict(color='#1a1a1a', width=2, dash='dot')))
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        with col_log:
            st.markdown('<div class="section-title">System Event Log</div>', unsafe_allow_html=True)
            log_content = "<br>".join(reversed(st.session_state.log_data[-50:]))
            st.markdown(f'<div class="failure-log">{log_content}</div>', unsafe_allow_html=True)

    time.sleep(speed)
    st.rerun()
else:
    st.info("Dataset analysis complete. Reset from control center to re-run.")
