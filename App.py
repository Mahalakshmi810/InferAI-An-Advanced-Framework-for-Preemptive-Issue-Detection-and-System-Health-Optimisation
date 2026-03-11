import streamlit as st
import psutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression

# ---------------- Page Config ----------------
st.set_page_config(page_title="InferAI Pro", layout="wide", page_icon="⚡")

# ---------------- Advanced CSS (Original Scheme Preserved) ----------------
st.markdown("""
<style>
body, .stApp {background-color:#FFF; color:#FF4500; font-family:'Inter', sans-serif;}
[data-testid="stSidebar"] {background-color:#000 !important; border-right: 2px solid #FF4500; color:#FF4500 !important;}
[data-testid="stSidebar"] * {color:#FF4500 !important;}
.title-container {
    text-align:center; padding:40px 0; margin-bottom:50px;
    background:linear-gradient(135deg,#000,#FF4500); border-radius:20px;
    box-shadow:0 4px 25px rgba(255,69,0,0.25); border:2px solid #FF4500;
}
.title-container .main-title {font-size:56px; font-weight:900; color:#FFF; letter-spacing:2px; text-shadow:2px 2px 8px #000;}
.title-container .subtitle {font-size:22px; color:#FFF; font-weight:500;}
.risk-card {
    padding: 20px; border-radius: 15px; background: #FFF;
    border: 2px solid #FF4500; margin-bottom: 20px; text-align: center;
    box-shadow: 0 4px 15px rgba(255,69,0,0.1);
}
.risk-card h3 {color: #000 !important; font-weight: 700;}
.failure-log {
    font-family: 'Courier New', monospace; font-size: 13px; background: #000;
    color: #FF4500; padding: 15px; border-radius: 10px; height: 250px; 
    overflow-y: scroll; border: 2px solid #FF4500;
}
/* Out-of-the-box Process Metric Styling */
.process-box {
    border-left: 5px solid #FF4500; background: #f9f9f9; padding: 10px; 
    margin-bottom: 5px; border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Initialize Persistent State ----------------
if "log_data" not in st.session_state:
    st.session_state.log_data = []
if "hist_df" not in st.session_state:
    st.session_state.hist_df = pd.DataFrame(columns=['Time', 'CPU', 'Mem', 'Risk'])

# ---------------- Title ----------------
st.markdown("""
<div class="title-container">
    <div class="main-title">InferAI</div>
    <div class="subtitle">Next-Gen Predictive Failure Analysis & Forecasting</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.title("🛡️ Engine Status")
status_placeholder = st.sidebar.empty()
update_interval = st.sidebar.slider("Scan Speed (Seconds)", 0.5, 5.0, 1.0)
st.sidebar.markdown("---")
# UNIQUE FEATURE: Monitoring Mode Toggle
monitor_mode = st.sidebar.selectbox("Analysis Focus", ["CPU Stress", "Memory Leakage", "Composite Risk"])

# ---------------- Core Logic Functions ----------------
def run_deep_scan():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    swap = psutil.swap_memory().percent
    zombies = len([p for p in psutil.process_iter(['status']) if p.info['status'] == psutil.STATUS_ZOMBIE])
    try: io_wait = psutil.cpu_times_percent().iowait
    except: io_wait = 0.0
    
    risk = 0
    alerts = []
    if cpu > 85: risk += 20; alerts.append("THERMAL: High sustained CPU load.")
    if mem > 90: risk += 25; alerts.append("CAPACITY: Physical RAM Exhaustion.")
    if swap > 25: risk += 30; alerts.append("HARDWARE: Swap thrashing/Disk wear.")
    if zombies > 0: risk += 15; alerts.append("KERNEL: OS Process Table Leak.")
    
    return cpu, mem, swap, io_wait, zombies, min(risk, 100), alerts

def predict_next_failure():
    df = st.session_state.hist_df
    if len(df) > 10:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Risk'].values.astype(float)
        model = LinearRegression().fit(X, y)
        return round(max(0, model.predict([[len(df) + 5]])[0]), 2)
    return 0.0

# OUT-OF-THE-BOX: ADVANCED PROCESS ANALYTICS
def get_advanced_procs():
    procs = []
    for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        try:
            pinfo = proc.info
            # Calculate Intensity Score (Conceptual: High Mem + High CPU = Critical)
            pinfo['intensity'] = (pinfo['cpu_percent'] * 0.7) + (pinfo['memory_percent'] * 0.3)
            procs.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if monitor_mode == "Memory Leakage":
        return sorted(procs, key=lambda x: x['memory_percent'], reverse=True)[:4]
    elif monitor_mode == "CPU Stress":
        return sorted(procs, key=lambda x: x['cpu_percent'], reverse=True)[:4]
    else:
        return sorted(procs, key=lambda x: x['intensity'], reverse=True)[:4]

# ---------------- Main Dashboard Loop ----------------
main_placeholder = st.empty()



while True:
    status_placeholder.markdown(f"🟢 **MODE: {monitor_mode.upper()}**")
    cpu, mem, swap, io_wait, zombies, risk, alerts = run_deep_scan()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    future_risk = predict_next_failure()
    st.session_state.log_data.append(f"[{timestamp}] RISK: {risk}% | CPU: {cpu}% | RAM: {mem}%")
    if len(st.session_state.log_data) > 50: st.session_state.log_data.pop(0)
    
    new_point = pd.DataFrame([[timestamp, cpu, mem, risk]], columns=['Time', 'CPU', 'Mem', 'Risk'])
    st.session_state.hist_df = pd.concat([st.session_state.hist_df, new_point], ignore_index=True).tail(100)

    with main_placeholder.container():
        # Top Row Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="risk-card"><h3>RISK INDEX</h3><h1>{risk}%</h1><p>STATE: {"CRITICAL" if risk > 60 else "STABLE"}</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="risk-card"><h3>CPU LOAD</h3><h1>{cpu}%</h1><p>FREQ: ACTIVE</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="risk-card"><h3>RAM USAGE</h3><h1>{mem}%</h1><p>SWAP: {swap}%</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="risk-card"><h3>AI FORECAST</h3><h1>{future_risk}%</h1><p>TREND: PREDICTIVE</p></div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("🚨 Integrity Diagnostics")
            for a in alerts: st.error(a)
            if not alerts: st.success("System integrity verified.")
            
            # OUT-OF-THE-BOX FEATURE: DYNAMIC HOGGER ANALYSIS
            st.subheader(f"🔍 Top Consumers ({monitor_mode})")
            top_procs = get_advanced_procs()
            for p in top_procs:
                color = "#FF4500" if p['intensity'] > 15 else "#000"
                st.markdown(f"""
                <div class="process-box">
                    <b style="color:{color};">{p['name'].upper()}</b><br>
                    <small>CPU: {p['cpu_percent']}% | RAM: {round(p['memory_percent'],1)}% | 
                    Intensity: {round(p['intensity'],1)}</small>
                </div>
                """, unsafe_allow_html=True)

        with col_right:
            st.subheader("📜 Predictive Scan Log")
            log_html = f"<div class='failure-log'>{'<br>'.join(reversed(st.session_state.log_data))}</div>"
            st.markdown(log_html, unsafe_allow_html=True)

        # Chart
        st.subheader("📊 Performance Correlation Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.hist_df['Time'], y=st.session_state.hist_df['CPU'], name="CPU %", line=dict(color='#FF4500')))
        fig.add_trace(go.Scatter(x=st.session_state.hist_df['Time'], y=st.session_state.hist_df['Mem'], name="RAM %", line=dict(color='#000')))
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=300, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True, key=f"cht_{time.time()}")

    time.sleep(update_interval)
