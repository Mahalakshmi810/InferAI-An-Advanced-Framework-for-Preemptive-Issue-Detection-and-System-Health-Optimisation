import streamlit as st
import psutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
import plotly.graph_objects as go
import time


st.set_page_config(page_title="InferAI", layout="wide", page_icon="🖥️")


st.markdown("""
<style>
body, .stApp {background-color:#FFF; color:#FF4500; font-family:'Inter',sans-serif;}
[data-testid="stSidebar"] {background-color:#000 !important; color:#FF4500 !important; padding:25px;}
[data-testid="stSidebar"] * {color:#FF4500 !important;}
.title-container {text-align:center; padding:40px 0; margin-bottom:50px;
background:linear-gradient(135deg,#000,#FF4500); border-radius:20px;
box-shadow:0 4px 25px rgba(255,69,0,0.25); border:2px solid #FF4500;}
.title-container .main-title {font-size:56px; font-weight:900; color:#FFF;
letter-spacing:2px; text-shadow:2px 2px 8px #000;}
.title-container .subtitle {font-size:22px; color:#FFF; font-weight:500;}
.stMetric {background:#FFF; border:2px solid #FF4500; border-radius:15px; padding:20px;}
</style>
""", unsafe_allow_html=True)

#Header File
st.markdown("""
<div class="title-container">
    <div class="main-title">InferAI</div>
    <div class="subtitle">Real-Time Hardware Failure Monitor</div>
</div>
""", unsafe_allow_html=True)

#Sidebar
st.sidebar.header("⚙️ Configuration")
history_length = st.sidebar.slider("History Length (Data Points)", 50, 500, 100)
update_interval = st.sidebar.slider("Update Interval (Seconds)", 0.1, 5.0, 0.5)

#State Detection
if "sensor_history" not in st.session_state:
    st.session_state.sensor_history = pd.DataFrame(
        columns=['CPU', 'Memory', 'Disk']
    )

if "anomaly_count" not in st.session_state:
    st.session_state.anomaly_count = 0

if "run" not in st.session_state:
    st.session_state.run = True

#Autoencoder
input_dim = 3
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

#LSTM
lstm_model = Sequential([
    LSTM(8, activation='relu', input_shape=(1, input_dim)),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#GUI 
st.subheader("📊 Live Sensor Metrics & Predictions")
placeholder = st.empty()

#Main Programming Loop
while st.session_state.run:

    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent

    new_data = pd.DataFrame({
        'CPU': [cpu],
        'Memory': [memory],
        'Disk': [disk]
    })

    st.session_state.sensor_history = pd.concat(
        [st.session_state.sensor_history, new_data],
        ignore_index=True
    )

    if len(st.session_state.sensor_history) > history_length:
        st.session_state.sensor_history = (
            st.session_state.sensor_history.iloc[-history_length:]
        )

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(st.session_state.sensor_history)

    autoencoder.fit(
        scaled_data,
        scaled_data,
        epochs=5,
        batch_size=16,
        verbose=0
    )

    reconstruction = autoencoder.predict(scaled_data, verbose=0)
    mse = np.mean(np.power(scaled_data - reconstruction, 2), axis=1)
    threshold = np.percentile(mse, 95)

    #Anamoly Count 
    if mse[-1] > threshold:
        st.session_state.anomaly_count += 1
    else:
        st.session_state.anomaly_count = 0

    ae_pred = st.session_state.anomaly_count >= 3

    #Rule Fixage
    xgb_pred = 1 if cpu > 85 or memory > 85 or disk > 90 else 0
    lstm_pred = 1 if cpu > 80 and memory > 80 else 0

    status = "✅ Normal"
    if xgb_pred == 1 or lstm_pred == 1 or ae_pred:
        status = "⚠️ Potential Failure Detected!"

    with placeholder.container():
        cols = st.columns(3)
        cols[0].metric("System Status", status)
        cols[1].metric("CPU Usage (%)", f"{cpu:.1f}")
        cols[2].metric("Memory Usage (%)", f"{memory:.1f}")
        st.metric("Disk Usage (%)", f"{disk:.1f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.sensor_history['CPU'],
            mode='lines+markers',
            name='CPU %'
        ))
        fig.add_trace(go.Scatter(
            y=st.session_state.sensor_history['Memory'],
            mode='lines+markers',
            name='Memory %'
        ))
        fig.add_trace(go.Scatter(
            y=st.session_state.sensor_history['Disk'],
            mode='lines+markers',
            name='Disk %'
        ))

        fig.update_layout(
            title="Hardware Usage Over Time",
            xaxis_title="Time Steps",
            yaxis_title="Percentage",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    time.sleep(update_interval)
