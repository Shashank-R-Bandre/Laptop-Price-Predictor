"""
streamlit run app.py

pip install pipreqs
pipreqs --encoding=utf8

git init

"""
import streamlit as st
import pickle
import numpy as np
import pandas
import sklearn

# Import the model
pipe = pickle.load(open('pipee.pkl', 'rb'))
df = pickle.load(open('dff.pkl', 'rb'))

st.title('Laptop Price Predictor')

# Brands of Laptop
brand = st.selectbox('Brand', df['Company'].unique())

# Type of Laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram of Laptop
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight of Laptop
weight = st.number_input('Weight of the Laptop')

# Touchscreen in Laptop
touchscreen = st.selectbox('Touch Screen', ['Yes', 'No'])

# IPS Panel in Laptop
ips = st.selectbox('IPS Panel', ['Yes', 'No'])

# Screen Size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution',
                          ['1920 x 1080', '1366 x 768', '1600 x 900', '3840 x 2160', '3200 x 1800', '2880 x 1800',
                           '2560 x 1600', '2560 x 1440', '2304 x 1440'])

# CPU
cpu = st.selectbox('CPU', df['CPU Brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['GPU Brand'].unique())

# OS
os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Laptop Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = (((X_res ** 2) + (Y_res ** 2)) ** 0.5) / screen_size

    query = np.array([brand, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)
    st.title('Laptop Price : ' + str(int((np.exp(pipe.predict(query))))))
