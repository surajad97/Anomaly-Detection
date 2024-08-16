import streamlit as st
import numpy as np
import pandas as pd

import tensorflow as tf

from pipeline import detect_anomalies

autoencoder = tf.keras.models.load_model('AER_AD_model.keras')
threshold = 0.6356108989251031

st.title('Real-Time Anomaly Detection with Autoencoder')

st.write('Upload your test data for anomaly detection.')

# File upload
fault = '01'
uploaded_file = st.file_uploader('./data/d' + fault + '_te', type='csv')

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.write('Data preview:')
    st.dataframe(data.head())

    # Assuming the data needs to be preprocessed or scaled in the same way as training data
    # Here, you would add any necessary preprocessing steps
    # For example, if you used StandardScaler during training, load and apply it here

    # Convert data to numpy array for prediction
    data_np = data.to_numpy()

    # Detect anomalies
    anomalies, reconstruction_error = detect_anomalies(data, autoencoder,  threshold)

    # Add the results to the dataframe
    data['Reconstruction Error'] = reconstruction_error
    data['Anomaly'] = anomalies

    # Show results
    st.write('Anomaly Detection Results:')
    st.dataframe(data)

    # Optionally, visualize anomalies
    st.write('Anomalies:')
    st.write(data[data['Anomaly']])
else:
    st.write("Please upload a file.")