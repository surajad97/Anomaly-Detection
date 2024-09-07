import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from io import StringIO

# Load the pre-trained models and scaler
Autoencoder = load_model('AER_AD_model.keras')
threshold = 0.6356108989251031  # Pre-defined threshold
scaler = joblib.load('AER_scaler.pkl')
fault = '01'  # Default fault type

# Function to detect anomalies
def detect_anomalies(data, model, threshold):
    reconstructed_data = model.predict(data)
    reconstruction_error = np.mean(np.square(data - reconstructed_data), axis=1)
    anomalies = reconstruction_error > threshold
    nr_anomalies = np.sum(anomalies)
    return nr_anomalies, reconstruction_error

# Function to plot the error distribution
def plot_error_dist(data, model, threshold):
    nr_anomalies, reconstruction_error = detect_anomalies(data, model, threshold)
    fig, ax = plt.subplots()
    ax.plot(reconstruction_error, label='Reconstructed MSE')
    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {np.round(threshold, 2)}')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.legend()
    plt.title('Reconstruction Error vs Anomaly Threshold')
    return fig

# Streamlit app layout
st.title('Anomaly Detection Web App')

st.write('This web app allows you to upload a CSV file and detects anomalies using an autoencoder model.')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Button to trigger prediction
if st.button('Predict'):
    if uploaded_file is not None:
        try:
        # Read the uploaded file
            st.write(f"File uploaded: {uploaded_file.name}")
            data_df = pd.read_csv(StringIO(uploaded_file.getvalue().decode('utf-8')))

            st.write("Preview of the uploaded data:")
            st.write(data_df.head())

            # Convert the DataFrame to NumPy for model input
            data = data_df.to_numpy()

            # Preprocess the data
            RT_data_stream = np.hstack((data[:, :22], data[:, 41:]))
            RT_data_norm_stream = scaler.transform(RT_data_stream)

            # Plot error distribution
            fig = plot_error_dist(RT_data_norm_stream, Autoencoder, threshold)

            # Display the plot
            st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error('Please upload a CSV file.')
