import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to process and detect anomalies in real-time
def detect_anomalies(data, model, threshold):

    # Predict the reconstruction from the model
    reconstructed_data = model.predict(data)

    # Calculate reconstruction error
    reconstruction_error = np.mean(np.square(data - reconstructed_data), axis=1)

    # Check if the error exceeds the threshold
    anomalies = reconstruction_error > threshold
    nr_anomalies = np.sum(anomalies)

    return nr_anomalies


fault = '01'
test_data = np.loadtxt('./data/d' + fault + '_te' + '.dat')
scaler = joblib.load('AER_scaler.pkl')
encoding_dim = 32
threshold = 0.6356108989251031

#load saved model weights
Autoencoder = tf.keras.models.load_model('AER_AD_model.keras')

RT_data_stream = np.hstack((test_data[:,:22],test_data[:,41:]))
RT_data_norm_stream = scaler.transform(RT_data_stream)
input_dim = RT_data_stream.shape[1]

detect_anomalies(RT_data_norm_stream, Autoencoder, threshold)

