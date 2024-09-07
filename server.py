import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = FastAPI()

Autoencoder = load_model('AER_AD_model.keras')
threshold = 0.6356108989251031  
scaler = joblib.load('AER_scaler.pkl')
fault = '01'  

def detect_anomalies(data, model, threshold):
    # Predict the reconstruction from the model
    reconstructed_data = model.predict(data)
    # Calculate reconstruction error
    reconstruction_error = np.mean(np.square(data - reconstructed_data), axis=1)
    # Check if the error exceeds the threshold
    anomalies = reconstruction_error > threshold
    nr_anomalies = np.sum(anomalies)

    return nr_anomalies, reconstruction_error

#plot the error distributions
def plot_error_dist(data, model, threshold):
    nr_anomalies, reconstruction_error = detect_anomalies(data, model, threshold)
    fig, ax = plt.subplots()
    ax.plot(reconstruction_error, label = 'Reconstructed MSE')
    ax.axhline(y=threshold, color='red', linestyle = '--', label = f'Threshold = {np.round(threshold, 2)}')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.legend()
    plt.title('Reconstruction Error vs Anomaly Threshold')
    return fig

@app.get('/')
def home():
    return "Process Fault Detection API"

@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    # Convert the CSV content to a pandas DataFrame
    data_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    data = data_df.to_numpy()
    
    RT_data_stream = np.hstack((data[:,:22],data[:,41:]))
    RT_data_norm_stream = scaler.transform(RT_data_stream)
    
    nr_anomalies = detect_anomalies(RT_data_norm_stream, Autoencoder, threshold)[0]
    plot_error_dist(RT_data_norm_stream, Autoencoder, threshold)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer to the beginning

    return StreamingResponse(buf, media_type="image/png")

if __name__ == '__main__':
    uvicorn.run(app, host= '127.0.0.1', port= 8000)
#uvicorn server:app --reload