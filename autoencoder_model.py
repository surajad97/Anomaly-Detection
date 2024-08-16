import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler  

import tensorflow as tf
from tensorflow.keras.models import Model

#load data
data_dict = {}
for i in range(22):
  i_str = str(i)
  if len(i_str) == 1:
    term = '0'+i_str
  else:
    term = i_str
  for split in ['','_te']:
    term = term + split
    data_dict[term] = np.loadtxt('./data/d' + term + '.dat')
data_dict['00'] = data_dict['00'].T

fault = '01'
X_train = np.hstack((data_dict['00'][:,:22],data_dict['00'][:,41:]))

#standard scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)

X_test = np.hstack((data_dict[fault+'_te'][:,:22],data_dict[fault+'_te'][:,41:]))
X_test_norm = scaler.transform(X_test)

input_dim = X_train.shape[1]
encoding_dim = 32

Autoencoder = tf.keras.Sequential([
    # Encoder part
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(encoding_dim, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),

    # Decoder part
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(encoding_dim, activation="relu"),
    tf.keras.layers.Dense(input_dim, activation="linear")  # Output layer
])


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='AER_AD_weights.weights.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

Autoencoder.compile(optimizer='adam', loss="mse")

history = Autoencoder.fit(X_train_norm, X_train_norm, epochs=200, batch_size=32, validation_split=0.2,
                    shuffle=True,
                    callbacks=[checkpoint_callback]
                    )

#saving model and scaler
Autoencoder.save('AER_AD_model.keras')
joblib.dump(scaler, 'AER_scaler.pkl')