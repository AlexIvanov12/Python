import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt 

df = pd.read_csv('---')
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', drop=True, inplace=True)
X = df[['price_entry', 'price_exit', 'macd_entry', 'macd_exit', 'signal_entry', 'signal_exit', 'rsi_entry', 'rsi_exit', 'volume_entry', 'volume_exit']]
Y = df['profit_loss']

#fix, ax = plt.subplot(2, 1, figsize = (20, 7))

# Normalizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation= 'relu', input_shape = (X_train.shape[1],)),
    Dense(64, activation = 'relu'),
    Dense(3, activation = 'linear')
])

model.compile(optimizer='adam' loss='mean_squred_error')
model.fit(X_train, y_train, batch_size= 32, epochs= 25, validation_split=0.2)
model.summary()

callback = [
    EarlyStopping(monitor='val_loss', patience= 5),
    ModelCeckpoint(filepath = "TRAiding_model.h5", monitor = 'val_loss', save_best_only = True),
    ReduceLROnPlateau(monitor='val_loss', factor= 0.1, patience= 2)
]


