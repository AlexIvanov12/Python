import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.regularizers import L1L2

df = pd.read_csv('BTC-USD.csv')
df.head()

to_drop = ['High','Low', 'Adj Close', 'Volume']
df.drop(columns=to_drop, inplace=True, axis=1)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', drop=True, inplace=True)
df.head()

fig, ax = plt.subplots(2, 1, figsize = (20,7))
ax[0].plot(df['Open'], label = 'Open', color = 'green')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Price')
ax[0].legend()
ax[1].plot(df['Close'], label = 'Close', color = 'red')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Price')
ax[1].legend()
fig.show()

MS = MinMaxScaler()
df[df.columns] = MS.fit_transform(df)
df.head()

training_size = round(len(df)*0.80)
train_data = df[:training_size]
test_data = df[training_size:]

def create_sequence(dataset):
  sequences = []
  labels = []
  start_idx = 0
  for stop_idx in range(5, len(dataset)):
    sequences.append(dataset.iloc[start_idx:stop_idx])
    labels.append(dataset.iloc[stop_idx])
    start_idx +=1
  return (np.array(sequences), np.array(labels))

train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

# Будування моделі LSTM з регуляризацією та нормалізацією
# Assuming X_train has been defined with the appropriate shape
reg = L1L2(l1=0.01, l2=0.01)  # L1 and L2 regularization

model = Sequential([
    LSTM(units=150, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2]), kernel_regularizer=reg),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(units=100, return_sequences=True, kernel_regularizer=reg),
    BatchNormalization(),
    Dropout(0.01),
    LSTM(units=70, return_sequences=False, kernel_regularizer=reg),
    Dropout(0.1),
    Dense(units=50, kernel_regularizer=reg),
    Activation('relu'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(units=30, kernel_regularizer=reg),
    Activation('relu'),
    Dropout(0.1),
    Dense(units=2)  # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'accuracy'])

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

history = model.fit(train_seq, train_label, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=2)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

test_predicted = model.predict(test_seq)
test_inverse_predicted = MS.inverse_transform(test_predicted)
# Assuming 'test_inverse_predicted' is an array with shape (57, 2)
corrected_test_inverse_predicted = test_inverse_predicted

# Ensure the index from 'df' matches the length of 'corrected_test_inverse_predicted'
predicted_df = pd.DataFrame(corrected_test_inverse_predicted, columns=['open_predicted', 'close_prediction'])
predicted_df.index = df.iloc[-corrected_test_inverse_predicted.shape[0]:].index

# Concatenate using the DataFrame with matching number of rows
df_slic_data = pd.concat([
    df.iloc[-corrected_test_inverse_predicted.shape[0]:].copy(),
    predicted_df
], axis=1)

# If 'Open' and 'Close' columns were transformed using MinMaxScaler, we need to inverse the transformation
# Assuming the original 'df' had only 'Open' and 'Close' columns.
df_slic_data[['Open', 'Close']] = MS.inverse_transform(df_slic_data[['Open', 'Close']])

df_slic_data.head()

df_slic_data[['Open', 'open_predicted']].plot(figsize = (20, 7))
plt.xticks(rotation = 45)
plt.xlabel('Date', size = 15)
plt.ylabel('Stock Price', size = 15)
plt.title('Actual vs Predicted for Open price', size = 15)
plt.show()

df_slic_data[['Close', 'close_prediction']].plot(figsize=(12, 6))  # Adjusted column name
plt.xticks(rotation=45)
plt.xlabel('Date', size=15)
plt.ylabel('Stock Price', size=15)
plt.title('Actual vs Predicted for Close Price', size=15)
plt.show()
