import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler #for feature scaling

import pickle

# Importing data

file= 'DJI 1985 to 2019 for Practical Assessment 1.xlsx'
df = pd.read_excel(file)

df.describe()

# Data Pre-Processing

train = df.iloc[:, 1:2].values #selecting the open column

#feature scaling

scaled= MinMaxScaler(feature_range= (0,1))

scaled_train= scaled.fit_transform(train)

X_train = []
y_train = []

#timestamp of 50
for i in range(50, len(scaled_train)):
    X_train.append(scaled_train[i-50:i, 0])
    y_train.append(scaled_train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the LSTM model

from keras.layers import LSTM

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



# Initialising the RNN
pred = Sequential()

# first layer
pred.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
pred.add(Dropout(0.1))

# second layer
pred.add(LSTM(units = 50, return_sequences = True))
pred.add(Dropout(0.1))

# third layer
pred.add(LSTM(units = 50, return_sequences = True))
pred.add(Dropout(0.1))

#fourth layer
pred.add(LSTM(units = 50))
pred.add(Dropout(0.1))

# Output layer
pred.add(Dense(units = 1))

pred.compile(optimizer = 'adam', loss = 'mean_squared_error')


pred.fit(X_train, y_train, epochs = 3, batch_size = 30)

filename = 'pkl_file.sav'
pickle.dump(pred, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

test_data_file = pd.read_excel('test.xlsx')
test = test_data_file.iloc[:, 1:2].values


dataset = pd.concat((train, test), axis = 0)

pred_input = dataset[len(dataset) - len(test) - 50:].values

pred_input = pred_input.reshape(-1,1)

pred_input = scaled.transform(pred_input)

x_test = []

for i in range(50, 70):
    x_test.append(pred_input[i-50: i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred_dji = loaded_model.predict(x_test)

pred_dji = scaled.inverse_transform(pred_dji)

# Visualising the results
plt.plot(test, color = 'red', label = 'Real DJI value')
plt.plot(pred_dji, color = 'blue', label = 'Predicted DJI value')
plt.title('DJI Prediction')
plt.xlabel('Time')
plt.ylabel('DJI value')
plt.legend()
plt.show()