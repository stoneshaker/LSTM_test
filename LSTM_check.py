import pandas as pd
import numpy as np
from keras import layers
from keras import models

count = 0
X_train = []
y_train = []
data = pd.read_csv('.\\data\\dataVelTruth.csv', sep=',', header=0)
print(data)

#X = np.zeros(1,(len(data)))
#y = np.zeros(1,(len(data)))

print(type(data))
X = np.array(data['Time'].values.tolist())
y = np.array(data['Position'].values.tolist())
print(X)
print(X[0:5])
print(type(X))
X_predict = np.array([1.001,1.005,1.1])

print(np.shape(X))
for i in range(len(X)):
    X_train.append(X[i])
    y_train.append(y[i])
XX_train = data.loc[:,["Time"]].values
yy_train = data.loc[:,["Position"]].values
X_train, y_train = np.array(X_train), np.array(y_train)
print('Double check',np.shape(XX_train))
print('Double check',type(XX_train))
print(np.transpose(X_train))
# define model
model = models.Sequential()
model.add(layers.Input(shape=(np.shape(XX_train[1]))))
model.add(layers.Dense(50, activation='linear'))
model.add(layers.Dense(10, activation='linear'))
model.add(layers.Dense(3, activation='linear'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
#X_train = np.reshape(X_train, (X_train.shape[0]. X_train.shape[1], 1))
print('Triple check',np.shape(X_train))
model.fit(XX_train, yy_train, epochs=500, verbose=1)
# demonstrate prediction

yhat = model.predict(X_predict, verbose=0)
print(yhat)
