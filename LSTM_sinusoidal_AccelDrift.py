import pandas as pd
import numpy as np
from keras import layers
from keras import models
from keras import callbacks
from keras import metrics

count = 0
X_train = []
y_train = []
dataAccelDrift = pd.read_csv('.\\data\\Sinusoidal_TestData_AccelDrift_VelDrift.csv', sep=',', header=0)
data = dataAccelDrift
print(data)
activation_type = 'linear'
#activation_type = 'relu'
#X = np.zeros(1,(len(data)))
#y = np.zeros(1,(len(data)))

print(type(data))
X = np.array(data['Time'].values.tolist())
y = np.array(data['True Position'].values.tolist())
print(X)
print(X[0:5])
print(type(X))
print(y[-1])
X_predict = np.array([1.001,1.005,1.1])
X_predict = np.ndarray(shape=(3,3))#,buffer=[1.001,1],[1.005,1],[1.1,1])
X_predict[0,:] = [1.001,3.101126,8.398864806]
X_predict[1,:] = [1.005,3.107637,8.411282332]
X_predict[2,:] = [1.1,4.106141,8.753936799]
print(X_predict)
print(X_predict.shape[1])

saved_model_name = 'best_sinusoidal_model_batch32_sigmoid_time.keras'
callbacks = [
    callbacks.ModelCheckpoint(
        saved_model_name, save_best_only=True, monitor="val_loss"
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
#print(np.shape(X))
#for i in range(len(X)):
#    X_train.append(X[i])
#    y_train.append(y[i])
XX_train = []
XX_train = [data.loc[:,"Time"].values,data.loc[:,"Estimated Velocity"].values, data.loc[:,"Estimated Position"].values]
yy_train = data.loc[:,["True Position"]].values
X_train, y_train = np.transpose(np.array(XX_train)), np.transpose(np.array(yy_train))
print('Double check',X_train.shape)
print('Double checker',type(X_train))
print("X_train =",X_train)

# define model
model = models.Sequential()
model.add(layers.Input(shape=[3]))
#model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
#model.add(layers.LSTM(100)) #, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(500, activation=activation_type))
model.add(layers.Dense(100, activation=activation_type))
model.add(layers.Dense(10, activation=activation_type))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=[metrics.MeanSquaredError()])
model.summary()
# fit model
#X_train = np.reshape(X_train, (X_train.shape[0]. X_train.shape[1], 1))
print('Triple check',np.shape(X_train))
model.fit(X_train, yy_train, epochs=500, verbose=1,
    batch_size=32,
    callbacks=callbacks#,
    #validation_split=0.2
)
# demonstrate prediction

yhat = model.predict(X_predict, verbose=0)
print(yhat)
