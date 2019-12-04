import sys
sys.path.append("..") 

import preproccessor.preprocessing as prep
import os
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# constant for training
num_spheres = prep.num_spheres
time_predicted = 5
velocity = False
if not velocity:
    num_features = 3
input_size = time_predicted * num_spheres * num_features

def LSTM_model(shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=(shape[1], shape[2])))
    model.add(Dense(3))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

def main():
    temp_data = prep.merging_files(velocity)
    data = prep.to_supervised(temp_data, n_in=time_predicted).values
    train_size = int(data.shape[0] * 0.7)

    train, test = data[:train_size, :], data[train_size:, :]
    train_X, train_y = train[:, :input_size], train[:, input_size:]
    test_X, test_y = test[:, :input_size], test[:, input_size:]
    train_X = train_X.reshape((train_X.shape[0], 
        time_predicted, num_features * num_spheres))
    test_X = test_X.reshape((test_X.shape[0], 
        time_predicted, num_features * num_spheres))

    model = LSTM_model(train_X.shape)
    history = model.fit(train_X, train_y, epochs=50, 
        batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    yhat = model.predict(test_X)
    rmse = math.sqrt(mean_squared_error(test_y, yhat))
    print(rmse)

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    

if __name__ == "__main__":
    main()
    