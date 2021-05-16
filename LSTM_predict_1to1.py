import pandas as pd
#coding=utf-8
# file path should be changed
conflict = pd.read_excel("H:/output/total/AR01total_conflict.xlsx",header=None,engine='openpyxl')
conflict['conflict']= pd.read_excel("H:/output/total/AR01total_conflict.xlsx",header=None,engine='openpyxl')
datasets = pd.DataFrame()
datasets['conflict'] = conflict['conflict']

def training(datasets):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    from pandas import DataFrame
    from pandas import concat

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # Input (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # Output (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    dataset = datasets
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, 1, 1)

    values = reframed.values
    n_train_hours = 1440 * 5
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    from sklearn.metrics import mean_squared_error,mean_absolute_error
    from keras.models import Sequential
    from keras.layers import Dense,Dropout
    from keras.layers import LSTM
    from keras.optimizers import Adam
    from matplotlib import pyplot
    import numpy as np
    model = Sequential()
    model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    history = model.fit(train_X, train_y, epochs=150, batch_size=240, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    from numpy import concatenate
    from keras.layers import LSTM
    from math import sqrt
    # 开始预测
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    predict_list = []
    real_list =[]
    for i in range(len(inv_yhat)):
        predict_list.append(inv_yhat[i])
    for j in range(len(inv_y)):
        real_list.append(inv_y[j])

    rmse = sqrt(mean_squared_error(predict_list, real_list))
    mae = mean_absolute_error(predict_list, real_list)
    print('Test RMSE: %.7f' % rmse)
    print('Test MAE: %.7f' % mae)
    print("training completed")
    return rmse, mae

###################################

r_m_s_e = []
m_a_e = []
import math
import numpy as np
import time
start = time.time()
for i in range(10):
    rmse, mae = training(datasets)
    r_m_s_e.append(rmse)
    m_a_e.append(mae)
print('Ave MAE: %.5f' % np.mean(m_a_e))
print('Ave RMSE: %.5f' % np.mean(r_m_s_e))
