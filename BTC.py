import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator

df =  pd.read_csv('/content/drive/MyDrive/BTC/train_dataset.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['24h Open (USD)', '24h High (USD)', '24h Low (USD)'], inplace=True)


close_data = df['Closing Price (USD)'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.65
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

look_back = 20

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import losses, optimizers

model = Sequential()
model.add(LSTM(60,return_sequences=True, activation='relu',input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(LSTM(60,return_sequences=True, activation='relu',input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(LSTM(60,return_sequences=True, activation='relu',input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(LSTM(60,return_sequences=True, activation='relu',input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer=optimizers.Adam() , loss=losses.mean_squared_error)

num_epochs = 80
model.fit(train_generator, epochs=num_epochs, verbose=1)

import plotly.graph_objects as go
prediction = model.predict(test_generator)

close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "Google Stock",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 30
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)

trace1 = go.Scatter(
    x = forecast_dates,
    y = forecast,
    mode = 'lines',
    name = 'Data'
)
layout = go.Layout(
    title = "BTC Stock",
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Close"}
)

fig = go.Figure(data=[trace1], layout=layout)
fig.show()
