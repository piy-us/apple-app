from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import pickle
import time
import os
import math
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from flask import render_template


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

app = Flask(__name__)

df = pd.read_csv('AAPL (6) (1).csv')
df1 = df.reset_index()['Close']
scaler = MinMaxScaler(feature_range=(0,1))
with open('apple22-23', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/plot')
def plot():
    timestamp = int(time.time())
    fig = make_subplots()

    trace = go.Scatter(x=df1.index, y=df1, mode='lines', name='Close Price')
    fig.add_trace(trace)

    fig.update_layout(title='Train vs Test Predictions',
                    xaxis=dict(title='Index'),
                    yaxis=dict(title='Close Price'),
                    legend=dict(font=dict(color='white')),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(color='white')
    fig.update_yaxes(color='white')


    graph_filename = 'plot.html'
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    graph_filepath = os.path.join(static_dir, graph_filename)

    fig.write_html(graph_filepath)
    return render_template('plot.html', graph_filename=graph_filename, timestamp=timestamp)

def preprocessing_data():
    df = pd.read_csv('AAPL (6) (1).csv')
    df1 = df.reset_index()['Close']
    df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
    training_size = int(len(df1) * 0.65)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]
    #print('pd',df1.shape)
    return train_data, test_data, df1


def predicting():
    train_data, test_data, df1 = preprocessing_data()
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    #print('ping',df1.shape)

    return train_predict, test_predict, df1, y_train, y_test


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


@app.route('/performance')
def performance():
    train_predict, test_predict, df1, y_train, y_test = predicting()

    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    return render_template('performance.html', train_rmse=train_rmse, test_rmse=test_rmse)


@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        num_days = int(request.form['num_days'])
        timestamp = int(time.time())
        train_predict, test_predict, df1, y_train, y_test = predicting()
        timestamp = int(time.time())
        print('pd1',df1.shape)


        look_back = 100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

        fig = make_subplots()
        #fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=scaler.inverse_transform(df1).flatten(), mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=trainPredictPlot.flatten(), mode='lines', name='Train Predicted',line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=scaler.inverse_transform(df1).flatten(), mode='lines', name='Actual',line=dict(color='blue')))

        fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=testPredictPlot.flatten(), mode='lines', name='Test Predicted',line=dict(color='green')))


        fig.update_layout(title='Train vs Test Predictions',
                        xaxis=dict(title='Index'),
                        yaxis=dict(title='Close Price'),
                        legend=dict(font=dict(color='white')),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(color='white')
        fig.update_yaxes(color='white')

        graph_filename1 = 'train_predictions.html'
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        graph_filepath1 = os.path.join(static_dir, graph_filename1)
        fig.write_html(graph_filepath1)




        returndata=preprocessing_data()
        test_data=returndata[1]
        x_input = test_data[116:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        lst_output = []
        n_steps = 100
        num_days = num_days
        if num_days>0:
            i = 0
            while i < num_days:
                if len(temp_input) > 100:
                    x_input = np.array(temp_input[1:])
                    x_input = x_input.reshape(1, -1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    temp_input = temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i = i + 1
                else:
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i = i + 1
        day_new = np.arange(1, 101)
        day_pred = np.arange(101, 101 + num_days)
        fig = make_subplots()
        #fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=scaler.inverse_transform(df1).flatten(), mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=day_new, y=scaler.inverse_transform(df1[518:]).flatten(), mode='lines', name='Train',line=dict(color='blue')))
        fig.add_trace(go.Scatter(x= day_pred, y=scaler.inverse_transform(lst_output).flatten(), mode='lines', name='Prediction',line=dict(color='green')))

        #fig.add_trace(go.Scatter(x=day_pred, y=testPredictPlot.flatten(), mode='lines', name='Test Predicted',line=dict(color='blue')))


        fig.update_layout(title='Train vs Test Predictions',
                        xaxis=dict(title='Index'),
                        yaxis=dict(title='Close Price'),
                        legend=dict(font=dict(color='white')),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(color='white')
        fig.update_yaxes(color='white')

        graph_filename2 = 'listoutput.html'
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        graph_filepath2 = os.path.join(static_dir, graph_filename2)
        fig.write_html(graph_filepath2)


        df22=pd.read_csv('AAPL (5) (1).csv')
        df23=df22.reset_index()['Close']
        day_df23 = np.arange(101, 101 + num_days)
        df23=df23[179:]
        day_df23 = df23.index
        actual_data = df23[:num_days].values

        #plt.plot(day_pred,scaler.inverse_transform(lst_output))
        if num_days<=71:
            fig = make_subplots()
            #fig.add_trace(go.Scatter(x=np.arange(len(df1)), y=scaler.inverse_transform(df1).flatten(), mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=day_new, y=scaler.inverse_transform(df1[518:]).flatten(), mode='lines', name='Train Predicted',line=dict(color='blue')))
            fig.add_trace(go.Scatter(x= day_pred, y=scaler.inverse_transform(lst_output).flatten(), mode='lines', name='Prediction',line=dict(color='green')))
            fig.add_trace(go.Scatter(x=day_pred, y=actual_data, mode='lines', name='Actual',line=dict(color='orange')))


            #fig.add_trace(go.Scatter(x=day_pred, y=testPredictPlot.flatten(), mode='lines', name='Test Predicted',line=dict(color='blue')))


            fig.update_layout(title='Train vs Test Predictions',
                            xaxis=dict(title='Index'),
                            yaxis=dict(title='Close Price'),
                            legend=dict(font=dict(color='white')),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)')
            fig.update_xaxes(color='white')
            fig.update_yaxes(color='white')

            graph_filename3 = 'day23.html'
            static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
            graph_filepath3 = os.path.join(static_dir, graph_filename3)
            fig.write_html(graph_filepath3)


    
        return render_template('forecast.html', num_days=num_days,
                               graph_filename1=graph_filename1,graph_filename2=graph_filename2,graph_filename3=graph_filename3,
                            timestamp=timestamp)

    return render_template('forecast.html')


if __name__ == '__main__':
    app.run()
