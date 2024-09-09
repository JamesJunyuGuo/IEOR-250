import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def read_data():
    df = pd.read_csv('AirPassengers.csv',index_col='Month',parse_dates=True)
    df.index.freq = 'MS'
    train_data = df.iloc[:108] # assume we have observed the first 75% data, 108 data points
    test_data = df.iloc[108:] # consider 36-step forecast
    plt.clf()
    ax_ = train_data['#Passengers'].plot(legend=True, label='TRAIN')
    test_data['#Passengers'].plot(legend=True, label='TEST', figsize=(12, 8), ax = ax_)
    fig = ax_.get_figure()
    fig.savefig('historical.pdf')
    return train_data, test_data

def run_task1(train_data):
    '''
    Task 1: Time Series Decomposition
    '''
    plt.clf()
    fig = seasonal_decompose(train_data['#Passengers']).plot().get_figure()
    fig.savefig('decomposition.pdf')

def run_task2(train_data, test_data):
    '''
    Task 2 Single exponential smoothing
    '''
    alpha = 0.1 # specify a value close to 1
    model = SimpleExpSmoothing(train_data['#Passengers']).fit(smoothing_level=alpha, optimized=False)
    test_predictions = model.forecast(36).rename('SES Forecast')
    MSE = mean_squared_error(test_data['#Passengers'], test_predictions)
    plt.clf()
    ax_ = train_data['#Passengers'].plot(legend=True,label='TRAIN')
    test_data['#Passengers'].plot(legend=True,label='TEST',figsize=(12,8), ax = ax_)
    test_predictions.plot(legend=True,label='SES Forecast', title = 'SES with MSE ' + str(MSE), ax = ax_)
    fig = ax_.get_figure()
    fig.savefig('SES.pdf')

def run_task3(train_data, test_data):
    '''
    Task 3: Double Exponential Smoothing
    '''
    double_model = ExponentialSmoothing(train_data['#Passengers'],trend='add').fit()
    test_predictions = double_model.forecast(36).rename('DES Forecast')
    MSE = mean_squared_error(test_data['#Passengers'], test_predictions)
    plt.clf()
    ax_ = train_data['#Passengers'].plot(legend=True,label='TRAIN')
    test_data['#Passengers'].plot(legend=True,label='TEST',figsize=(12,8), ax = ax_)
    test_predictions.plot(legend=True,label='DES Forecast', title = 'DES with MSE ' + str(MSE), ax = ax_)
    fig = ax_.get_figure()
    fig.savefig('DES.pdf')

def run_task4(train_data, test_data):
    '''
    Task 4: Triple exponential
    '''
    triple_model = ExponentialSmoothing(train_data['#Passengers'],trend='add',seasonal='add',seasonal_periods=12).fit()
    test_predictions = triple_model.forecast(36).rename('TES Forecast')
    MSE = mean_squared_error(test_data['#Passengers'], test_predictions)
    plt.clf()
    ax_ = train_data['#Passengers'].plot(legend=True,label='TRAIN')
    test_data['#Passengers'].plot(legend=True,label='TEST',figsize=(12,8), ax = ax_)
    test_predictions.plot(legend=True, label='TES Forecast', title='TES with MSE ' + str(MSE), ax = ax_)
    fig = ax_.get_figure()
    fig.savefig('TES.pdf')

if __name__ == '__main__':
    train_data, test_data = read_data()
    run_task1(train_data)
    run_task2(train_data, test_data)
    run_task3(train_data, test_data)
    run_task4(train_data, test_data)