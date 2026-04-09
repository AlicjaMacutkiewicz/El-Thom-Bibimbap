import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def convert_to_stationary(dataframe, col):
    # differencing the data to make the time series stationary
    # and dropping the missing values created after differencing
    dataframe[col] = dataframe[col].diff().dropna()

    # differencing the second time if the data is still stationary
    # after the first transformation
    if check_stationarity(dataframe[col].dropna())==False: # why do I have to put .dropna here sklsdjksjfdsflj
        dataframe[col] = dataframe[col].diff().dropna()
    # TODO: for now the if the data is still not stationary after the
    # second transformation the code continues as if the time series
    # were stationary. Check whether there is a better way to ensure
    # stationarity without overdifferencing

def check_stationarity(data_series):
    result = adfuller(data_series)
    # the data is stationary if the p-value drawn from the
    # ADF test is less or equal 0.05
    if result[1]<=0.05:
        return True
    else:
        return False

# Splitting the data into training and testing datasets:
# The data from first two flights is the training data
# The data from the third flight is the testing data

training_sensors1 = pd.read_csv('../../../model_translator/src/output/flight_0_best_sensors.csv')
training_flight1 = pd.read_csv('../../../model_translator/src/output/flight_0.out', header=None)

# Uncomment to read all the data from files
# TODO:  modify reading the .out files

# training_sensors1.set_index("Time", inplace=True)
# training_sensors2 = pd.read_csv('../../../model_translator/src/output/flight_1_best_sensors.csv')
# training_flight2 = pd.read_csv('../../../model_translator/src/output/flight_1.out', header=None)

# test_sensors = pd.read_csv('../../../model_translator/src/output/flight_0_best_sensors.csv')
# test_data = pd.read_csv('../../../model_translator/src/output/flight_2.out', header=None)

# Checking if the data is stationary and converting it if its not
# TODO: check all the data, work out how to interpret .out files

for col in training_sensors1.columns:
    if check_stationarity(training_sensors1[col]) == False:
        print ("{} is not stationary".format(col))
        convert_to_stationary(training_sensors1, col)
    else:
        print ("{} is stationary".format(col))
