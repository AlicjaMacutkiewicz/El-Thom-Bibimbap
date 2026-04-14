import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def convert_to_stationary(dataframe, col):
    # differencing the data to make the time series stationary
    
    print ("Differencing the column: {}".format(col))
    series = dataframe[col]
    diff1 = series.diff()
    
    if check_stationarity(diff1):
        print ("{} is now stationary".format(col))
        dataframe[col] = diff1
        return
    
    print ("{} is not stationary after the first differentiation".format(col))
    print ("Differencing {} again".format(col))
    diff2 = diff1.diff()

    if check_stationarity(diff2):
        print ("{} is now stationary".format(col))
        dataframe[col] = diff2
        return
    print ("{} is still not stationary after the second differentiation".format(col))
    dataframe[col] = diff2
    
    # TODO: for now the if the data is still not stationary after the
    # second transformation the code continues as if the time series
    # were stationary. Check whether there is a better way to ensure
    # stationarity without overdifferencing

def check_stationarity(data_series):
    data_series = data_series.dropna()
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
training_sensors1.set_index("Time", inplace=True)

# Uncomment to read all the data from files

# training_sensors2 = pd.read_csv('../../../model_translator/src/output/flight_1_best_sensors.csv')
# training_sensors2.set_index("Time", inplace=True)
# test_sensors = pd.read_csv('../../../model_translator/src/output/flight_0_best_sensors.csv')
# test_sensors.set_index("Time", inplace=True)

# Checking if the data is stationary and converting it if its not

for col in training_sensors1.columns:
    if check_stationarity(training_sensors1[col]) == False:
        print ("{} is not stationary".format(col))
        convert_to_stationary(training_sensors1, col)
    else:
        print ("{} is stationary".format(col))
