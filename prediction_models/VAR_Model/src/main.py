import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller

from var import create_var
from vecm import create_vecm

def is_cointegrated(df, alpha, det_order = -1,  k_ar_diff = 5):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,3)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


    return False

def check_final_diff_order(data, signif, test_sample = 20000, max_diff_order = 2):
    final_diff_order=0
    for col in data.columns:
        # pereparing the data for the ADF test - using only a small part of our data
        # since the test will not work for datasets as large as those generated in our project
        series = data[col].dropna().iloc[:test_sample]

        # ensuring there are no constant columns - otherwise adfuller will crash
        if series.std() == 0:
            print(f"Skipping constant column {col}.")
            continue
        
        current_diff_order = 0
        # testing stationarity using the ADF test 
        # the column is stationary if the  p value calculated in the test
        # and stored in results[1] is less or equal 0.05 (signif)
        p_value = adfuller(series)[1]

        # differencing the column until it becomes stationary or
        # until the data is differenced maximum number of times
        while p_value > signif or current_diff_order < max_diff_order:
            series = series.diff().dropna()
            p_value = adfuller(series)[1]
            current_diff_order += 1

        # tracking the highest differencing order needed across all columns
        final_diff_order = max(final_diff_order, current_diff_order)
        
        if p_value > signif:
            print(f"Warning: Column {col} is still not stationary after {max_diff_order} differences.")
    return final_diff_order

def prepare_data(data, signif=0.05):
    # checking how many times each of the columns need 
    # to be differenced for them to become stationary
    # and choosing the biggest value 

    final_diff_order = check_final_diff_order(data, signif);
    
    if final_diff_order > 0:
        if is_cointegrated(data, signif):
            print(f"The series is cointegrated. The VECM model will be used.")
            return data.diff(final_diff_order).dropna(), "VECM"
        else: 
            print(f"The series is not cointegrated. The VAR model will be used.")
            print(f"The series is now stationary. It was differenced {final_diff_order} times.")
            return data.diff(final_diff_order).dropna(), "VAR"
    else:
        print("The series was already stationary. The VAR model will be used.")       
        return data.copy(), 0

def read_sensor_data(directory, index='Time'):
    sensors = pd.read_csv(directory)
    sensors[index] = pd.to_datetime(sensors[index])
    sensors.set_index(index, inplace=True)
    return sensors

# Splitting the data into training and testing datasets:
# The data from first two flights is the training data
# The data from the third flight is the testing data

training_sensors1 = read_sensor_data('../../../model_translator/src/output/flight_0_best_sensors.csv')
training_sensors2 = read_sensor_data('../../../model_translator/src/output/flight_1_best_sensors.csv')
test_sensors = read_sensor_data('../../../model_translator/src/output/flight_0_best_sensors.csv')

stationary_df, model = prepare_data(training_sensors1)
#stationary_df, model = prepare_data(training_sensors2)
#test_sensors = prepare_data(test_sensors)
if model=="VAR":
    model = create_var()
else:
    model = create_vecm()
