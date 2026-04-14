import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def convert_to_stationary(data, signif=0.05, max_diff_order = 2, test_sample = 20000):
    final_diff_order = 0
    
    # checking how many times each of the columns need 
    # to be differenced for them to become stationary
    for col in data.columns:
        # performing the test on a small part of the dataset
        # (ADF test will not work for datasets as large as those generated in our project)
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
        while p_value > signif and current_diff_order < max_diff_order:
            series = series.diff().dropna()
            p_value = adfuller(series)[1]
            current_diff_order += 1

        # tracking the highest differencing order needed across all columns
        final_diff_order = max(final_diff_order, current_diff_order)
        
        if p_value > signif:
            print(f"Warning: Column {col} is still not stationary after {max_diff_order} differences.")
    
    if final_diff_order > 0:
        print(f"The series is now stationary. It was differenced {final_diff_order} times.")
        stationary_data = data.diff(final_diff_order).dropna()
    else:
        print("The series was already stationary.")
        stationary_data = data.copy()
    return stationary_data


# Splitting the data into training and testing datasets:
# The data from first two flights is the training data
# The data from the third flight is the testing data

training_sensors1 = pd.read_csv('../../../model_translator/src/output/flight_0_best_sensors.csv')
training_sensors1['Time'] = pd.to_datetime(training_sensors1['Time'])
training_sensors1.set_index("Time", inplace=True)

# Uncomment to read all the data from files

# training_sensors2 = pd.read_csv('../../../model_translator/src/output/flight_1_best_sensors.csv')
# training_sensors2.set_index("Time", inplace=True)
# test_sensors = pd.read_csv('../../../model_translator/src/output/flight_0_best_sensors.csv')
# test_sensors.set_index("Time", inplace=True)

data_slice1 = training_sensors1.index[20000]
training_sensors1.loc[:data_slice1].plot(subplots=True)

stationary_df = convert_to_stationary(training_sensors1)
data_slice2 = stationary_df.index[20000]
stationary_df.loc[:data_slice2].plot(subplots=True)
plt.show()
