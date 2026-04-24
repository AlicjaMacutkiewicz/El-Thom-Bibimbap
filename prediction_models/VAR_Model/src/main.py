from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from var import *
from vecm import *


def read_sensor_data(directory, index='Time'):
    sensors = pd.read_csv(directory)
    sensors.set_index(index, inplace=True)
    return sensors

def check_final_diff_order(data, signif = 0.05, test_sample = 20000, max_diff_order = 2):
    # checking how many times each of the columns need 
    # to be differenced for them to become stationary
    # and choosing the biggest value 
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
        p_value = adfuller(series, autolag="AIC")[1]

        # differencing the column until it becomes stationary or
        # until the data is differenced maximum number of times
        while p_value > signif and current_diff_order < max_diff_order:
            series = series.diff().dropna()
            p_value = adfuller(series, autolag="AIC")[1]
            current_diff_order += 1

        # tracking the highest differencing order needed across all columns
        final_diff_order = max(final_diff_order, current_diff_order)
        
        if p_value > signif:
            print(f"Warning: Column {col} is still not stationary after {max_diff_order} differences.")
    return final_diff_order

def is_cointegrated(data, col_idx = 1, det_order = -1, max_lag = 20):
    # Calculating the number of lagged differences in the model (k_arr_diff)
    # TODO: when a better approach to calculate lag values is implemented
    #       we can use those lag values here instead of recalculating them
    lag_values = VAR(data).select_order(maxlags=max_lag)

    k_ar_diff = lag_values.selected_orders.get("aic")
    if k_ar_diff is None:
       k_ar_diff = 1

    # Performing the Johanson's Cointegration Test
    # det_order controls deterministic terms 
    # (here det_order = -1 so there are no deterministic terms)
    result = coint_johansen(data, det_order, k_ar_diff)

    # Extracting trace test statistics to measure
    # the strength of cointegration relationships
    trace = result.lr1
    
    # Extracting  critical values for significance level
    # chosen by col_idx (here we have 1 so it's 5%)
    # critical values are a boundary from which we see the output
    # as statistcally significant
    crit = result.cvt[:, col_idx]

    # Comparing test statistics with critical values
    # If trace statistic is greater than the critical value there is evidence of cointegration
    r = sum(trace > crit)
    return r > 0, lag_values, r

def choose_model(data, final_diff_order):
    # if the data is already stationary we use VAR
    # otherwise additional checks are needed 
    if final_diff_order > 0:
        # VAR will not work properly on cointegrated data, therefore
        # we need to do a cointegration test and use a different model
        # if the data is cointegrated
        is_c, lag_values, r = is_cointegrated(data)
        if is_c:
            print("The series is cointegrated. The VECM model will be used.")
            return "VECM", lag_values, r
        else:
            print("The series is not cointegrated. The VAR model will be used.")
            print(f"The series was differenced {final_diff_order} times.")
            return "VAR", lag_values, r
    else:
        print("The series was already stationary. The VAR model will be used.")
        return "VAR", None, None

def prepare_data(data, model, final_diff_order, frac):
    # preprocessing the data - if it isn't stationary, but it also isn't cointegrated
    # (ie if we use the VAR model for non-stationary data) the whole time series has to be
    # differenced the maximum amount of times needed for a single column to become stationary
    # for a VAR model with data that was already stationary and the VECM model no
    # additional preprocessing is needed

    new_data = data.copy()
    if (model=="VAR" and final_diff_order > 0):
        for _ in range(final_diff_order):
            new_data = new_data.diff()

        split = int(len(new_data) * frac)
        training_data = new_data.iloc[:split]
        test_data = new_data.iloc[split:]
        return training_data.dropna(), test_data.dropna()
    else:
        split = int(len(new_data) * frac)
        training_data = new_data.iloc[:split]
        test_data = new_data.iloc[split:]

        scaler = StandardScaler()
        scaled_training_data = pd.DataFrame(
            scaler.fit_transform(training_data),
            columns=training_data.columns,
            index=training_data.index
        )

        scaled_test_data = pd.DataFrame(
            scaler.fit_transform(test_data),
            columns=test_data.columns,
            index=test_data.index
        )

        return scaled_training_data, scaled_test_data

# iterates through all specified parameters and chooses the best model
# super expensive in resources
# todo check results on fast computer
def find_best_parameters_for_VECM_bruteforce(data, max_r, max_lag, current_result):
    best_aic = np.inf
    best_r = 1
    best_lag = 1
    best_result = current_result

    for r in range(1, max_r + 1):
        for lag in range(1, max_lag + 1):

            print(f"Testing r: {r}, lag: {lag}")

            try:
                model = VECM(data, k_ar_diff=lag, coint_rank=r)
                result = model.fit()

                if result.aic < best_aic:
                    best_aic = result.aic
                    best_lag = lag
                    best_r = r
                    best_result = result

            except Exception:
                continue

    return best_r, best_lag, best_result

def test_saved_model():
    training_sensors = read_sensor_data('../../../model_translator/src/output/flight_0_best_sensors.csv')[:50847]
    final_diff_order = check_final_diff_order(training_sensors)
    model_type, lag_values, r = choose_model(training_sensors, final_diff_order)
    training_data, test_data = prepare_data(training_sensors, model_type, final_diff_order, 0.9)
    with open("model.pkl", "rb") as f:
        result = pickle.load(f)

    test_vecm(result, test_data, n=500)

if __name__ == '__main__':
    # Reading the data from csv files
    # TODO: split the data into a testing and training sets and
    #       check the preferable dataset size for VAR models -
    #       if the data for one flight won't be enough, look into how
    #       can the data from more flights be merged witout messing up
    #       the time series

    print("start")

    training_sensors = read_sensor_data('../../../model_translator/src/output/flight_0_best_sensors.csv')[:50847]

    print("checking final difference")

    # calculating how many times the data needs to be differenced
    # in order for it to become stationary
    final_diff_order = check_final_diff_order(training_sensors)

    print("choosing model")

    # choosing the right model for the dataset (VAR or VECM)
    model_type, lag_values, r = choose_model(training_sensors, final_diff_order)

    print("preparing data")

    # todo frac to config
    training_data, test_data = prepare_data(training_sensors, model_type, final_diff_order, 0.9)

    # TODO: calculate lag order here

    if model_type=="VAR":
        model = create_var(training_data)

        if lag_values is None:
            lag_values = model.select_order(maxlags=20)

        result = model.fit()
        print(test_var(model, test_data))
    else:
        #lag_order = lag_values.selected_orders["aic"]
        #model = create_vecm(training_data, lag_order, r)

        #result = model.fit()

        #r, lag, result = find_best_parameters_for_VECM_bruteforce(data=training_data, max_r=6, max_lag=6, current_result=result)

        result = create_vecm(training_data, lag_values.aic, r)
        result = train_vecm(result)
        save_vecm(result, lag_values.aic, r)


    test_vecm(result, test_data, n=500)
