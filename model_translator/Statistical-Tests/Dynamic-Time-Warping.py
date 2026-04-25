#todo do
import matplotlib.pyplot as plt
import pandas as pd

columns = [
    'Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z',
    'Best_AngVel_X', 'Best_AngVel_Y', 'Best_AngVel_Z',
    'Thrust', 'Barometer_Value', 'Sensor_Value'
]

N = 10
tests = []

for i in range(N):
    tests.append(pd.read_parquet(f'../src/output/flight_{i}.parquet'))

values = []
for col in columns:

    fig, plots = plt.subplots(1, N)
    fig.set_dpi(1000)

    for i in range(N):
        p = tests[i][col]
        plots[i].plot(p)


    fig.suptitle(col)
    fig.show()