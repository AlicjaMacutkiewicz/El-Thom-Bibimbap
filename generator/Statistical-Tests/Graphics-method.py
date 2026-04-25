# creates overlying graphs of each column

import matplotlib.pyplot as plt
import pandas as pd

columns = ['Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z', 'Best_AngVel_X',
       'Best_AngVel_Y', 'Best_AngVel_Z', 'Barometer_Value', 'Sensor_Value',
       'Thrust', 'Mass', 'Position_X', 'Position_Y', 'Position_Z',
       'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z']

N = 12
tests = []

for i in range(N):
    a = pd.read_parquet(f'../src/output/flight_{i}.parquet')
    tests.append(a)

for col in columns:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    for i in range(N):
        p = tests[i][col]
        ax.plot(p, alpha=(1 / N))

    ax.set_title(col)
    ax.legend()

    plt.show()