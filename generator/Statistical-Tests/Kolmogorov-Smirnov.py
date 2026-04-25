# is data linear correlated

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

columns = ['Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z', 'Best_AngVel_X',
           'Best_AngVel_Y', 'Best_AngVel_Z', 'Barometer_Value', 'Sensor_Value',
           'Thrust', 'Mass', 'Position_X', 'Position_Y', 'Position_Z',
           'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z']

N = 12
matrix = np.zeros((N, N))

flights = []
for i in range(N):
    df = pd.read_parquet(f'../src/output/flight_{i}.parquet')
    flights.append(df)

for i in range(N):
    for j in range(N):

        print(i, " ", j)

        ks_vals = []

        for col in columns:
            x = flights[i][col].values
            y = flights[j][col].values

            stat, _ = stats.ks_2samp(x, y)
            ks_vals.append(stat)

        matrix[i, j] = np.mean(ks_vals)

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("KS statistic (mean over features)")
plt.show()