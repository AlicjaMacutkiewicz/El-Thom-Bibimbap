import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial import distance

columns = [
    'Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z',
    'Best_AngVel_X', 'Best_AngVel_Y', 'Best_AngVel_Z',
    'Thrust', 'Barometer_Value', 'Sensor_Value'
]

N = 10
matrix = np.zeros((N, N))

flights = []

for i in range(N):
    df = pd.read_parquet(f'../src/output/flight_{i}.parquet')
    flights.append(df[columns].mean().values)

flights = np.array(flights)

VI = np.linalg.inv(np.cov(flights.T))

for i in range(N):
    for j in range(N):
        d = distance.mahalanobis(flights[i], flights[j], VI)
        matrix[i, j] = d

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mahalanobis distance")
plt.show()