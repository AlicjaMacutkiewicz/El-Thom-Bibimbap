# Znajduje anomalie. nie mam pojecia jak dizala ale brzmi cool
# chyba znajduje loty z najwieksza iloscia anomalii

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

columns = ['Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z', 'Best_AngVel_X',
           'Best_AngVel_Y', 'Best_AngVel_Z', 'Barometer_Value', 'Sensor_Value',
           'Thrust', 'Mass', 'Position_X', 'Position_Y', 'Position_Z',
           'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z']

N = 12
WINDOW_SIZE = 50
N_COMPONENTS = 10

flights = []
for i in range(N):
    df = pd.read_parquet(f'../src/output/flight_{i}.parquet')
    flights.append(df[columns].values)

def create_windows(X, window_size):
    return np.array([
        X[i:i+window_size].flatten()
        for i in range(len(X) - window_size)
    ])

scaler = StandardScaler()
scaler.fit(flights[0])
flights_scaled = [scaler.transform(f) for f in flights]

flights_windows = [create_windows(f, WINDOW_SIZE) for f in flights_scaled]

pca = PCA(n_components=N_COMPONENTS)
pca.fit(flights_windows[0])

errors = []

print("errors")
for i in range(N):
    print(i)
    X_w = flights_windows[i]

    X_proj = pca.inverse_transform(pca.transform(X_w))
    err = np.mean((X_w - X_proj) ** 2, axis=1)

    errors.append(err)

matrix = np.zeros((N, N))

print()
for i in range(N):
    for j in range(N):
        print(i, " ", j)

        e_i = np.percentile(errors[i], 95)
        e_j = np.percentile(errors[j], 95)

        matrix[i, j] = np.abs(e_i - e_j)

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("PCA Reconstruction Error (95th percentile)")
plt.xlabel("Flight")
plt.ylabel("Flight")
plt.show()