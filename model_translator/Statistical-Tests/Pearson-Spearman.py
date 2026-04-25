# todo cos tu nie gra


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

columns = [
    'Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z',
    'Best_AngVel_X', 'Best_AngVel_Y', 'Best_AngVel_Z',
    'Thrust', 'Barometer_Value', 'Sensor_Value'
]

N = 10
matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        df1 = pd.read_parquet(f'../src/output/flight_{i}.parquet')
        df2 = pd.read_parquet(f'../src/output/flight_{j}.parquet')

        print(i, " ", j)

        pearson_vals = []
        spearman_vals = []

        for col in columns:
            r_p, _ = stats.pearsonr(df1[col], df2[col])
            r_s, _ = stats.spearmanr(df1[col], df2[col])

            pearson_vals.append(r_p)
            spearman_vals.append(r_s)

        #matrix[i, j] = (np.mean(pearson_vals) + np.mean(spearman_vals)) / 2
        matrix[i, j] = (np.mean(spearman_vals) + np.mean(spearman_vals)) / 2

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=[f"F{i}" for i in range(N)],
    yticklabels=[f"F{i}" for i in range(N)]
)

plt.title("Podobieństwo lotów (Pearson-Spearman p-value)")
plt.show()