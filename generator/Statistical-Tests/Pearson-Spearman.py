# is data linear correlated

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

columns = ['Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z', 'Best_AngVel_X',
       'Best_AngVel_Y', 'Best_AngVel_Z', 'Barometer_Value', 'Sensor_Value',
       'Thrust', 'Mass', 'Position_X', 'Position_Y', 'Position_Z',
       'Acceleration_X', 'Acceleration_Y', 'Acceleration_Z']

N = 12
matrix = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        df1 = pd.read_parquet(f'../src/output/flight_{i}.parquet')
        df2 = pd.read_parquet(f'../src/output/flight_{j}.parquet')

        print(i, " ", j)

        pearson_vals = []
        spearman_vals = []

        for col in columns:
            min_len = min(len(df1), len(df2))

            x = df1[col].values[:min_len]
            y = df2[col].values[:min_len]

            x = (x - x.mean()) / (x.std() + 1e-8)
            y = (y - y.mean()) / (y.std() + 1e-8)

            r_p, _ = stats.pearsonr(x, y)
            r_s, _ = stats.spearmanr(x, y)

            pearson_vals.append(r_p)
            spearman_vals.append(r_s)

        matrix[i, j] = (np.mean(pearson_vals) + np.mean(spearman_vals)) / 2

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=[f"F{i}" for i in range(N)],
    yticklabels=[f"F{i}" for i in range(N)]
)

plt.title("Podobieństwo lotów (Pearson/Spearman correlation)")
plt.show()