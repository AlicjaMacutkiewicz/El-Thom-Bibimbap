# todo cos tu nie gra


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import rel_entr

columns = [
    'Best_Acc_X', 'Best_Acc_Y', 'Best_Acc_Z',
    'Best_AngVel_X', 'Best_AngVel_Y', 'Best_AngVel_Z',
    'Thrust', 'Barometer_Value', 'Sensor_Value'
]

N = 10
bins = 30

matrix = np.zeros((N, N))

def get_hist(x):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist + 1e-10
    return hist / hist.sum()

for i in range(N):
    for j in range(N):

        print(i, " ", j)

        df1 = pd.read_parquet(f'../src/output/flight_{i}.parquet')
        df2 = pd.read_parquet(f'../src/output/flight_{j}.parquet')

        kl_vals = []

        for col in columns:
            p = get_hist(df1[col].values)
            q = get_hist(df2[col].values)

            kl = np.sum(rel_entr(p, q))
            kl_vals.append(kl)

        matrix[i, j] = np.mean(kl_vals)

plt.figure(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=[f"F{i}" for i in range(N)],
    yticklabels=[f"F{i}" for i in range(N)]
)

plt.title("KL divergence między lotami")
plt.show()