from matplotlib import pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM
import numpy as np
from scipy.stats import chi2_contingency

def create_vecm(data, lag, r):
    model = VECM(data, k_ar_diff=lag, coint_rank=r)
    return model

def test_vecm(fitted_model, test_data, n):
    prediction = fitted_model.predict(steps=n)
    test = np.array(test_data[:n])

    types = test_data.columns

    for i in range(len(types)):
        prediction_i = prediction[:, i]
        test_i = test[:, i]

        #data = [test_i, prediction_i]
        #stat, p, dof, expected = chi2_contingency(data)

        print(types[i])
        mae = np.mean(np.abs(test_i - prediction_i))
        mse = np.mean((test_i - prediction_i) ** 2)
        mean = np.mean(prediction_i)
        sigma3 = 3 * np.sqrt(np.var(prediction_i))
        valid = 0
        for value in prediction_i:
            if abs(value - mean) <= sigma3:
                valid += 1

        print(f"3 sigma valid: {(valid / prediction_i.size) * 100}%")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")


        x = np.arange(1, prediction_i.size + 1)
        plt.plot(x, prediction_i, label="Prediction")
        plt.plot(x, test_i, label="Test")
        plt.title(types[i])
        plt.legend()
        plt.show()