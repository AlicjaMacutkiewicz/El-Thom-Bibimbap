from statsmodels.tsa.vector_ar.vecm import VECM


def create_vecm(data):
    model = VECM(data, deterministic="ci", seasons=4)
    return model

def test_vecm(fitted_model, test_data, n):
    prediction = fitted_model.predict(n)
    test = test_data[:n]
    