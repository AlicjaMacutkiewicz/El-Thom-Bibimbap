import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

def create_var(data):
    model = VAR(data)
    return model

def test_var(model, test_data):
    pass
