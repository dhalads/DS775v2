# computational imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import json
# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set_style("darkgrid")

data4 = pd.read_csv("./data/age_height.csv")

print(data4.describe())

def SS_func( coef, *args):
    b0 = coef[0]
    b1 = coef[1]
    x = args[0]
    y = args[1]
    yhat = b0 + b1 * x
    yhat = b0 + b1 * x
    SS = sum( (y - yhat)**2)
    return(SS) # here's the minus sign!

result = minimize(SS_func,[0,0],args=(data4['age'], data4['height']))
b0 = result.x[0]
b1 = result.x[1]
print(f"The minimum for sum of squares has intercept b0 = {b0:2.3f} and slope b1 = {b1:2.3f}")

# # Use to check loss function
# Y = data4['height']
# X = data4['age']
# X = sm.add_constant(X)
# model = sm.OLS(Y, X).fit()
# print(model.summary())
