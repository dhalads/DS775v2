# EXECUTE FIRST

# computational imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

# helper function for nicely printing dollar amounts
def dollar_print(x):
    if x < 0:
        return(f"-${-x:,.2f}")
    else:
        return(f"${x:,.2f}")

weather = []
weather.append("clear")
for i in range(999):
    forecast = weather[i]
    if forecast == "clear":
        probrain = 0.2
    else :
        probrain = 0.6
    rand = np.random.uniform(low=0.0, high=1.0, size=1)
    if rand <= probrain:
        weather.append("rain")
    else:
        weather.append("clear")

print(f"Weather forecast for the next {len(weather)} is  {weather}")
print(f"Days rain are {weather.count('rain')} and days clear are {weather.count('clear')}")
