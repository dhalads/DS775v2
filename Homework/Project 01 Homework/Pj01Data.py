

# Extra code to test OLS assumptions.

# Ordinary Least Squares(OLS) is a commonly used technique for linear regression analysis. OLS makes certain assumptions about the data like linearity, no multicollinearity, no autocorrelation, homoscedasticity, normal distribution of errors.(ref https://aiaspirant.com/ols-assumptions/)
    
# * linearity
# * no multicollinearity
# * errors are normally distributed
# * Autocorrelation
# * homoscedasticity



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import scipy
import seaborn as sns

import sysconfig
print(sysconfig.get_paths()["purelib"])


execfile("PJ01.py")

np.random.seed(444)
np.set_printoptions(precision=3)

# d = np.random.laplace(loc=15, scale=3, size=500)
para1 = 'COUPON'
para2 = 'FARE'
para = 'log' + para1
data = pd.read_csv("data/Airfares.csv")
data[para] = [math.log(x, 10) for x in data[para1]]

result = scipy.stats.pearsonr(data[para1], data[para2])
print(result)

plt.boxplot(data[['COUPON', "HI", "DISTANCE"]])

# test linearity
fig, ax = plt.subplots(1, 1)
sns.residplot(lm_dict[para2].predict(), data[para2], lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color':'red'}, ax=ax)
ax.title.set_text('Linearity Test Residuals vs Fitted')
ax.set(xlabel='Fitted', ylabel='Residuals')

# test collinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif_df = data[['COUPON','HI', 'DISTANCE']] #subset the dataframe
print(vif_df.head())
print(vif_df.shape[1])
vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(vif_df, i) for i in range(vif_df.shape[1])]
vif["VIF Factor"] = [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])]
vif["features"] = vif_df.columns
print(vif)

# test qq plot
fig, ax = plt.subplots(1, 1)
sm.ProbPlot(lm_dict[para2].resid).qqplot(line='s', color='#1f77b4', ax=ax)
ax.title.set_text('QQ Plot')

# test Autocorrelation
# One of the common tests for autocorrelation of residuals is the Durbin-Watson test.
# It ranges from 0 to 4. d value of 2 indicates that there is no autocorrelation. There is negative autocorrelation if the value of d is nearing 4 and positive correlation if the value is close to 0.
# The Durbin-Watson test is printed with the statsmodels summary.

# test for homoscedasticity
lr = lm_dict[para2]
fig, ax = plt.subplots(1, 1)
standardized_resid1 = np.sqrt(np.abs(lr.get_influence().resid_studentized_internal))
sns.regplot(lr.predict(), standardized_resid1, color='#1f77b4', lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color':'red'}, ax=ax)
ax.title.set_text('Homoscedasticity Scale Location')
ax.set(xlabel='Fitted', ylabel='Standardized Residuals')

# # An "interface" to matplotlib.axes.Axes.hist() method
# n, bins, patches = plt.hist(x=data[para], bins='auto', color='#0504aa')
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('My Very Own Histogram')
# # plt.text(23, 45, r'$\mu=15, b=3$')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

# fig = sm.qqplot(data[para], line='45')
# plt.show()

from scipy.stats import shapiro

#perform Shapiro-Wilk test
# shapiro(data[para])

# Seaborn visualization library
# import seaborn as sns# Create the default pairplot
sns.pairplot(data, vars = ['FARE', 'COUPON', 'HI', 'DISTANCE'])
sns.pairplot(data, vars = ['PAX', 'COUPON', 'HI', 'DISTANCE'])
sns.pairplot(data, vars = ['S_INCOME', 'COUPON', 'HI', 'DISTANCE'])
sns.pairplot(data, vars = ['E_INCOME', 'COUPON', 'HI', 'DISTANCE'])


# sns.pairplot(df[df['year'] >= 2000], 
#              vars = ['life_exp', 'log_pop', 'log_gdp_per_cap'], 
#              hue = 'continent', diag_kind = 'kde', 
#              plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
#              size = 4);# Title 
# plt.suptitle('Pair Plot of Socioeconomic Data for 2000-2007', 
#              size = 28);
plt.show()
