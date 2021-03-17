# computational imports
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from scipy.stats import uniform, randint
from GPyOpt.methods import BayesianOptimization
from tpot import TPOTRegressor
from pprint import pprint

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

# Do not change this cell for loading and preparing the data
import pandas as pd
# from pandas_ml import ConfusionMatrix
import numpy as np


X = pd.read_csv('./data/loans_subset.csv')

# split into predictors and target
# convert to numpy arrays for xgboost, OK for other models too
y = np.array(X['status_Bad']) # 1 for bad loan, 0 for good loan
X = np.array(X.drop(columns = ['status_Bad']))

# split into test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn.linear_model import LogisticRegression

# we do need to go higher than the default iterations for the solver to get convergence
# and the explicity declaration of the solver avoids a warning message, otherwise
# the parameters are defaults.
logreg_model = LogisticRegression(solver='lbfgs',max_iter=1000)

logreg_model.fit(X_train, y_train)

# Use score method to get accuracy of model
score = logreg_model.score(X_test, y_test) # this is accuracy
print(score)

def my_classifier_results(model):
    # Use score method to get accuracy of model
    score = model.score(X_test, y_test) # this is accuracy
    print('Model score from test data: {:0.4f}'.format(score))

    # obtaining the confusion matrix and making it look nice
    y_pred = model.predict(X_test)
    y_pred = [1 if x>0.5 else 0 for x in y_pred]
    m_confusion = confusion_matrix(y_test, y_pred, labels=[1,0])
    # Confusion_Matrix = ConfusionMatrix(y_test, y_pred)
    # Confusion_Matrix.print_stats()
    m_accuracy = (m_confusion[1][1]+m_confusion[0][0])/len(y_pred)
    print('Model accuracy from test data: {:0.4f}'.format(m_accuracy))
    m_sensitivity = m_confusion[1][1]/(m_confusion[1][1]+m_confusion[1][0]) #TP/(TP+FN)
    print('Model sensitivity from test data: {:0.4f}'.format(m_sensitivity))
    m_precision = m_confusion[1][1]/(m_confusion[1][1]+m_confusion[0][1]) #TP/(TP+FP)
    print('Model precision from test data: {:0.4f}'.format(m_precision))


    # must put true before predictions in confusion matrix function
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[1,0]),
        index=['true:bad', 'true:good'],
        columns=['pred:bad','pred:good']
    )
    display(cmtx)

# P2.1
# from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_train,y_train)

my_classifier_results(rf_model)

# define the grid
params = {
    "n_estimators": [10, 100,150],
    "max_features": [0.8, 1],
    "min_samples_split": [1, 3],
    "min_samples_leaf": [1, 3],
    "bootstrap": [True, False]
}

# setup the grid search
grid_search = GridSearchCV(xgbr_model,
                           param_grid=params,
                           cv=5,
                           verbose=1,
                           n_jobs=1,
                           return_train_score=True)

grid_search.fit(X_train, y_train)

grid_search.best_params_

my_regression_results(grid_search)

# from tpot import TPOTRegressor

# tpot_config = {
#     'RandomForestRegressor': {
#         'n_estimators': [100],
#         'max_depth': range(1, 11),
#         'learning_rate': np.append(np.array([.001,.01]),np.arange(0.05,1.05,.05)),
#         'subsample': np.arange(0.05, 1.01, 0.05),
#         'min_child_weight': range(1, 21),
#         'reg_alpha': np.arange(1.0,5.25,.25),
#         'reg_lambda': np.arange(1.0,5.25,.25),
#         'nthread': [1],
#         'objective': ['reg:squarederror']
#     }
# }

# tpot = TPOTRegressor(scoring = 'r2',
#                      generations=50,
#                      population_size=20,
#                      verbosity=2,
#                      config_dict=tpot_config,
#                      cv=3,
#                      template='Regressor', #no stacked models
#                      random_state=8675309)

# tpot.fit(X_train, y_train)
# tpot.export('tpot_XGBregressor.py') # export the model

# my_regression_results(tpot)
