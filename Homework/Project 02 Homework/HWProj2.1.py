# EXECUTE FIRST

# computational imports
from sklearn.model_selection import train_test_split
import warnings
from IPython.core.display import HTML
from IPython.display import display, IFrame
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
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
# display imports

# import notebook styling for tables and width etc.
response = urllib.request.urlopen(
    'https://raw.githubusercontent.com/DataScienceUWL/DS775v2/master/ds755.css')
HTML(response.read().decode("utf-8"))

# import warnings

# imports in first cell of notebook
# from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
print(diabetes.DESCR)

# import numpy as np
X = np.array(diabetes.data)
y = np.array(diabetes.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)


# Here is all the code in one cell with most of it wrapped into a function for reuse

def my_regression_results(model):
    score_test = model.score(X_test, y_test)
    print('Model r-squared score from test data: {:0.4f}'.format(score_test))

    y_pred = model.predict(X_test)
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 6))
    plt.plot(y_test, y_pred, 'k.')
    plt.xlabel('Test Values')
    plt.ylabel('Predicted Values')

    # from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print('Mean squared error on test data: {:0.2f}'.format(mse))
    print('Root mean squared error on test data: {:0.2f}'.format(rmse))

# define the grid




def getParams(x):
    params = {
        "n_estimators": [10, 100, 150],
        "max_features": [0.05, 1],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 10, 20],
        "bootstrap": [0, 1]
    }
    if x == 1:
        output = params
    elif x == 2:
        output = [{
            'name': 'n_estimators',
            'type': 'discrete',
            'domain': params.get("n_estimators")
        }, {
            'name': 'max_features',
            'type': 'discrete',
            'domain': params.get("max_features")
        }, {
            'name': 'min_samples_split',
            'type': 'discrete',
            'domain': params.get("min_samples_split")
        }, {
            'name': 'min_samples_leaf',
            'type': 'discrete',
            'domain': params.get("min_samples_leaf")
        }, {
            'name': 'bootstrap',
            'type': 'discrete',
            'domain': params.get("bootstrap")
        }]
    elif x == 3:
        print()
    else:
        print()
    return output


x = getParams(1)
print(x)
x = getParams(2)
print(x)

def get_num_combinations():
    params = getParams(1)
    num = 1
    for key in params.keys():
        num = num * len(params.get(key))
    return num

print("num comb=" + str(get_num_combinations()))

col_list = list(getParams(1).keys())
col_list.insert(0, "method")
col_list.append("opt score R^2")
col_list.append("mse")
col_list.append("rmse")
col_list.append("num_fits")
m_output = pd.DataFrame(columns=col_list)

def addOutput(method, x, model, fits):
    global m_output
    temp = {"method":method}
    temp.update(x)
    score_test = model.score(X_test, y_test)
    temp.update({"opt score R^2":score_test})
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    temp.update({"mse":mse})
    rmse = np.sqrt(mse)
    temp.update({"rmse":rmse})
    temp.update({"num_fits":fits})
    output = m_output.append(temp, ignore_index=True)
    m_output = output

# from sklearn.ensemble import RandomForestRegressor


rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_train, y_train)

# run GridSearchCV with our xgbr_model to find better hyperparameters
# from sklearn.model_selection import GridSearchCV


# setup the grid search
grid_search = GridSearchCV(rf_model,
                           param_grid=getParams(1),
                           cv=2,
                           verbose=1,
                           n_jobs=1,
                           return_train_score=True)

grid_search.fit(X_train, y_train)
my_regression_results(grid_search)
addOutput("grid", grid_search.best_params_, grid_search, 2*get_num_combinations())
display(m_output)

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform, randint

random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=getParams(1),
    random_state=8675309,
    n_iter=2,
    cv=2,
    verbose=1,
    n_jobs=1,
    return_train_score=True)

random_search.fit(X_train, y_train)
num = random_search.get_params().get("cv")*random_search.get_params().get("n_iter")
addOutput("random", random_search.best_params_, random_search, num)
display(m_output)

# unfold to see code
np.random.seed(8675309)  # seed courtesy of Tommy Tutone
# from GPyOpt.methods import BayesianOptimization
# from sklearn.model_selection import cross_val_score, KFold


# Optimization objective
def cv_score(hyp_parameters):
    hyp_parameters = hyp_parameters[0]
    rf_model = RandomForestRegressor(random_state=0,
                                     n_estimators=hyp_parameters[0],
                                     max_features=int(hyp_parameters[1]),
                                     min_samples_split=int(hyp_parameters[2]),
                                     min_samples_leaf=hyp_parameters[3],
                                     bootstrap=int(hyp_parameters[4])
                                     )

    scores = cross_val_score(rf_model,
                             X=X_train,
                             y=y_train,
                             cv=KFold(n_splits=5))
    return np.array(scores.mean())  # return average of 5-fold scores


optimizer = BayesianOptimization(f=cv_score,
                                 domain=getParams(2),
                                 model_type='GP',
                                 acquisition_type='EI',
                                 acquisition_jitter=0.05,
                                 exact_feval=True,
                                 maximize=True,
                                 verbosity=True)

optimizer.run_optimization(max_iter=20, verbosity=True)

best_hyp_set = {}
hp_bounds = getParams(2)
for i in range(len(hp_bounds)):
    if hp_bounds[i]['type'] == 'continuous':
        best_hyp_set[hp_bounds[i]['name']] = optimizer.x_opt[i]
    else:
        best_hyp_set[hp_bounds[i]['name']] = int(optimizer.x_opt[i])
# print(best_hyp_set)

bayopt_search = RandomForestRegressor(random_state=0,**best_hyp_set)
bayopt_search.fit(X_train,y_train)

# my_regression_results(bayopt_search)
addOutput("bayesian", best_hyp_set, bayopt_search, 20*5)
display(m_output)
# from tpot import TPOTRegressor

tpot_config = {
    'sklearn.ensemble.RandomForestRegressor': getParams(1)
}

tpot = TPOTRegressor(scoring='r2',
                     generations=2,
                     population_size=20,
                     verbosity=2,
                     config_dict=tpot_config,
                     cv=3,
                     template='Regressor',  # no stacked models
                     random_state=8675309)

tpot.fit(X_train, y_train)
tpot.export('tpot_XGBregressor.py')  # export the model
best_parm = {}
for key in getParams(1).keys():
    value = tpot.fitted_pipeline_.get_params().get("randomforestregressor__"+key)
    best_parm[key]= value
addOutput("tpot", best_parm,tpot, 2*3)
display(m_output)
# my_regression_results(tpot)

