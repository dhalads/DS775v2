# EXECUTE FIRST

# computational imports
import numpy as np
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
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

# import notebook styling for tables and width etc.
response = urllib.request.urlopen('https://raw.githubusercontent.com/DataScienceUWL/DS775v2/master/ds755.css')
HTML(response.read().decode("utf-8"));

# import warnings
import warnings

# imports in first cell of notebook
# from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
print(diabetes.DESCR)

# import numpy as np
X = np.array(diabetes.data)
y = np.array(diabetes.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Here is all the code in one cell with most of it wrapped into a function for reuse

# from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(X_train,y_train) # this could be inside the function below too

def my_regression_results(model):
    score_test = model.score(X_test,y_test)
    print('Model r-squared score from test data: {:0.4f}'.format(score_test))

    y_pred = model.predict(X_test)
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(9,6))
    plt.plot(y_test,y_pred,'k.')
    plt.xlabel('Test Values')
    plt.ylabel('Predicted Values');

    # from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('Mean squared error on test data: {:0.2f}'.format(mse))
    print('Root mean squared error on test data: {:0.2f}'.format(rmse))

my_regression_results(model_lr)

# from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_train,y_train)

my_regression_results(rf_model)

# import xgboost as xgb

xgbr_model = xgb.XGBRegressor(objective ='reg:squarederror')
xgbr_model.fit(X_train,y_train)

my_regression_results(xgbr_model)

# from sklearn.model_selection import cross_val_score, KFold
scores = cross_val_score(xgbr_model, X=X_train, y=y_train, cv = 3)
print(f"The average score across the folds is {scores.mean():.4f}")

# from pprint import pprint
pprint(xgbr_model.get_xgb_params())

# run GridSearchCV with our xgbr_model to find better hyperparameters
# from sklearn.model_selection import GridSearchCV

# define the grid
params = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [2, 4, 6],
    "n_estimators": [10, 100,150],
    "subsample": [0.8, 1],
    "min_child_weight": [1, 3],
    "reg_lambda": [1, 3],
    "reg_alpha": [1, 3]
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


# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform, randint

params = {
    "learning_rate": [0.001, 0.01, 0.1, 0.5, 1.],
    "max_depth": randint(1, 10),
    "n_estimators": randint(10, 150),
    "subsample": uniform(0.05, 0.95),  # so uniform on [.05,.05+.95] = [.05,1.]
    "min_child_weight": randint(1, 20),
    "reg_alpha": uniform(0, 5),
    "reg_lambda": uniform(0, 5)
}

random_search = RandomizedSearchCV(
    xgbr_model,
    param_distributions=params,
    random_state=8675309,
    n_iter=25,
    cv=5,
    verbose=1,
    n_jobs=1,
    return_train_score=True)

random_search.fit(X_train, y_train)

random_search.best_params_

my_regression_results(random_search)

#unfold to see code
np.random.seed(8675309)  # seed courtesy of Tommy Tutone
# from GPyOpt.methods import BayesianOptimization
# from sklearn.model_selection import cross_val_score, KFold

hp_bounds = [{
    'name': 'learning_rate',
    'type': 'continuous',
    'domain': (0.001, 1.0)
}, {
    'name': 'max_depth',
    'type': 'discrete',
    'domain': (1, 10)
}, {
    'name': 'n_estimators',
    'type': 'discrete',
    'domain': (10, 150)
}, {
    'name': 'subsample',
    'type': 'continuous',
    'domain': (0.05, 1.0)
}, {
    'name': 'min_child_weight',
    'type': 'discrete',
    'domain': (1, 20)
}, {
    'name': 'reg_alpha',
    'type': 'continuous',
    'domain': (0, 5)
}, {
    'name': 'reg_lambda',
    'type': 'continuous',
    'domain': (0, 5)
}]


# Optimization objective
def cv_score(hyp_parameters):
    hyp_parameters = hyp_parameters[0]
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                 learning_rate=hyp_parameters[0],
                                 max_depth=int(hyp_parameters[1]),
                                 n_estimators=int(hyp_parameters[2]),
                                 subsample=hyp_parameters[3],
                                 min_child_weight=int(hyp_parameters[4]),
                                 reg_alpha=hyp_parameters[5],
                                 reg_lambda=hyp_parameters[6])
    scores = cross_val_score(xgb_model,
                             X=X_train,
                             y=y_train,
                             cv=KFold(n_splits=5))
    return np.array(scores.mean())  # return average of 5-fold scores


optimizer = BayesianOptimization(f=cv_score,
                                 domain=hp_bounds,
                                 model_type='GP',
                                 acquisition_type='EI',
                                 acquisition_jitter=0.05,
                                 exact_feval=True,
                                 maximize=True,
                                 verbosity=True)

optimizer.run_optimization(max_iter=20,verbosity=True)

best_hyp_set = {}
for i in range(len(hp_bounds)):
    if hp_bounds[i]['type'] == 'continuous':
        best_hyp_set[hp_bounds[i]['name']] = optimizer.x_opt[i]
    else:
        best_hyp_set[hp_bounds[i]['name']] = int(optimizer.x_opt[i])
best_hyp_set

bayopt_search = xgb.XGBRegressor(objective='reg:squarederror',**best_hyp_set)
bayopt_search.fit(X_train,y_train)

my_regression_results(bayopt_search)

# from tpot import TPOTRegressor

tpot_config = {
    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': np.append(np.array([.001,.01]),np.arange(0.05,1.05,.05)),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'reg_alpha': np.arange(1.0,5.25,.25),
        'reg_lambda': np.arange(1.0,5.25,.25),
        'nthread': [1],
        'objective': ['reg:squarederror']
    }
}

tpot = TPOTRegressor(scoring = 'r2',
                     generations=50,
                     population_size=20,
                     verbosity=2,
                     config_dict=tpot_config,
                     cv=3,
                     template='Regressor', #no stacked models
                     random_state=8675309)

tpot.fit(X_train, y_train)
tpot.export('tpot_XGBregressor.py') # export the model

my_regression_results(tpot)

lr = np.array([.001,.01])
print(lr)

np.append(np.array([.001,.01]),np.arange(0.05,1.05,.05))

# from tpot import TPOTRegressor

tpot = TPOTRegressor(scoring = 'r2',
                     generations=10,
                     population_size=40,
                     verbosity=2,
                     cv=5,
                     random_state=8675309)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_optimal_pipeline.py')

my_regression_results(tpot)