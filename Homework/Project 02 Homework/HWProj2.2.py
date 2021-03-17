# computational imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython.core.display import HTML
from IPython.display import display, IFrame
import urllib.request
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
from tpot import TPOTClassifier
from pprint import pprint

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# for reading files from urls
# display imports

# Do not change this cell for loading and preparing the data
# from pandas_ml import ConfusionMatrix


X = pd.read_csv('./data/loans_subset.csv')

# split into predictors and target
# convert to numpy arrays for xgboost, OK for other models too
y = np.array(X['status_Bad'])  # 1 for bad loan, 0 for good loan
X = np.array(X.drop(columns=['status_Bad']))

# split into test and training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0)

cv_setting = 5
pop_setting = 5
gen_setting = 5
def getParams(x):
    # test
    params = {
        "n_estimators": [10, 60, 100],
        "max_depth": [2, 3, 5],
        "min_child_weight": [1, 10, 20],
        "learning_rate": [0.001, 0.01, 0.1, 0.5],
        "subsample": [0.5, 1],
        "reg_lambda": [0,1,3],
        "reg_alpha": [0,3,4]
    }
    # dev
    params_dev = {
        "n_estimators": [100],
        "max_depth": [3],
        "min_child_weight": [1],
        "learning_rate": [0.1],
        "subsample": [1],
        "reg_lambda": [1],
        "reg_alpha": [0]
    }
    if x == 1:
        output = params
    elif x == 2:
        output = [{
            'name': 'n_estimators',
            'type': 'discrete',
            'domain': params.get("n_estimators")
        }, {
            'name': 'max_depth',
            'type': 'discrete',
            'domain': params.get("max_depth")
        }, {
            'name': 'min_child_weight',
            'type': 'discrete',
            'domain': params.get("min_child_weight")
        }, {
            'name': 'learning_rate',
            'type': 'discrete',
            'domain': params.get("learning_rate")
        }, {
            'name': 'subsample',
            'type': 'discrete',
            'domain': params.get("subsample")
        }, {
            'name': 'reg_lambda',
            'type': 'discrete',
            'domain': params.get("reg_lambda")
        }, {
            'name': 'reg_alpha',
            'type': 'discrete',
            'domain': params.get("reg_alpha")
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
col_list.append("score")
col_list.append("accuracy")
col_list.append("sensitivity")
col_list.append("precision")
col_list.append("specificity")
col_list.append("num_fits")
m_output = pd.DataFrame(columns=col_list)

def addOutput(method, x, model, fits):
    global m_output
    temp = {"method":method}
    temp.update(x)
    score_test = model.score(X_test, y_test)
    temp.update({"score":score_test})
    y_pred = model.predict(X_test)
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]
    m_confusion = confusion_matrix(y_test, y_pred, labels=[1, 0])
    m_accuracy = (m_confusion[1][1]+m_confusion[0][0])/len(y_pred)
    temp.update({"accuracy":m_accuracy})
    m_sensitivity = m_confusion[1][1] / (m_confusion[1][1]+m_confusion[1][0])  # TP/(TP+FN)
    temp.update({"sensitivity":m_sensitivity })
    m_precision = m_confusion[1][1] /(m_confusion[1][1]+m_confusion[0][1])  # TP/(TP+FP)
    temp.update({"precision":m_precision})
    m_specificity = m_confusion[0][0] /(m_confusion[0][0]+m_confusion[0][1])  # TP/(TP+FP)
    temp.update({"specificity":m_specificity})
    temp.update({"num_fits":fits})
    output = m_output.append(temp, ignore_index=True)
    m_output = output


def my_classifier_results2(model):
    # Use score method to get accuracy of model
    score = model.score(X_test, y_test)  # this is accuracy
    print(f"Model:{type(model)}")
    print('Model score from test data: {:0.4f}'.format(score))

    # obtaining the confusion matrix and making it look nice
    y_pred = model.predict(X_test)
    y_pred = [1 if x > 0.5 else 0 for x in y_pred]
    m_confusion = confusion_matrix(y_test, y_pred, labels=[1, 0])
    # Confusion_Matrix = ConfusionMatrix(y_test, y_pred)
    # Confusion_Matrix.print_stats()
    m_accuracy = (m_confusion[1][1]+m_confusion[0][0])/len(y_pred)
    print('Model accuracy from test data: {:0.4f}'.format(m_accuracy))
    m_sensitivity = m_confusion[1][1] / \
        (m_confusion[1][1]+m_confusion[1][0])  # TP/(TP+FN)
    print('Model sensitivity from test data: {:0.4f}'.format(
        m_sensitivity))
    m_precision = m_confusion[1][1] / \
        (m_confusion[1][1]+m_confusion[0][1])  # TP/(TP+FP)
    print('Model precision from test data: {:0.4f}'.format(
        m_precision))

    # must put true before predictions in confusion matrix function
    cmtx = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=[1, 0]),
        index=['true:bad', 'true:good'],
        columns=['pred:bad', 'pred:good']
    )
    display(cmtx)

# P2.1
# from sklearn.ensemble import RandomForestRegressor


xgbr_model = xgb.XGBClassifier(eval_metric="error")
xgbr_model.fit(X_train, y_train)

# my_classifier_results(xgbr_model)

# setup the grid search
grid_search = GridSearchCV(xgbr_model,
                           param_grid=getParams(1),
                           cv=cv_setting,
                           verbose=1,
                           n_jobs=1,
                           return_train_score=True)

grid_search.fit(X_train, y_train)
my_classifier_results2(grid_search)
addOutput("grid", grid_search.best_params_, grid_search, 2*get_num_combinations())
display(m_output)



random_search = RandomizedSearchCV(
    xgbr_model,
    param_distributions=getParams(1),
    random_state=8675309,
    n_iter=5,
    cv=cv_setting,
    verbose=1,
    n_jobs=1,
    return_train_score=True)

random_search.fit(X_train, y_train)

my_classifier_results2(random_search)

num = random_search.get_params().get("cv")*random_search.get_params().get("n_iter")
addOutput("random", random_search.best_params_, random_search, num)
display(m_output)


# Optimization objective


def cv_score(hyp_parameters):
    hyp_parameters = hyp_parameters[0]
    xgb_model = xgb.XGBClassifier(eval_metric="error",
                                  n_estimators=int(hyp_parameters[0]),
                                  max_depth=int(hyp_parameters[1]),
                                  min_child_weight=int(hyp_parameters[2]),
                                  learning_rate=hyp_parameters[3],
                                  subsample=hyp_parameters[4],
                                  reg_alpha=hyp_parameters[5],
                                  reg_lambda=hyp_parameters[6]
                                  )
    scores = cross_val_score(xgb_model,
                             X=X_train,
                             y=y_train,
                             cv=KFold(n_splits=cv_setting))
    return np.array(scores.mean())  # return average of 5-fold scores


hp_bounds = getParams(2)
optimizer = BayesianOptimization(f=cv_score,
                                 domain=hp_bounds,
                                 model_type='GP',
                                 acquisition_type='EI',
                                 acquisition_jitter=0.05,
                                 exact_feval=True,
                                 maximize=True,
                                 verbosity=True)

optimizer.run_optimization(max_iter=20, verbosity=True)

best_hyp_set = {}
for i in range(len(hp_bounds)):
    if hp_bounds[i]['type'] == 'continuous':
        best_hyp_set[hp_bounds[i]['name']] = optimizer.x_opt[i]
    else:
        best_hyp_set[hp_bounds[i]['name']] = int(optimizer.x_opt[i])
best_hyp_set

bayopt_search = xgb.XGBClassifier(eval_metric="error", **best_hyp_set)
bayopt_search.fit(X_train, y_train)

my_classifier_results2(bayopt_search)
addOutput("bayesian", best_hyp_set, bayopt_search, 20*5)
display(m_output)

tpot_params = getParams(1)
tpot_params.update({"eval_metric":["error"]}) # added to prevent getting warning about using logloss vs error
tpot_config = {
    'xgboost.XGBClassifier': tpot_params
}


tpot = TPOTClassifier(scoring='accuracy',
                      generations=gen_setting,
                      population_size=pop_setting,
                      verbosity=2,
                      config_dict=tpot_config,
                      cv=cv_setting,
                      template='Classifier',  # no stacked models
                      random_state=8675309)

tpot.fit(X_train, y_train)
tpot.export('tpot_XGBClassifier.py')  # export the model
best_parm = {}
for key in getParams(1).keys():
    value = tpot.fitted_pipeline_.get_params().get(tpot.op_list[0].lower()+"__"+key)
    best_parm[key]= value
my_classifier_results2(tpot)
addOutput("tpot", best_parm,tpot, 2*3)
display(m_output)


tpot=TPOTClassifier(scoring = 'accuracy',
            generations = gen_setting,
            population_size = pop_setting,
            verbosity = 2,
            cv = cv_setting,
            random_state = 8675309)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_optimal_pipeline.py')

my_classifier_results2(tpot)
addOutput("tpot general", {},tpot, 2*3)
display(m_output)
