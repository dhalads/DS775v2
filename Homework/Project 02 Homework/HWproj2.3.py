#For Regression
# from pycaret.regression import *#For Classification
from pycaret.classification import *#For Clustering
# from pycaret.clustering import *#For Anomaly Detection
# from pycaret.anomaly import *#For NLP
# from pycaret.nlp import *#For association rule mining
# from pycaret.arules import *

X = pd.read_csv('./data/loans_subset.csv')


from pycaret.datasets import get_data
dataset = get_data('credit')
dataset.shape
data = dataset.sample(frac=0.95, random_state=786) #random_state=786
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
exp_clf101 = setup(data = data, target = 'default', session_id=123)



# exp_clf102 = setup(data = data, target = 'default', session_id=123,
#                   normalize = True,
#                   transformation = True,
#                   ignore_low_variance = True,
#                   remove_multicollinearity = True, multicollinearity_threshold = 0.95,
#                   bin_numeric_features = ['LIMIT_BAL', 'AGE'],
#                   group_features = [['BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'],
#                                    ['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']],
#                   log_experiment = True, experiment_name = 'credit1')


best_model = compare_models()
print(best_model)
# reg = setup(data = X_train, target =  "ddd")

# create_model('model_ID')

# create_model('model_ID',fold = n)

# compare_models()

# compare_models(n_select = n)

# compare_models(n_select = n, sort ‘ AUC’)

# dt = create_model('dt') #dt stands for the Decision Tree

# tuned = tune_model(dt, n_iter = 50)

# model = create_model('Model_name')
# plot_model(model)

# model = create_model('Model_name')
# interpret_model(model)

# model = create_model('Model_name')
# finalize_model(model)

# model = create_model('Model_name')final_model = finalize_model(model)
# deploy_model(final_model, model_name = 'Model_name_aws', platform = 'aws',
#              authentication = {'bucket' : 'pycaret-test'})

# predictions = predict_model(model_name = 'lr_aws', data = data_unseen, platform = 'aws', authentication = { 'bucket' : 'pycaret-test' })