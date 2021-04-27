# computational imports
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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
import os

# check version
from pycaret.utils import version
print(version())

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns


#read in the data
df = pd.read_csv('METABRIC_RNA_Mutation.csv', nrows=300, verbose=True, usecols=range(1,31))

#print the shape of the dataframe
print(f"The shape is {df.shape}")

#get the column info
print(df.info())
print(df.isna().sum())
df = df.dropna()
display(missing_values_table(df))
print(f"The shape is {df.shape}")

# https://stackoverflow.com/questions/65012601/attributeerror-simpleimputer-object-has-no-attribute-validate-data-in-pyca
# https://github.com/pycaret/pycaret/issues/1107

from pycaret.classification import *
clf1 = setup(df, target = 'overall_survival', imputation_type='iterative', session_id=123, log_experiment=True, experiment_name='exp1',fix_imbalance=True)
