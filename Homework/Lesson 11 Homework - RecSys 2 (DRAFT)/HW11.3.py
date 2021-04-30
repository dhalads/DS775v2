# computational imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic, SVD

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer

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
HTML(response.read().decode("utf-8"))

import os
from surprise.model_selection import cross_validate

#read in the data
df = pd.read_csv('data/restaurant/rating_final.csv')
df['rating'] = df['rating'] + df['food_rating'] + df['service_rating'] + 1
df = df.drop(columns=["food_rating", "service_rating"])
df.info()
print(df.describe())
print(df.shape)
print(df.head())

#The Reader object helps in parsing the file or dataframe containing ratings
reader = Reader(rating_scale=(1,7)) # defaults to (0,5)

#Create the dataset to be used for building the filter
data = Dataset.load_from_df(df, reader)
#List that will hold the sum of square values for different cluster sizes
ss = []
nums = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]
#define a random seed for consistent results
np.random.seed(1)
for i in nums:
    #Define the algorithm object; in this case kNN
    knn = KNNBasic(k=i, verbose=False)
    #This code cross validates (evaluates) the model
    knn_cv = cross_validate(knn, data, measures=['RMSE'], cv=5, verbose=False)
    knn_RMSE = np.mean(knn_cv['test_rmse'])
    ss.append(knn_RMSE)
 #Plot the Elbow Plot of SS v/s K
sns.pointplot(x=[j for j in nums], y=ss)

