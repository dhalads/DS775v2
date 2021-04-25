# EXECUTE FIRST

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
HTML(response.read().decode("utf-8"))
import os

#read in the data
df = pd.read_csv('data/tmdb_5000_movies.csv', encoding = "ISO-8859-1", usecols=("title", "runtime", "budget", "production_companies", "vote_count", "vote_average"))

# #print the shape of the dataframe
# print(f"The shape is {df.shape}")

# #get the column info
df.info()

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)

def printOut(x):
    print(f"m = {x[0]}")
    print(f"C = {x[1]}")
    df = x[2].sort_values("score", ascending = False)
    display(df[['title','vote_count', 'vote_average', 'score']].head(10), display_id=True)

def filter(df):
    df = df[df['runtime'] <= 200]
    df = df[(df['budget'] > 0) & (df['budget'] < 3000000)]
    return(df)

def approachA(df):
    vote_count80 = df['vote_count'].quantile(0.8)
    m = df[df['vote_count'] >= vote_count80].shape[0]
    C = df['vote_average'].mean()
    df['score'] = df.apply(weighted_rating, args=(m,C), axis=1)
    # df = df[df['runtime'] <= 200]
    # df = df[(df['budget'] > 0) & (df['budget'] < 3000000)]
    df = filter(df)
    return((m, C, df))

# printOut(approachA(df))
pass
# print(df[:1]['production_companies'])

# convert str to dic
# import ast
# x = ast.literal_eval("{'foo' : 'bar', 'hello' : 'world'}")
# type(x)
# import json
# x = json.loads("{'foo' : 'bar', 'hello' : 'world'}")
# type(x)