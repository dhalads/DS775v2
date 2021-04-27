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
# print(df[:1]['production_companies'])
# df[:1]['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# convert str to dic
# import ast
# x = ast.literal_eval("{'foo' : 'bar', 'hello' : 'world'}")
# type(x)
# import json
# x = json.loads("{'foo' : 'bar', 'hello' : 'world'}")
# type(x)
df['production_companies'] =  df['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# #print the shape of the dataframe
# print(f"The shape is {df.shape}")

# #get the column info
df.info()

#Convert all NaN into stringified empty lists and apply literal eval and convert to list by chaining functions
# df['production_companies'] = df['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

def uniqueCompanies(df):
    #in steps
    #convert the genres list in a series of columns
    step1 = df.apply(lambda x:pd.Series(x['production_companies']),axis=1)
    print(f"Step 1\n{step1}")

    #this step converts the rows into columns and "stacks" them all together
    step2 = step1.stack()
    print(f"Step 2\n{step2}")

    #let's get just the unique values from this series
    step3 = step2.unique()
    print(f"Step 3\n{step3}")
    print(f"Step 3 is a \n{type(step3)}")

    #numpy arrays can be joined just like lists, so let's join it to create a comma-delimited string
    step4 = ', '.join(step3)
    print(f"Step 4\n{step4}")


    # #let's do it all in one step
    # allGenres = ', '.join(snip.apply(lambda x:pd.Series(x['genres']),axis=1).stack().unique())
    # allGenres

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)

def printOut(label, x):
    print(f"approach {label}")
    print(f"m = {x[0]}")
    print(f"C = {x[1]}")
    df = x[2].sort_values("score", ascending = False)
    num_movies = df.shape[0]
    print(f"Number movies is {num_movies}")
    display(df[['title','vote_count', 'vote_average', 'score']].head(10), display_id=True)

def filter(df):
    df = df[df['runtime'] <= 200]
    df = df[(df['budget'] > 0) & (df['budget'] < 3000000)]
    selected1 = 'Universal Pictures'
    selected2 = ' Warner Bros.'
    df = df[( ( df['production_companies'].apply(lambda x: selected1 not in x) & df['production_companies'].apply(lambda x: selected2 not in x) ) )]
    return(df)

def approachA(df):
    vote_count80 = df['vote_count'].quantile(0.8)
    m = df[df['vote_count'] >= vote_count80].shape[0]
    C = df['vote_average'].mean()
    df['score'] = df.apply(weighted_rating, args=(m,C), axis=1)
    df = filter(df)
    return((m, C, df))

def approachB(df):
    df = filter(df)
    vote_count80 = df['vote_count'].quantile(0.8)
    m = df[df['vote_count'] >= vote_count80].shape[0]
    C = df['vote_average'].mean()
    df['score'] = df.apply(weighted_rating, args=(m,C), axis=1)
    return((m, C, df))

def approachC(df):
    vote_count80 = df['vote_count'].quantile(0.8)
    df = df[df['vote_count'] >= vote_count80]
    df = filter(df)
    vote_count80 = df['vote_count'].quantile(0.8)
    m = df[df['vote_count'] >= vote_count80].shape[0]
    C = df['vote_average'].mean()
    df['score'] = df.apply(weighted_rating, args=(m,C), axis=1)
    return((m, C, df))

# uniqueCompanies(df)
print(id(df))
print(df.shape)
printOut('A', approachA(df.copy()))
print(id(df))
print(df.shape)
printOut('B', approachB(df.copy()))
print(id(df))
print(df.shape)
printOut('C', approachC(df.copy()))
print(id(df))
print(df.shape)

