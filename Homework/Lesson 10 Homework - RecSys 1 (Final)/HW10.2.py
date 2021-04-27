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
df = pd.read_csv('data/tmdb_5000_movies.csv', encoding = "ISO-8859-1", usecols=("title", "genres", "runtime", "budget", "production_companies", "vote_count", "vote_average"))

df['genres'] =  df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


def uniqueGenres(df):
    # #let's do it all in one step
    allGenres = ', '.join(df.apply(lambda x:pd.Series(x['genres']),axis=1).stack().unique())
    return(allGenres)


def build_chart(gen_df, percentile=0.8):

    #Ask for preferred genres
    print(uniqueGenres(df))
    print("Input preferred genre")
    genre = input()

    print("Input second preferred genre")
    genre2 = input()

    #Ask for lower limit of duration
    print("Input shortest duration")
    low_time = int(input())

    #Ask for upper limit of duration
    print("Input longest duration")
    high_time = int(input())

    #Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
    movies = gen_df.copy()

    f = lambda x: genre in x
    print(f(genre))

    #Filter based on the condition
    movies = movies[(movies['genres'].apply(lambda x: genre in x) | movies['genres'].apply(lambda x: genre2 in x)) & #updated filtering based on a list.
                    (movies['runtime'] >= low_time) &
                    (movies['runtime'] <= high_time)]

    #Compute the values of C and m for the filtered movies
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)

    # #Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
    # q_movies = movies.copy().loc[movies['vote_count'] >= m]

    #Calculate score using the IMDB formula
    movies['score'] = movies.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average'])
                                       + (m/(m+x['vote_count']) * C)
                                       ,axis=1)

    #Sort movies in descending order of their scores
    movies = movies.sort_values('score', ascending=False)

    return movies

out_movies = build_chart(df, .8)
# display(out_movies.head)
#display the final result with just name and scores
display(out_movies[['title','genres', 'runtime', 'vote_count', 'vote_average', 'score']].head(5))