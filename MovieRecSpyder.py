# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:39:05 2018

@author: Manisha
"""
import pandas as pd
import numpy as np
craw=pd.read_csv('tmdb_5000_credits.csv',low_memory=False)
craw
metadata=pd.read_csv('tmdb_5000_movies1.csv',low_memory=False)
metadata
C = metadata['vote_average'].mean()
print(C)
m = metadata['vote_count'].quantile(0.90)
print(m)
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)
p=q_movies[['title', 'vote_count', 'vote_average', 'score']].head(50)
metadata['overview'].head(2)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
metadata['overview'] = metadata['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(metadata['overview'])
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]
get_recommendations('Batman Begins')
craw['id'] = craw['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')
a=pd.merge(craw,metadata, on='id')
a.head(2)
from ast import literal_eval
features = ['crew', 'keywords','genres']
for feature in features:
    a[feature] = a[feature].apply(literal_eval)
    
def get_director(x):
 for i in x:
  if i['job'] == 'Director':
   return i['name']  
        
 return np.nan

def get_list(x):
      if isinstance(x, list):
          names = [i['name'] for i in x]
          if len(names) > 3:
              names = names[:3]
          return names
    
    
    
    
      return []
  
a['director'] = a['crew'].apply(get_director)
features = ['keywords', 'genres']
for feature in features:
    a[feature] = a[feature].apply(get_list)
a[['title_x', 'director', 'keywords','genres']].head(3)


def clean_data(x):
    if isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            return ''
        
features = ['director', 'keywords', 'genres']
for feature in features:
    a[feature] = a[feature].apply(clean_data)  
def create_soup(x):
    return ' '.join(x['keywords']) +  ' ' + ' ' + x['director'] + ' ' + ' ' + x['director'] + ' '.join(x['genres'])     
a['soup'] = a.apply(create_soup, axis=1)
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(a['soup'])
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)   
a = a.reset_index()
indices = pd.Series(a.index, index=a['title_x'])
get_recommendations('The Dark Knight', cosine_sim2)
Enter=input("Enter the name of the movie =  ")
print("10 Movies similiar to it are..", get_recommendations(Enter, cosine_sim2))



