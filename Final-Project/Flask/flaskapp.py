from flask import Flask, render_template, request
import csv 
import sys
import re
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from flask import jsonify
#from flask_restful import Resource, Api
import dill as pickle
from surprise import Reader, Dataset, SVD, evaluate
cos = pd.read_csv("/var/www/FlaskApp/FlaskApp/cosine_sim.csv")
cosine_sim = cos.values
movies_metadata = pd.read_csv("/var/www/FlaskApp/FlaskApp/new_metadata.csv")
movies_metadata = movies_metadata.drop("level_0",axis = 1)
links = pd.read_csv('/var/www/FlaskApp/FlaskApp/links.csv')
svd = SVD()
reader = Reader()
ratings = pd.read_csv('/var/www/FlaskApp/FlaskApp/ratings.csv')



movies_metadata = movies_metadata.reset_index()
titles = movies_metadata['title']
indexes = pd.Series(movies_metadata.index, index=movies_metadata['title'])

no_of_votes = movies_metadata[movies_metadata['vote_count'].notnull()]['vote_count'].astype('int')
vote_mean = movies_metadata[movies_metadata['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_mean.mean()
m = no_of_votes.quantile(0.95)

links.drop('imdbId',axis=1,inplace=True)
links.columns=['movieId', 'id']
id_map = links.merge(movies_metadata[['title', 'id']], on='id').set_index('title')

indices_map = id_map.set_index('id')

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=2)

evaluate(svd, data, measures=['RMSE', 'MAE'])

popularitybased = pickle.load(open("/var/www/FlaskApp/FlaskApp/popularitybased.pkl", "rb" ))
hybrid = pickle.load(open("/var/www/FlaskApp/FlaskApp/hybrid.pkl", "rb" ))
contentbased = pickle.load(open("/var/www/FlaskApp/FlaskApp/contentbased.pkl", "rb" ))
df = pd.read_csv("/var/www/FlaskApp/FlaskApp/editedmovies.csv")

print("All models Loaded")

app = Flask(__name__)
CORS(app)

dict1 = {k: g["title"].tolist() for k,g in df.groupby("genres")}



## main function ## 
@app.route('/movies',methods=['POST'])
def getmovies():
     content=request.json
     genre = content['genre']
     #fetch the movies for a given genre
     list1 = getMoviesForGenre(genre)
     return packageAndSend(list1)

@app.route('/recommend',methods=['POST'])
def getRecommendations():
     content=request.json
     movie = content['movie1']
     user = content['user']
     print(user)
     print(movie)
     movie1, b = movie.split("(")
     movie1 = movie1.strip()
     print(movie1 )
     #fetch the movies for a given genre
     dict  = getSimilarMovies3(movie1,user)
     return jsonify(dict)

def getMoviesForGenre(genre):
     list1 = dict1.get(genre)
     return list1

def packageAndSend(movieList):
     modelsandvalues = []
     for x in movieList:
            modelsandvalues.append({'id':x,'name':x})
     return jsonify(modelsandvalues)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def getSimilarMovies3(movie,user): 

     listForpopularitybased  = popularitybased(movie).values.tolist()
     listForhybrid = hybrid(user,movie).values.tolist()
     listForcontentbased  = contentbased(movie).values.tolist()

     print(listForpopularitybased)
     print(listForhybrid)
     print(listForcontentbased)
     dict = {'popularity':listForpopularitybased  ,'hybrid':listForhybrid ,'content':listForcontentbased  }

     return dict


@app.route('/', methods=['GET','POST'])
def movieRecommendation():
     return  render_template('movieRecommendation.html', option_list = dict1.keys())

if __name__ == '__main__':
   app.run(debug = True)