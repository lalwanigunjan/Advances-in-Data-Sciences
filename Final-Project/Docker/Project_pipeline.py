#Import Necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import os
import sys
import boto3
import io
import zipfile
import requests
# Libraries required for NLP
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet,stopwords
import string
from requests import get

import dill as pickle

# Library for Collaborative filtering
from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

print("Done importing packages")

nltk.download('stopwords')
nltk.download('wordnet')

print("Done downloading nltk")


url = 'https://s3.amazonaws.com/csvfilesonline/csvfiles.zip'
rzip = requests.get(url)
zf = zipfile.ZipFile(io.BytesIO(rzip.content))
zf.extractall('downloaded3files')


movies_metadata = pd.read_csv('downloaded3files/csvfiles/movies_metadata.csv')


print("Done movies_metadata reading csv")


movies = pd.read_csv('downloaded3files/csvfiles/movies.csv')


print("Reading the movies from our movies data set")


df1 = movies['genres'].apply(lambda genrelist : str(genrelist).split("|"))
df1 = pd.Series(df1).apply(frozenset).to_frame(name='givengenres')
for givengenres in frozenset.union(*df1.givengenres):
    df1[givengenres] = df1.apply(lambda _: int(givengenres in _.givengenres), axis=1)
df1.drop('givengenres',axis=1,inplace=True)
df1['movieId']=movies['movieId']
df1 = pd.merge(movies,df1,on='movieId')
df1.head()
genre_columns= ['Film-Noir',
       'Romance', 'Western', 'Documentary', 'Thriller', 'Action', 'Musical',
       'War', 'Drama', 'IMAX', 'Crime', 'Children', 'Adventure', 'Horror',
       'Fantasy', 'Animation', 'Comedy', 'Mystery', '(no genres listed)',
       'Sci-Fi']


print("Done with feature engineering")


links = pd.read_csv('downloaded3files/csvfiles/links.csv')


print("Done reading links csv")

movies_metadata = movies_metadata[movies_metadata.id.isin(links['tmdbId'].astype(str).apply(lambda x:x[:-2]).tolist())]


movies_metadata['genres'] = movies_metadata['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

def convert_int(x):
    try:
        return int(x)
    except:
        return 0

links['tmdbId'] = links['tmdbId'].apply(convert_int)
links['tmdbId']

movies_metadata['id'] = movies_metadata['id'].apply(convert_int)



print("Changed metadata csv")


def return_movieId(tmdbId):
    return links[links['tmdbId']==tmdbId]['movieId'].iloc[0]


#Get movie Id to the movies_metadata
movies_metadata['movieId'] = movies_metadata['id'].apply(return_movieId)



no_of_votes = movies_metadata[movies_metadata['vote_count'].notnull()]['vote_count'].astype('int')
vote_mean = movies_metadata[movies_metadata['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_mean.mean()


m = no_of_votes.quantile(0.95)

print("weighted average calculated")

movies_metadata['year'] = pd.to_datetime(movies_metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


top_movies = movies_metadata[(movies_metadata['vote_count'] >= m) & (movies_metadata['vote_count'].notnull()) & (movies_metadata['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
top_movies['vote_count'] = top_movies['vote_count'].astype('int')
top_movies['vote_average'] = top_movies['vote_average'].astype('int')


print("Got top movies")


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

top_movies['wr'] = top_movies.apply(weighted_rating, axis=1)

top_movies = top_movies.sort_values('wr', ascending=False).head(100)


y = movies_metadata.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
y.name = 'genre'
gen_data = movies_metadata.drop('genres', axis=1).join(y)


print("Genre data ready")

def top_movies_genre(genre, percentile=0.85):
    df = gen_data[gen_data['genre'] == genre]
    no_of_votes = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_mean = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_mean.mean()
    m = no_of_votes.quantile(percentile)

    top_movies = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    top_movies['vote_count'] = top_movies['vote_count'].astype('int')
    top_movies['vote_average'] = top_movies['vote_average'].astype('int')

    top_movies['wr'] = top_movies.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    top_movies = top_movies.sort_values('wr', ascending=False).head(100)

    return top_movies


ratings = pd.read_csv('downloaded3files/csvfiles/ratings.csv')


print("Done reading ratings csv")

ratings[(ratings['userId']==1) & (ratings['rating']>2.5)]['movieId'].tolist()



def text_process(mess):
    """
    1. remove punc
    2. remove stop words
    3. apply lemmatization
    4. apply stemmization
    5. return list clean overview

    """
    #Remove Stopwords and punctuations
    nopunc = [char for char in mess if char not in string.punctuation]
    stopwords = nltk.corpus.stopwords.words('english')
    nopunc = ''.join(nopunc)

    #Apply tokenization
    tokenized_list = []
    tokenized_list =  [word for word in nopunc.split() if word.lower() not in stopwords]


    wordnet_lemmatizer = WordNetLemmatizer()
    snowball_stemmer = SnowballStemmer('english')

    #Applying Lemmatization

    lemmatized_words = []
    for word in tokenized_list:
        lemmatized_words.append(wordnet_lemmatizer.lemmatize(word))

   #Applying Stemmization

    cleaned_list  = []
    for word in lemmatized_words:
        cleaned_list.append(snowball_stemmer.stem(word))
    return ' '.join(cleaned_list)


movies_metadata['overview'] = movies_metadata['overview'].astype(str)


movies_metadata['pro_overview'] = movies_metadata['overview'].apply(text_process)


percentile = 0.90
no_of_votes = movies_metadata[movies_metadata['vote_count'].notnull()]['vote_count'].astype('int')
vote_mean = movies_metadata[movies_metadata['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_mean.mean()
m = no_of_votes.quantile(percentile)

top_movies = movies_metadata[(movies_metadata['vote_count'] >= m) & (movies_metadata['vote_count'].notnull()) & (movies_metadata['vote_average'].notnull())][['movieId','title', 'year', 'vote_count', 'vote_average', 'popularity','pro_overview']]
top_movies['vote_count'] = top_movies['vote_count'].astype('int')
top_movies['vote_average'] = top_movies['vote_average'].astype('int')


top_movies.sort_values(by='vote_count',ascending=False).head()

sss_url = "http://swoogle.umbc.edu/SimService/GetSimilarity"

def sss(s1, s2, type='relation', corpus='webbase'):
    try:
        response = get(sss_url, params={'operation':'api','phrase1':s1,'phrase2':s2,'type':type,'corpus':corpus})
        return float(response.text.strip())
    except:
        #print ('Error in getting similarity for %s: %s' % ((s1,s2), response))
        return 0.0


user_1_movies=[]
for movieId in ratings[(ratings['userId']==1) & (ratings['rating']>2.5)]['movieId'].tolist():
    user_1_movies.append(movies_metadata[movies_metadata['movieId']==movieId]['pro_overview'].iloc[0])
user_1_movies = ' '.join(user_1_movies)

top_movies['similarity'] = top_movies['pro_overview'].apply(lambda x:sss(user_1_movies,x))
top_movies[top_movies.movieId.isin(ratings[ratings['userId']!=1]['movieId'].tolist())][['title','similarity','vote_count','vote_average']].sort_values(by='similarity',ascending=False).head(10)

print("Defining function for user taste based recommendations")

def user_taste_recommender(userId,percentile = 0.90):
    no_of_votes = movies_metadata[movies_metadata['vote_count'].notnull()]['vote_count'].astype('int')
    vote_mean = movies_metadata[movies_metadata['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_mean.mean()
    m = no_of_votes.quantile(percentile)

    top_movies = movies_metadata[(movies_metadata['vote_count'] >= m) & (movies_metadata['vote_count'].notnull()) & (movies_metadata['vote_average'].notnull())][['movieId','title', 'year', 'vote_count', 'vote_average', 'popularity','pro_overview']]
    top_movies['vote_count'] = top_movies['vote_count'].astype('int')
    top_movies['vote_average'] = top_movies['vote_average'].astype('int')

    user_movies=[]
    for movieId in ratings[(ratings['userId']==userId) & (ratings['rating']>2.5)]['movieId'].tolist():
        user_movies.append(movies_metadata[movies_metadata['movieId']==movieId]['pro_overview'].iloc[0])
    user_movies = ' '.join(user_movies)

    top_movies['similarity'] = top_movies['pro_overview'].apply(lambda x:sss(user_movies,x))
    top_movies = top_movies[top_movies.movieId.isin(ratings[ratings['userId']!=userId]['movieId'].tolist())][['title','similarity','vote_count','vote_average']].sort_values(by='similarity',ascending=False).head(10)

    return top_movies

user_taste_recommender(100)

movies_metadata['tagline'] = movies_metadata['tagline'].fillna('')
movies_metadata['description'] = movies_metadata['pro_overview'] + movies_metadata['tagline']
movies_metadata['description'] = movies_metadata['description'].fillna('')


movies_metadata['description'].head()

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies_metadata['description'])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print("Got cosine similarity with tfidf matrix")

movies_metadata = movies_metadata.reset_index()
titles = movies_metadata['title']
indexes = pd.Series(movies_metadata.index, index=movies_metadata['title'])


#To get pairwise similarity score for movie with index 0
similarity =  list(enumerate(cosine_sim[0]))
print(similarity[:10])

print("writing function for description based recommendations")

def desc_based_recommendation(title):
    idx = indexes[title]
    sim = list(enumerate(cosine_sim[idx]))
    #Sorting the list by descending order of similarity
    sim = sorted(sim, key=lambda x: x[1], reverse=True)
    #Taking top 30 similar movies
    sim = sim[1:31]
    rec_movies_indexes = [i[0] for i in sim]
    return titles.iloc[rec_movies_indexes]

desc_based_recommendation('Star Wars')


#loading data from credits.csv for cast and crew, and Keywords.csv for keywords related to movies
credits = pd.read_csv('downloaded3files/csvfiles/credits.csv')
keywords = pd.read_csv('downloaded3files/csvfiles/keywords.csv')

print("got credits and keyworks csv")


#Converting id's to int
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
movies_metadata['id'] = movies_metadata['id'].astype('int')

# Add Cast and Crew column to our movies dataset
movies_metadata = movies_metadata.merge(credits, on='id')
#Add Keywords to the dataset
movies_metadata = movies_metadata.merge(keywords, on='id')


#Checking for Python literal structures: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, and None.
movies_metadata['cast'] = movies_metadata['cast'].apply(literal_eval)
movies_metadata['crew'] = movies_metadata['crew'].apply(literal_eval)
movies_metadata['keywords'] = movies_metadata['keywords'].apply(literal_eval)
#Get the cast and crew size
movies_metadata['cast_size'] = movies_metadata['cast'].apply(lambda x: len(x))
movies_metadata['crew_size'] = movies_metadata['crew'].apply(lambda x: len(x))

def get_director(d):
    for i in d:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


movies_metadata['director'] = movies_metadata['crew'].apply(get_director)


movies_metadata['cast'] = movies_metadata['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies_metadata['cast'] = movies_metadata['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)


movies_metadata['keywords'] = movies_metadata['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])



movies_metadata['cast'] = movies_metadata['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


movies_metadata['director'] = movies_metadata['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
#Add more weight to director
movies_metadata['director'] = movies_metadata['director'].apply(lambda x: [x,x, x])

k = movies_metadata.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
k.name = 'keyword'


k = k.value_counts()
k[:5]


k = k[k > 1]


stemmer = SnowballStemmer('english')
stemmer.stem('forests')


def filter_keywords(x):
    words = []
    for i in x:
        if i in k:
            words.append(i)
    return words


movies_metadata['keywords'] = movies_metadata['keywords'].apply(filter_keywords)
movies_metadata['keywords'] = movies_metadata['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
movies_metadata['keywords'] = movies_metadata['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

movies_metadata['analyzer'] = movies_metadata['keywords'] + movies_metadata['cast'] + movies_metadata['director'] + movies_metadata['genres']
movies_metadata['analyzer'] = movies_metadata['analyzer'].apply(lambda x: ' '.join(x))


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(movies_metadata['analyzer'])


# Get pairwise cosine similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)



movies_metadata = movies_metadata.reset_index()
titles = movies_metadata['title']
indexes = pd.Series(movies_metadata.index, index=movies_metadata['title'])


movies_metadata.to_csv("new_metadata.csv",index = False)

desc_based_recommendation('Star Wars').head(10)


desc_based_recommendation('Inception').head(10)


print("Defining poplarity based recommender")

def popularity_based_recommendations(title,percentile=0.70):
    idx = indexes[title]
    sim = list(enumerate(cosine_sim[idx]))
    sim = sorted(sim, key=lambda x: x[1], reverse=True)
    sim = sim[1:26]
    req_index = [i[0] for i in sim]

    movies = movies_metadata.iloc[req_index][['title', 'vote_count', 'vote_average', 'year']]
    no_of_votes = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_mean = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    m = no_of_votes.quantile(percentile)
    C = vote_mean.mean()
    top_movies = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    top_movies['vote_count'] = top_movies['vote_count'].astype('int')
    top_movies['vote_average'] = top_movies['vote_average'].astype('int')
    top_movies['wr'] = top_movies.apply(weighted_rating, axis=1)
    top_movies = top_movies.sort_values('wr', ascending=False).head(10)
    return top_movies.title


popularity_based_recommendations('Star Wars')



popularity_based_recommendations('The Dark Knight Rises')



reader = Reader()

ratings = pd.read_csv('downloaded3files/csvfiles/ratings.csv')

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)

print("SVD defining")

svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])


svd.predict(1, 302, 3)


links.drop('imdbId',axis=1,inplace=True)
links.columns=['movieId', 'id']
id_map = links.merge(movies_metadata[['title', 'id']], on='id').set_index('title')

indices_map = id_map.set_index('id')

print("Defining function for hybrid recommender")

def hybrid(userId, title):
    idx = indexes[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = movies_metadata.iloc[movie_indices][['title', 'vote_count','year', 'id']]
    movies['est rating'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est rating', ascending=False)
    return movies.title

print("Pickling models")

pickle.dump( desc_based_recommendation, open( "contentbased.pkl", "wb" ),protocol = 2)

contentbased = pickle.load(open("contentbased.pkl", "rb" ))


pickle.dump(popularity_based_recommendations, open( "popularitybased.pkl", "wb"),protocol = 2)

popularitybased = pickle.load(open("popularitybased.pkl", "rb"))



pickle.dump(hybrid, open("hybrid.pkl", "wb"),protocol = 2)

hybrid = pickle.load(open("hybrid.pkl", "rb"))

print("Pckled all the models")

def zipdir(path,ziph):
    ziph.write(os.path.join('hybrid.pkl'))
    ziph.write(os.path.join('popularitybased.pkl'))
    ziph.write(os.path.join('contentbased.pkl'))


zipf = zipfile.ZipFile('AllModels.zip','w',zipfile.ZIP_DEFLATED)
zipdir('/',zipf)
zipf.close()


def upload_to_s3(Inputlocation,Access_key,Secret_key,bucket):
    print("Uploading files to amazon")
    try:

        buck_name=bucket

        S3_client = boto3.client('s3',Inputlocation,aws_access_key_id= Access_key, aws_secret_access_key= Secret_key)

        if Inputlocation == 'us-east-1':
            S3_client.create_bucket(Bucket=buck_name)
        else:
            S3_client.create_bucket(Bucket=buck_name,CreateBucketConfiguration={'LocationConstraint': Inputlocation})

        print("connection successful")
        S3_client.upload_file("AllModels.zip", buck_name,"AllModels.zip"),
        #Callback=ProgressPercentage("AllModels.zip")

        print("Files uploaded successfully")

    except Exception as e:
        print("Error uploading files to Amazon s3" + str(e))



argLen=len(sys.argv)
Access_key=''
Secret_key=''
Inputlocation=''
bucket=''

for i in range(1,argLen):
    val=sys.argv[i]
    if val.startswith('accessKey='):
        pos=val.index("=")
        Access_key=val[pos+1:len(val)]
        continue
    elif val.startswith('secretKey='):
        pos=val.index("=")
        Secret_key=val[pos+1:len(val)]
        continue
    elif val.startswith('location='):
        pos=val.index("=")
        Inputlocation=val[pos+1:len(val)]
        continue
    elif val.startswith('bucket='):
        pos=val.index("=")
        bucket=val[pos+1:len(val)]
        continue

upload_to_s3(Inputlocation,Access_key,Secret_key,bucket)
print('files uploaded')
