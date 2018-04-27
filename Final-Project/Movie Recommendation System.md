# Movie Recommendation System

<img src = "https://i.imgur.com/CaTD18H.jpg">

The movie recommendation system provides personalized recommendation to each user based on 2 aspects:\
**1. **Content Based Filtering****\
**2. Collaborative Based Filtering**\
**3. Hybrid method**

### Content Based Filtering:
Content Based Filtering is based on the user data by providing ratings on the movies. Based on this, the user profile is generated and recommendations are given.

### Collaborative Filtering:
Collaborative filtering is based on previous behaviour of the user and not based on present context.

a) User-User Collaborative filtering:
It is based on segmenting users on the similarity matrix. Users with similar tastes are grouped together. 

b) Item-Item Collaborative filtering: It is based on segmenting movies(here) on the similarity matrix. 

### Hybrid Method:
Hybrid Recommender leverages the best of both Content based and collaborative filtering techniques.

### Implementation:
We have implemented the Movie Recommendation System in 2 ways:\
**1. Using NTLK and Surprise Library**\
**2. GraphLabs**

## Using NLTK and Surprise Library:
Along with Movie Lens small data set, we have also used the movies metadata available from IMDb which has all the information about a movie like genre, budget, title, revenue, status etc. We have extracted metadata for only those movies that are in the Movie Lens data set. 
So now, we have movies user ratings data from Movie Lens as well as movies metadata from IMDb.

From Exploratory data analysis, we get understand:

 - **Most of the users prefer Drama in movies followed by Comedy and thriller**
 - **Average rating of movies is around 6 out of 10**

### Content based Filtering

We then use IMDbs weighted formula rule to understand the top rated movies. \
Weighted Rating (WR) = (v/(v+m)) R+(m/(v+m)) C

Where, R = average for the movie (mean) = (Rating) v = number of votes for the movie = (votes) m = minimum votes required to be listed in the C = the mean vote across the whole report.

In order **to feature in top movies chart**, a movie should have **minimum votes more than 95% of the movies in the list**. So to qualify this, a movie should have **at least 2079 votes** with an **average rating of 5.916**. We get about **455 movies which satisfy this condition.**

In order to provide personalized recommendation system also called Content based recommendation, we have to find similarity between movies and users input genre and movies.

We have used different methods:
1. Movie description based recommender
2. Metadata based recommender
3. Popularity based recommender

#### 1. Movie description based recommender:
We have build recommendation system based on:
Movie description using NLTK for text processing, tokenizing. Find top 90 percentile of movies and we have then calculated the similarity between the combined overview of the movies user has watched and the overview of the movies user hasn't watched.

We have also tried Cosine Similarity using the TF-IDF Vectorizer to understand similarity between two movies.

#### 2. Metadata based recommender
The metadata is used to provide additional information about a movie. So analysis based on metadata will result in better understanding of which movies to recommend.  The keywords, director, cast and crew helps to understand better which movies people prefer. To get movies with same director more often, we will add director 3 times and provide additional weight to this feature
Using Snowball Stemmer, we understand the root of the words used.
This proves that adding weight to the director definitely works, as most of the movies in Top 10 is of Christopher Nolan.

#### 3. Popularity Based Recommender
Since our current recommender doesn't take popularity and ratings into account, it will show a not so popular movie over a popular one. Returning a popular movie with high ratings will make more sense. Hence, taking popularity into consideration is extremely important.


### Collaborative based Filtering
It provides more better results than Content based as it provides overall user view on a movie than focusing on a particular users tastes.We have built a CF model using Scikit learn’s Surprise library which provides a simple data ingestion for making recommendations through CF. It also provides powerful algorithms like Singular Value Decomposition(SVD) to minimize RMSE and provide great recommendations.

### Hybrid Method
Hybrid Recommender leverages the best of both Content based and collaborative filtering techniques.
We take the user inputs of Genre name and movie he/she like of that genre and gives the table with list of all ratings of similar movies sorted on the basis of expected ratings by that particular user.

## GraphLabs:
It is a graph-based, high performance, distributed computation framework written in C++ which is used for Machine Learning in Python. 

![GraphLab](https://www.analyticsvidhya.com/wp-content/uploads/2015/12/architechture.png)

GraphLab is used for:
1. Large scale computation
2. Interactive Data visualization using Canvas
3. Machine Learning modelling
4. Production automation

This includes Model Based Collaborative Filtering which fetched the best results. Movielens has provided a rich dataset that allows us to study past behaviors to provide recommendations to users.We tested our dataset on 3 models: 
1. Item Based Recommendation Models 
2. Content Based Recommendation Models 
3. Popularity Based Recommendation Models 

### A Collaborative Filtering Model

The core idea works in 2 steps:

Find similar items by using a similarity metric
For a user, recommend the items most similar to the items (s)he already likes
This is done by making an item-item matrix in which we keep a record of the pair of items which were rated together.

In this case, an item is a movie. Once we have the matrix, we use it to determine the best recommendations for a user based on the movies he has already rated.

There are 3 types of item similarity metrics supported by graphlab. These are:

#### Jaccard Similarity: 
Similarity is based on the number of users which have rated item A and B divided by the number of users who have rated either A or B

#### Cosine Similarity:
Similarity is the cosine of the angle between the 2 vectors of the item vectors of A and B, Closer the vectors, smaller will be the angle and larger the cosine

#### Pearson Similarity
Similarity is the pearson coefficient between the two vectors.

In our case, **we have used cosine similarity.**

### 1. Item Based Recommendation Models:

![Item Based recommender](https://github.com/lalwanigunjan/Advances-in-Data-Sciences/blob/master/Final-Project/Screenshots/Screen%20Shot%202018-04-27%20at%203.44.51%20AM.jpg)

#### Results:
![](https://github.com/lalwanigunjan/Advances-in-Data-Sciences/blob/master/Final-Project/Screenshots/Screen%20Shot%202018-04-27%20at%203.45.04%20AM.jpg)

### 2. Content Based Recommendation Models:
![](https://github.com/lalwanigunjan/Advances-in-Data-Sciences/blob/master/Final-Project/Screenshots/Screen%20Shot%202018-04-27%20at%203.47.39%20AM.jpg)

#### Results:
![](https://github.com/lalwanigunjan/Advances-in-Data-Sciences/blob/master/Final-Project/Screenshots/Screen%20Shot%202018-04-27%20at%203.47.47%20AM.jpg)

### 3. Popularity based recommender:
Popularity based recommender understands the popularity of a movie and recommends movies based on high popularity and high ratings.\
![](https://github.com/lalwanigunjan/Advances-in-Data-Sciences/blob/master/Final-Project/Screenshots/Screen%20Shot%202018-04-27%20at%203.44.08%20AM.jpg)

#### Results:
![](https://github.com/lalwanigunjan/Advances-in-Data-Sciences/blob/master/Final-Project/Screenshots/Screen%20Shot%202018-04-27%20at%203.44.18%20AM.jpg)


Reference: https://www.analyticsvidhya.com/blog/2015/12/started-graphlab-python/
