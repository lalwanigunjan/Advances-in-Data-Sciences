# Movie Recommendation System

The movie recommendation system provides personalized recommendation to each user based on 2 aspects:
1. Content Based Filtering
2. Collaborative Based Filtering

### Content Based Filtering:
Content Based Filtering is based on the user data by providing ratings on the movies. Based on this, the user profile is generated and recommendations are given.

### Collaborative Filtering:
Collaborative filtering is based on previous behaviour of the user and not based on present context.

a) User-User Collaborative filtering:
It is based on segmenting users on the similarity matrix. Users with similar tastes are grouped together. 

b) Item-Item Collaborative filtering: It is based on segmenting movies(here) on the similarity matrix. 

### Implementation:
We have implemented the Movie Recommendation System in 2 ways:
1. Using NTLK and Surprise Library
2. GraphLabs

### Using NLTK and Surprise Library:
Along with Movie Lens small data set, we have also used the movies metadata available from IMDb which has all the information about a movie like genre, budget, title, revenue, status etc. We have extracted metadata for only those movies that are in the Movie Lens data set. 
So now, we have movies user ratings data from Movie Lens as well as movies metadata from IMDb.

From Exploratory data analysis, we get understand:

 - **Most of the users prefer Drama in movies followed by Comedy and thriller**
 - **Average rating of movies is around 6 out of 10**

We then use IMDbs weighted formula rule to understand the top rated movies. 

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



### GraphLabs:
It is a graph-based, high performance, distributed computation framework written in C++ which is used for Machine Learning in Python. 

![GraphLab](https://www.analyticsvidhya.com/wp-content/uploads/2015/12/architechture.png)

GraphLab is used for:
1. Large scale computation
2. Interactive Data visualization using Canvas
3. Machine Learning modelling
4. Production automation


Reference: https://www.analyticsvidhya.com/blog/2015/12/started-graphlab-python/
