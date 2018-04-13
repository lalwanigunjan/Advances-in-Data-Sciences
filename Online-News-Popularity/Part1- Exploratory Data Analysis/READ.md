# Exploratory Data Analysis

EDA helps us to explore the data and find out insights from it by visualizing the data. It is extremely important step in order to proceed further for analysis on it.
For the Online news popularity analysis, we need to analyze when can a online news article be popular.
All the 58 features affect the number of shares. However, which one is most important we can understand through EDA.

 1. Which channel content are users more interested in reading?
 2. Which day of the week will an article be more popular??
 3. What should be the content of the article?
 4. What should be the title of my article?
 5. Should my content be positive, negative or neutral?
 
 All these questions can be answered by looking at the data by visualizing it.
1.	Title Sentiment Polarity:
How does the title of an article affect its shares?
 
Title sentiment Polarity vs Shares

RESULT:
Mostly the articles have titles which are not too positive or negative. It lies within the range of -0.5 to 0.5. However highest concentration can be seen in the 0 axis i.e. high no. of articles is neutral in nature.

2.	Weekday vs Weekend:
When are most articles published? Is it more on weekdays or weekends? 



 
Weekends vs Weekday Pie chart

RESULT:
The above graph shows that most of the articles are published in Weekdays. However, in order to get more popularity on must publish the article on weekends.

3.	Global subjectivity:
Understanding the global subjectivity of an article 
 
Distribution of Global subjectivity


RESULT:
Maximum of global subjectivity lies from 0.2 to 0.8. A significant outlier lies at global subjectivity of 0.503 with share of 957. Hence, we conclude that most of the articles with medium global subjectivity are maximum shares.

4.	Average Positive Polarity:
 

Average Positive Polarity

RESULT:
Most of the shared articles are slightly to medium positive in polarity.

5.	Average Negative Polarity:
 
Average Negative polarity

RESULT:
Most articles shared are neutral in nature while some are slightly negative.

