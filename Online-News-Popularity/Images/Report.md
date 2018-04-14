## ONLINE NEWS POPULARITY ANALYSIS


1. OVERVIEW <br/>
2. PROBLEM STATEMENT <br/>
3. METRICS <br/>
4. ANALYSIS <br/>
>4.1 DATA EXPLORATION AND VISUALIZATION <br/>
>4.1.1.USING MATPLOTLIB, SEABORN, PLOTLY <br/>
>4.1.2 USING TABLEAU <br/>
>4.2 ALGORITHMS AND TECHNIQUES <br/>
5. METHODOLOGY <br/>
>5.1 DATA PREPROCESSING <br/>
>5.2 IMPLEMENTATION<br/>
>5.2.1 MANUALLY CREATED CLUSTERS <br/>
>5.2.2 CLUSTERS USING K-MEANS <br/>
>5.3 HYPERPARAMETER TUNING <br/>
6. SUMMARY <br/>
7. CONCLUSION

### 1. OVERVIEW:
***

With the help of Internet, the online news can be instantly spread around the world. Most of peoples now have the habit of reading and sharing news online, for instance, using social media like Twitter and Facebook. Typically, the news popularity can be indicated by the number of reads, likes or shares. For the online news stake holders such as content providers or advertisers, it's very valuable if the popularity of the news articles can be accurately predicted prior to the publication. Thus, it is interesting and meaningful to use the machine learning techniques to predict the popularity of online news articles. We have dataset with 39,643 articles from the Mashable website. In this project, based on the dataset including 39,643 news articles from website Mashable, we will try to find the best classification learning algorithm to accurately predict if a news article will become popular or not prior to publication.

### 2. PROBLEM STATEMENT:
***

In this project, we have used machine learning techniques to solve a binary classification problem, which is to predict if an online news article will become popular or not prior to publication. The popularity is characterized by the number of shares. If the number of shares is higher than a pre-defined threshold, the article is labeled as popular, otherwise it is labeled as unpopular.  Thus, the problem is to utilize a list of article’s features and find the best machine learning model to accurately classify the target label (popular/unpopular) of the articles to be published. As the problem can be formulated as a binary classification problem, we have implemented and compared three classification learning algorithms including Logistic Regression, RF, and Adaboost. The best model is selected based on the metrics introduced in next part.

### 3. METRICS:
***
As a classification task, we have calculated the following three evaluation metrics: accuracy, F1-score and AUC. For all three metrics, the higher value of the metric means the better
performance of model.

-  **Accuracy:** Accuracy is direct indication of the proportion of correct classification.It considers both true positives and true negatives with equal weight and it can be computed as -

    accuracy = (true positives + true negatives)/ dataset size
    
    Although the measure of accuracy might be naive when the data class distribution is highly skewed, but it is still an intuitive indication of model's performance.

- **F1-score:** F1-score is an unweighted measure for accuracy by taking harmonic mean of precision and recall, which can be computed as- 
 F1 = (2 *precision*recall)/ (precision + recall)
 It is a robust measurement since it is independent of data class distribution.

- **AUC:** The AUC is the area under the ROC (Receiver Operating Characteristics) curve, which is a plot of the True Positive Rate versus the False Positive Rate. AUC value is a good measure of classifier's discrimination power and it is a more robust measure for model performance.


### 4. ANALYSIS
***
### 4.1	DATA EXPLORATION AND VISUALIZATION
#### 4.1.1 Using matplotlib, seaborn & plotly
The dataset is consisted of 39,643 news articles from an online news website called Mashable collected over 2 years from Jan. 2013 to Jan. 2015. It is downloaded from UCI Machine Learning Repository as https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity and this dataset is generously denoted by the author. For each instance of the dataset, it has 61 attributes which includes 1 target attribute (number of shares), 2 non-predictive features (URL of the article and Days between the article publication and the dataset acquisition) and 58 predictive features as shown in Fig. below.
The dataset has already been initially preprocessed. For examples, the categorical features like the published day of the week and article category have been transformed by one-hot encoding scheme, and the skewed feature like number of words in the article has been log- transformed.


#### 4.1.2 Using Tableau
Tableau is a great data visualizing tool using which one can explore data and get useful insights from it. We have visualized the dataset in Tableau as well and found useful insights from the dataset.

### 4.2 ALGORITHMS AND TECHNIQUES 
Since we formulate this problem as a binary classification problem.In this project, three classification learning algorithms including Logistic Regression, RF,and Adaboost, Decision Tree, Support Vector Machine, etc are implemented and compared based on the evaluation metric such as accuracy, F1-score and Area Under ROC Curve (AUC). All the learning algorithm are implemented by sklearn toolbox.


### 5. METHODOLOGY
***
### 5.1	DATA PREPROCESSING
As mentioned above, some data preprocessing works have been done by the data's donator. The categorical features like the published day of the week and article categoryhave been transformed by one-hot encoding scheme, and the skewed feature like number of words in the article has been log-transformed. Based on this, I further preprocess the dataset by normalizing the numerical feature to the interval [0; 1] such that each feature is treated equally when applying supervised learning. I also select the median of target attribute as
The threshold to convert the continuous target attribute to Boolean label.Since there are 58 features in the dataset, it is reasonable to conduct a feature selection to reduce the data noise and increase the algorithm running speed. One effective way is using recursive feature elimination(RFE) with cross validation (RFECV) to automatically select the most significant features for certain classifier. Sklearn provides a function called REFCV() that can help us.

Firstly, we run RFECV with a logistic regression estimator. The cross validation score versus the number of feature selected. From the figure, we can find there is drop of score when number of features is 29. Thus RFECV algorithm selects 29 most relevant features from 58 original features. The selected 29 features are listed below figure.
Interestingly, the day of week and article category features are included in these 29 featuresv as we discussed and visualized in above section.

### 5.2 IMPLEMENTATION
### 5.2.1 MANUALLY CREATED CLUSTERS


### 5.2.2	CLUSTERS CREATED USING K-MEANS

**Cluster Description-**
We got 5 clusters with the help of K-Means
Obscure- Obscure Cluster contains of minimum shares
Mediocre- Mediocre consist of shares higher than obscure but lower than Popular
Popular- It consists of shares that are popular.
Super Popular- They are less popular than viral
Viral- Viral cluster shows shares of online news which were viral

**Model Training & Testing-**
We have used a single function for training and testing the models which calls each model recursively. The Precision, Recall, F1-Score, Accuracy are calculated with the help of classification report

Accuracy:  Accuracy = TP+TN/TP+FP+FN+TN<br/>
Precision: The question that this Precision metric answer is of all passengers that labeled as survived, how many actually survived? <br/>
Precision = TP/TP+FP <br/>
Recall: The question recall answer is: Of all the passengers that truly survived, how many did we label? <br/>
Recall = TP/TP+FN <br/>
F1 score: Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different<br/>
F1 Score = 2*(Recall * Precision) / (Recall + Precision)

### 5.3 HYPER-PARAMETERS TUNING

In this part, I have used the grid search method for each of the three classifiers to tune their hyperparameters. The grid search method exhaustively search through all possible
combinations of model parameters, cross validate the model and then determine which set
of model parameters gives the best performance. Since the grid search can help select an
optimal model parameter, thus the learning algorithm can be optimized. In sklearn, we can
use the function GridSearchCV() to implement grid search.


### 6. SUMMARY
***

To summarize, the project is conducted through following steps:
(a) **Data collection:** The dataset of some 40,000 online news articles are downloaded from
UCI Machine Learning Repository, which is originally collected and donated by the
author .

(b) **Data preprocessing:** Based on the initial data preprocessing in original dataset, I fur-
ther preprocess the dataset by normalizing the numerical feature such that each feature
is treated equally when applying supervised learning. I also select the median of the
target attribute (number of shares) as an appropriate threshold to label all the data
as either popular or unpopular.

(c) **Data exploration and visualization:** I explore the relevance of certain feature by visu-
alization. And I also visualize the data distribution by PCA.

(d) **Feature selection:** To select the most relevant features among all 58 features, I use
RFECV to select the most relevant features for each of the classifier.

(e) **Classifier implementation and hyperparamter tuning:** The classification algorithms including Logistic Regression, RF and Adaboost ,Support Vector Machine, Neural Networks, Decision Tree are implemented using sklearn. Then the model's hyperparameters are tuned by grid search method.

(f) **Model evaluation and validation:** The refined models are evaluated and compared using
three metrics (accuracy, F1-score, AUC).

### 7.CONCLUSION
***

We have used 2 Approaches for Online News Popularity-
- Manually Created Cluster-
Popular, Unpopular

- Creation of Cluster with the help of K-Means-
Obscure, Mediocre, Popular, Super Popular, Viral

With the help of manually created clusters we got the best Model as RandomForestClassifier
In K-Means Cluster we got the best Model as RandomForestClassifier:
F1-score = 0.91
Precision = 0.88 
Recall as 0.94
