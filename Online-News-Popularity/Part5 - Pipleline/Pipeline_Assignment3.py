import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 
import joblib
from sklearn.grid_search import GridSearchCV
import pickle
from boto.s3.key import Key
from boto.s3.connection import S3Connection
import time
import datetime
import os
import boto.s3
import sys
import boto3
import threading
import logging

df = pd.read_csv("OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()


sharesclusters = df.shares.values().reshape(-1,1)

kmeans = KMeans(n_clusters=5, random_state=0).fit(sharesclusters)
labels = kmeans.labels_
df['clusters'] = labels

df = df.drop('shares',axis=1)
df['popularity']= df['clusters']

features_top15= ['LDA_00','LDA_02','is_weekend','weekday_is_friday','weekday_is_monday','weekday_is_thursday',
                 'weekday_is_tuesday','weekday_is_wednesday','LDA_04', 'LDA_01','LDA_03','n_non_stop_unique_tokens',
                'n_unique_tokens','avg_positive_polarity','avg_negative_polarity']

X = df[features_top15]

y = df['popularity']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)


# # Random Forest

# In[65]:


pipeline = make_pipeline(preprocessing.MinMaxScaler(), 
                         RandomForestClassifier())
hyperparameters = {'randomforestclassifier__n_estimators': [10,20,50,100,250,500],
                    'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
                  'randomforestclassifier__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline,param_grid= hyperparameters,cv=5)

clf.fit(X_train,y_train)

clf.best_estimator_

clf.predict(X_test)

rfscore = clf.score(X_test,y_test)

pkl_filename6 = "RandomForest.pkl"
with open(pkl_filename6, 'wb') as file:  
    pickle.dump(clf, file)


# # Logistic Regression

# In[66]:


lrpipeline = make_pipeline(preprocessing.MinMaxScaler(), 
                         LogisticRegression('lbfgs'))
parameters_LR = {"logisticregression__penalty": ['l1','l2'],
              "logisticregression__C": [0.1,0.5,1.,2.,2.5,5]}
lr = GridSearchCV(lrpipeline,param_grid= parameters_LR,cv=5)
lr.fit(X_train,y_train)

lr.best_estimator_

lrscore = lr.score(X_test,y_test)

pkl_filenamel = "LogisticRegression.pkl"
with open(pkl_filenamel, 'wb') as file:  
    pickle.dump(lr, file)


# # Adaboost

# In[67]:


adapipeline = make_pipeline(preprocessing.MinMaxScaler(), 
                         AdaBoostClassifier())
parameters_ADA = {"adaboostclassifier__n_estimators": [100,200,300,400],
              "adaboostclassifier__learning_rate": [0.1,0.5,1]}
ada = GridSearchCV(adapipeline,param_grid= parameters_ADA,cv=5)
ada.fit(X_train,y_train)

adascore = ada.score(X_test,y_test)

ada.best_estimator_

pkl_filename5 = "AdaptiveBoosting.pkl"
with open(pkl_filename5, 'wb') as file:  
    pickle.dump(ada, file)


# # Neural Nets

# In[68]:


neuralnets = make_pipeline(preprocessing.MinMaxScaler(), 
                         MLPClassifier())
parameters_NN = {"mlpclassifier__activation": ['logistic','tanh','relu'],
              "mlpclassifier__learning_rate": ['constant', 'invscaling', 'adaptive']}
nn = GridSearchCV(neuralnets,param_grid= parameters_NN,cv=5)
nn.fit(X_train,y_train)

nn.best_estimator_

nnscore = nn.score(X_test,y_test)

pkl_filename8 = "NeuralNetwork.pkl"
with open(pkl_filename8, 'wb') as file:  
    pickle.dump(nn, file)


# # SVC

# In[69]:


SVCpipeline = make_pipeline(preprocessing.MinMaxScaler(), SVC())
parameters_SVC = {}
svc = GridSearchCV(SVCpipeline,param_grid= parameters_SVC,cv=5)
svc.fit(X_train,y_train)

svcscore = svc.score(X_test,y_test)

svc.best_estimator_

pkl_filename3 = "SupportVectorMachine.pkl"
with open(pkl_filename3, 'wb') as file:  
    pickle.dump(svc, file)


# # Naive Bayes

# In[70]:


nbpipe = make_pipeline(preprocessing.MinMaxScaler(),GaussianNB())
parameters_nb = {}
nb = GridSearchCV(nbpipe,param_grid= parameters_nb,cv=5)
nb.fit(X_train,y_train)

nb.best_estimator_

nbscore = nb.score(X_test,y_test)

pkl_filename2 = "NaiveBayes.pkl"
with open(pkl_filename2, 'wb') as file:  
    pickle.dump(nb, file)


# # KNeighborsClassifier

# In[71]:


knnpipe = make_pipeline(preprocessing.MinMaxScaler(),KNeighborsClassifier())
parameters_knn = {}
knn = GridSearchCV(knnpipe,param_grid= parameters_knn,cv=5)
knn.fit(X_train,y_train)

knn.best_estimator_

knnscore = knn.score(X_test,y_test)

pkl_filename7 = "k-NearestNeighbors.pkl"
with open(pkl_filename7, 'wb') as file:  
     pickle.dump(knn, file)


# # DecisionTreeClassifier

# In[72]:


dtcpipe = make_pipeline(preprocessing.MinMaxScaler(),KNeighborsClassifier())
parameters_dtc = {}
dtc = GridSearchCV(dtcpipe,param_grid= parameters_dtc,cv=5)
dtc.fit(X_train,y_train)

dtc.best_estimator_

dtcscore = dtc.score(X_test,y_test)

pkl_filename4 = "DecisionTree.pkl"
with open(pkl_filename4, 'wb') as file:  
    pickle.dump(dtc, file)


# # Converting Metrics to csv

# In[73]:


Acc_GridSearch_Test = np.array([svcscore,rfscore,nnscore,knnscore,lrscore,dtcscore,adascore,nbscore])

df2 = pd.read_csv('accuracyonlinenewprediction.csv',index_col ='rank')

df2["Grid_Search"] = Acc_GridSearch_Test

newcols = ['Model','Accuracy_test','Accuracy_train', 'Grid_Search','F1_score_test','F1_score_train','Precision_test',
 'Precision_train','Recall_test','Recall_train','Accuract with cv','Deviation(+/-)','time']

df2 = df2[newcols]

df2['Model'] = df2['Model'].replace({'NaiveBayers': 'NaiveBayes'}) 

df2.to_csv("Evaluation_Metrics.csv")

def zipdir(path,ziph):
    ziph.write(os.path.join('LogisticRegression.pkl'))
    ziph.write(os.path.join('NaiveBayers.pkl'))
    ziph.write(os.path.join('SupportVectorMachine.pkl'))
    ziph.write(os.path.join('DecisionTree.pkl'))
    ziph.write(os.path.join('AdaptiveBoosting.pkl'))
    ziph.write(os.path.join('RandomForest.pkl'))
    ziph.write(os.path.join('k-NearestNeighbors.pkl'))
    ziph.write(os.path.join('NeuralNetwork.pkl'))
    ziph.write(os.path.join("Evaluation_Metrics.csv"))
    
zipf = zipfile.ZipFile('AllModels.zip','w',zipfile.ZIP_DEFLATED)
zipdir('/',zipf)
zipf.close()

