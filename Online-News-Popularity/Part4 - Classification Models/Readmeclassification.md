Classification Models-
For knowing the popularity of Online News K-Means Algorithm was used.
1.The Popularity of news article was divided into 5 clusters-
  a.Viral
  b.Super Popular 
  c.Popular
  d.Mediocre
  e.Obscure
2. Features were scaled using MinMaxScaler
3.Feature Selection was done with -
  a.Boruta
  b.RFE
 Features with the highest importance was selected for Training the Model
4.The training data is passed into all_classification_model fuction-
  which calls the function-
  a.logistic_regression_model
  b.naive_bayers_model
  c.support_vector_machine_model
  d.DecisionTreeClassifier_model
  e.AdaBoostClassifier_model
  f.RandomForestClassifier_model
  g.knn_model
  h.NeuralNetworkClassifier_model
  And all of the above models are trained with the training data.
 5.Then each of the model calls the function calculate_recall_precision function.
  which calls the function classification_report_csv
  and classification report for each individual model is created.
  Cross Validation is used to validate the models.
 6.Then the error_metircs,accuracy is created and stored in pandas dataframe.
 7.And the models are ranked by taking into consideration.
  Precision,Recall,F1_score,Accuracy_score.
 8. All the Models are pickled.
 9. Pickled model with error_metrics csv is uploaded to s3.
 
  
