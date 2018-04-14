# Classification Models-

This part is done for knowing the popularity of Online News using K-Means Algorithm.

1. The Popularity of news article was divided into 5 clusters-
- Viral
- Super Popular 
- Popular
- Mediocre
- Obscure
  

2. Features were scaled using MinMaxScaler

3. Feature Selection was done with -
  - Boruta
  - RFE
 
 Result:
 Features with the highest importance was selected for Training the Model
 
 4. The training data is passed into all_classification_model fuction-which recursively calls the other functions-
  - logistic_regression_model
  - naive_bayers_model
  - support_vector_machine_model
  - DecisionTreeClassifier_model
  - AdaBoostClassifier_model
  - RandomForestClassifier_model
  - knn_model
  - NeuralNetworkClassifier_model
  And all of the above models are trained with the training data.
  
  5. Then each of the model calls the function calculate_recall_precision function.
  which calls the function classification_report_csv
  and classification report for each individual model is created.
  Cross Validation is used to validate the models.
  
  6. Then the error_metircs,accuracy is created and stored in pandas dataframe.
 
  7. And the models are ranked by taking into consideration.
  - Precision
  - Recall
  - F1_score
  - Accuracy_score.
  
  8. Create pickle file for all the Model.
  
  9. Pickle files with error_metrics csv is uploaded to s3.
 
  
