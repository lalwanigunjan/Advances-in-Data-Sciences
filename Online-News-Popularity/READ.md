## Online news Popularity Analysis
This module focuses on deploying machine learning models in production.

The dataset is from UCI Machine learning repository: 
[Online News Popularity Dataset](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)

The process is divided into 2 parts:
-  Model Design
- Model Deployment

 ### Model Design:
 ****
 1. Exploratory Data analysis
 2. Feature engineering
 3. Machine learning models
 4. Evaluation
 5. Create Pickle file for models
 6. Create a Docker image
 7. Upload on Amazon S3


### Model Deployment:
****
1. A web application is created using Flask through which user can import data. The data can be ingested through forms or REST api using json.
2. The application should allow single or batch records of data upload.
3. Dockerize using Repo2Docker. Docker image should have the latest model from Amazon S3.
4. Web application gets the latest model even if the model changes by the use of Amazon Lambda.

### Workflow:
****

 1. User inputs data through Forms as well as REST Api calls using JSON or CSV file
 2. The docker image in Amazon S3 has the pickle file of all the model.
 3. The data is run through a machine learning pipeline with all models.
 4. The best model is the one with high accuracy which displayed to the user through a csv file and a table with results.
 
### Relevant Papers:
****

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.

### Citation :
**** 

K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News. Proceedings of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence, September, Coimbra, Portugal.
