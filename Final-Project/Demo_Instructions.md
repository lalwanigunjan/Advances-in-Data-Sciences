# Demo Instructions:

### Steps to run flask app:

Public DNS name - ec2-54-172-69-233.compute-1.amazonaws.com


### Docker Instructions:
1. Docker CE should be installed on the system.
2. Open docker terminal and run the following command:

> #### Docker pull command:  
```
docker pull rishabhjain27/movierecommendation:latest
```

> #### Docker Run Command: 
```
docker run -it rishabhjain27/movierecommendation:latest python Project_pipeline.py accessKey=<_accessKey_> secretKey=<_secretKey_> location=<_us-east-1_> bucket=<_bucket_name_>
```

