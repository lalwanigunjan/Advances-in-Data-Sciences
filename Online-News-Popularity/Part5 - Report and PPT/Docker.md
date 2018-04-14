# Docker Instructions

### Docker Image For Models:

### Pull Instruction:
```
docker pull annsara95/ann_docker
```

### Run Instruction:
```
docker run -d -p 8000:8000 annsara95/ann_docker python /src/Pipeline_Assignment3.py -p 8000
```


*****


### Docker Image For Flask:

### Pull Instruction:
```
docker pull annsara95/flask_docker
```

### Run Instruction:
```
docker run -d -p 8000:8000 annsara95/ann_docker python /src/flaskapp.py -p 8000
```
