# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory to /fp
WORKDIR /fp
# Copy the current directory contents into the container at /fp
ADD . /fp

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install scikit-surprise
RUN pip3 install lxml
