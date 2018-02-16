# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory to /assignment1part1
WORKDIR /assignment1part1

# Copy the current directory contents into the container at /assignment1part1
ADD . /assignment1part1

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install lxml

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run assignment1part1.py when the container launches
CMD ["python", "assignment1part1.py"]




























