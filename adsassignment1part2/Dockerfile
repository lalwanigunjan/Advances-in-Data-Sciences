# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory to /ads1part2
WORKDIR /ads1part2

# Copy the current directory contents into the container at /assignment1part1
ADD . /ads1part2

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install lxml

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run assignment1part2.py when the container launches
CMD ["python", "assignment1part2.py"]




























