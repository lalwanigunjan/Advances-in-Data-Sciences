FROM python:3

# Install app dependencies
RUN pip install Flask


# install additional python packages
RUN pip3 install ipython
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install BeautifulSoup4
RUN pip3 install scipy
RUN pip3 install requests
RUN pip install boto
RUN pip install boto3


# Bundle app source
COPY EdgarDataScrapping.py /src/EdgarDataScrapping.py
COPY EdgarLogfilesAnalysis.py /src/EdgarLogfilesAnalysis.py

CMD ["python", "/src/EdgarDataScrapping.py"]
CMD ["python", "/src/EdgarLogfilesAnalysis.py"]