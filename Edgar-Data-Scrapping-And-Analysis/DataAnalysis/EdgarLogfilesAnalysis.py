import os
import pandas as pd
import glob
import sys
import boto.s3
import boto3
import threading
import logging
from bs4 import BeautifulSoup
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import time
import datetime


def summary(all_data):
    data = pd.DataFrame()
    data = all_data
    
    
    # summary = pd.DataFrame()
    logging.debug('In the function : summary')
    csvpath=str(os.getcwd())
    
    # add Timestamp for the analysis purpose
    data['Timestamp'] = data[['date', 'time']].astype(str).sum(axis=1)
    # Create a summary that groups ip by date
    
    summary1=data['ip'].groupby(data['date']).describe()
    summaryipdescribe = pd.DataFrame(summary1)
    s=summaryipdescribe.transpose()
    s.to_csv(csvpath+"/summaryipbydatedescribe.csv")
    
    # Create a summary that groups cik by accession number
    summary2 = data['extention'].groupby(data['cik']).describe()
    summarycikdescribe = pd.DataFrame(summary2)
    summarycikdescribe.to_csv(csvpath+"/summarycikbyextentiondescribe.csv")
    
    # get Top 10 count of all cik with their accession number
    data['COUNT'] = 1  # initially, set that counter to 1.
    group_data = data.groupby(['date', 'cik', 'accession'])['COUNT'].count()  # sum function
    rankedData=group_data.rank()
    summarygroup=pd.DataFrame(rankedData)
    summarygroup.to_csv(csvpath+"/Top10cik.csv")
    
    
    # For anomaly detection -check the length of cik
    data['cik'] = data['cik'].astype('str')
    data['cik_length'] = data['cik'].str.len()
    data[(data['cik_length'] > 10)]
    data['COUNT'] = 1
    datagroup=pd.DataFrame(data)
    datagroup.to_csv(csvpath+"/LengthOfCikForAnomalyDetection.csv")
    
    
    # Per code count
    status = data.groupby(['code']).count()  # sum function
    status['COUNT']
    summary=pd.DataFrame(status)
    summary.to_csv(csvpath+"/PercodeCount.csv")

def replace_missingValues(all_data):
    data = pd.DataFrame()
    logging.debug('In the function : replace_missingValues')
    all_data.loc[all_data['extention'] == '.txt', 'extention'] = all_data["accession"].map(str) + all_data["extention"]
    all_data['browser'] = all_data['browser'].fillna('win')
    all_data['size'] = all_data['size'].fillna(0)
    all_data['size'] = all_data['size'].astype('int64')
    all_data = pd.DataFrame(all_data.join(all_data.groupby('cik')['size'].mean(), on='cik', rsuffix='_newsize'))
    all_data['size_newsize'] = all_data['size_newsize'].fillna(0)
    all_data['size_newsize'] = all_data['size_newsize'].astype('int64')
    all_data.loc[all_data['size'] == 0, 'size'] = all_data.size_newsize
    del all_data['size_newsize']
    data = all_data
    return data


def change_dataTypes(all_data):
    logging.debug('In the function : change_dataTypes')
    all_data['zone'] = all_data['zone'].astype('int64')
    all_data['cik'] = all_data['cik'].astype('int64')
    all_data['code'] = all_data['code'].astype('int64')
    all_data['idx'] = all_data['idx'].astype('int64')
    all_data['noagent'] = all_data['noagent'].astype('int64')
    all_data['norefer'] = all_data['norefer'].astype('int64')
    all_data['crawler'] = all_data['crawler'].astype('int64')
    all_data['find'] = all_data['find'].astype('int64')
    newdata = replace_missingValues(all_data)
    newdata.to_csv("merged.csv",encoding='utf-8')
    summary(newdata)
    return 0


def create_dataframe(path):
    logging.debug('In the function : create_dataframe')
    all_data = pd.DataFrame()
    for f in glob.glob(path + '/log*.csv'):
        df = pd.read_csv(f, parse_dates=[1])
        all_data = all_data.append(df, ignore_index=True)
    return all_data


def assure_path_exists(path):
    logging.debug('In a function : assure_path_exists')
    if not os.path.exists(path):
        os.makedirs(path)
    return 0


def get_dataOnLocal(monthlistdata, year):
    logging.debug('In the function : get_dataOnLocal')
    df = pd.DataFrame()
    foldername = str(year)
    path = str(os.getcwd()) + "/" + foldername
    assure_path_exists(path)
    for month in monthlistdata:
        with urlopen(month) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(path)
    df = create_dataframe(path)
    change_dataTypes(df)
    return 0


def get_allmonth_data(linkhtml, year):
    logging.debug('In the function : get_allmonth_data')
    allzipfiles = BeautifulSoup(linkhtml, "html.parser")
    ziplist = allzipfiles.find_all('li')
    monthlistdata = []
    count = 0
    for li in ziplist:
        zipatags = li.findAll('a')
        for zipa in zipatags:
            if "01.zip" in zipa.text:
                monthlistdata.append(zipa.get('href'))
    get_dataOnLocal(monthlistdata, year)
    return 0


def get_url(year):
    logging.debug('In the function : get_url')
    url = 'https://www.sec.gov/data/edgar-log-file-data-set.html'
    html = urlopen(url)
    soup = BeautifulSoup(html, "html.parser")
    all_div = soup.findAll("div", attrs={'id': 'asyncAccordion'})
    for div in all_div:
        h2tag = div.findAll("a")
        for a in h2tag:
            if str(year) in a.get('href'):
                global ahref
                ahref = a.get('href')
    linkurl = 'https://www.sec.gov' + ahref
    logging.debug('Calling the initial URL')
    linkhtml = urlopen(linkurl)
    print(linkhtml)
    get_allmonth_data(linkhtml, year)
    return 0


def valid_year(year):
    logging.debug('In the function : valid_year')
    logYear = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015',
               '2016']
    for log in logYear:
        try:
            if year in log:
                get_url(year)
        except:
            print("Data for" + year + "does not exist")
            "Data for" + year + "does not exist"
    return 0
            
def upload_to_s3(AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,inputLocation,filepaths):
  
    try:
        conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        print("Connected to S3")
    except:
        logging.info("Amazon keys are invalid!!")
        print("Amazon keys are invalid!!")
        exit()
        
    loc=''

    if inputLocation == 'APNortheast':
        loc=boto.s3.connection.Location.APNortheast
    elif inputLocation == 'APSoutheast':
        loc=boto.s3.connection.Location.APSoutheast
    elif inputLocation == 'APSoutheast2':
        loc=boto.s3.connection.Location.APSoutheast2
    elif inputLocation == 'CNNorth1':
        loc=boto.s3.connection.Location.CNNorth1
    elif inputLocation == 'EUCentral1':
        loc=boto.s3.connection.Location.EUCentral1
    elif inputLocation == 'EU':
        loc=boto.s3.connection.Location.EU
    elif inputLocation == 'SAEast':
        loc=boto.s3.connection.Location.SAEast
    elif inputLocation == 'USWest':
        loc=boto.s3.connection.Location.USWest
    elif inputLocation == 'USWest2':
        loc=boto.s3.connection.Location.USWest2
    
    try:   
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts)    
        bucket_name = 'adsassignment1part2'+str(st).replace(" ", "").replace("-", "").replace(":","").replace(".","")
        bucket = conn.create_bucket(bucket_name, location=loc)
        print("bucket created")
        s3 = boto3.client('s3',
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        
        print('s3 client created')
        
        for f in filepaths:
            try:
                s3.upload_file(f, bucket_name,os.path.basename(f),
                Callback=ProgressPercentage(os.path.basename(f)))
                print("File successfully uploaded to S3",f,bucket)
            except Exception as detail:
                print(detail)
                print("File not uploaded")
                exit()
        
    except:
        logging.info("Amazon keys are invalid!!")
        print("Amazon keys are invalid!!")
        exit()

#do not forget to use the variable filepaths
def zipdir(path, ziph, filepaths):
    ziph.write(os.path.join('merged.csv'))

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

def main():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    
    argLen=len(sys.argv)
    year=''
    accessKey=''
    secretAccessKey=''
    inputLocation=''

    for i in range(1,argLen):
        val=sys.argv[i]
        if val.startswith('year='):
            pos=val.index("=")
            year=val[pos+1:len(val)]
            continue
        elif val.startswith('accessKey='):
            pos=val.index("=")
            accessKey=val[pos+1:len(val)]
            continue
        elif val.startswith('secretKey='):
            pos=val.index("=")
            secretAccessKey=val[pos+1:len(val)]
            continue
        elif val.startswith('location='):
            pos=val.index("=")
            inputLocation=val[pos+1:len(val)]
            continue

    print("Year=",year)
    print("Access Key=",accessKey)
    print("Secret Access Key=",secretAccessKey)
    print("Location=",inputLocation)        
        
    logfilename = 'log_Edgar_'+ year + '_' + st + '.txt'
    print(logfilename)
    logging.basicConfig(filename=logfilename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Program Start')
    logging.debug('*************')    
    logging.debug('Calling the initial URL'.format(year))
    
    #generate files
    valid_year(year)
    
    #prepare log file so that it can be uploaded to cloud
    logger = logging.getLogger()
    logger.disabled = True
    
    filepaths = []
    filepaths.append(os.path.join(logfilename))
    filepaths.append(os.path.join('merged.csv'))
    
    logging.info('Compiled csv and log file zipped')
    
    upload_to_s3(accessKey,secretAccessKey,inputLocation,filepaths)
    
    logger.disabled = False

if __name__ == '__main__':
    main()

