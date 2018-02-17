#importing libraries
import requests
import zipfile,io
import pandas as pd
import csv
import re
import os
import logging
import sys
import shutil
import glob # to get the file recursively
import json
import boto.s3
from boto.s3.key import Key


##log file initialization
root = logging.getLogger()
root.setLevel(logging.DEBUG)

#output the  log to a file
log = logging.FileHandler('problem2_log.log')
log.setLevel(logging.DEBUG)
#creation of formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log.setFormatter(formatter)
root.addHandler(log)

#print the logs in console
console = logging.StreamHandler(sys.stdout )
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
root.addHandler(console)


# Enter the year
with open('config.json') as json_data:
    a=json.load(json_data)
    try:
        def incheck(arg):
            default = '2013'
            if re.match('[0-9]{4}',arg):
                return arg
    except Exception as e:
        print("Check the entered details")
        logging.info(str(e))
        sys.exit()

    yr = a['args'][0]

    try:
        year = incheck(yr)
        AWS_ACCESS_KEY_ID=a['args'][1]
        inputLocation=a['args'][2]
    except Exception as e:
        logging.error(str(e))
        logging.info("Check the entered details")
        quit()

#make directories
try:
    if not os.path.exists('downloaded_zips_unzipped'+year):
        os.makedirs('downloaded_zips_unzipped'+year, mode=0o777)
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__),'downloaded_zips_unzipped'+year),ignore_errors=False)
        os.makedirs('downloaded_zips_unzipped'+year, mode=0o777)
    logging.info('Directories cleanup completed')
except Exception as e:
    logging.error(str(e))
    exit()


#download the zip file
url ='https://www.sec.gov/dera/data/Public-EDGAR-log-file-data/'

date ='01'
month_link = list()
for month in range(1,13):
    if month < 4:
        qtr = 'Qtr1'
    elif month <7:
        qtr = 'Qtr2'
    elif month <10:
        qtr = 'Qtr3'
    else:
        qtr = 'Qtr4'

    try:
        l_month = url+year+'/'+qtr+'/log'+year+str(month).zfill(2)+date+'.zip'
        print(l_month)
        month_link.append(l_month)
        rzip = requests.get(l_month)
        zf = zipfile.ZipFile(io.BytesIO(rzip.content))
        zf.extractall('downloaded_zips_unzipped'+year)
        logging.info(' Downloaded Log file %s for First date of month.', l_month)

    except Exception as e:
        logging.info('File not available for %s of %s',month,year)


logging.info(' Downloaded all the Log file for %s',year)

# Loading and Merging of csv Files into Pandas DataFrame

try:
    filelists = glob.glob('downloaded_zips_unzipped'+year +"/*.csv")
    dataset = pd.DataFrame()
    list_ = []
    for file_ in filelists:
        df=pd.read_csv(file_,index_col=None,header=0)
        list_.append(df)
    dataset = pd.concat(list_)
    logging.info('All the csv read into individual dataframes')
except Exception as e:
    logging.error(str(e))
    exit()

#Detecting Anomalies
#Handling Missing Values and Computing
try:

    # total number of null values
    null_count = dataset.isnull().sum()
    logging.info('Count of Null Values in all variables are \n%s',null_count)

    #removing rows with no ip, date, time, cik , accession
    dataset.dropna(subset=['ip'])
    dataset.dropna(subset=['date'])
    dataset.dropna(subset=['time'])
    dataset.dropna(subset=['cik'])
    dataset.dropna(subset=['accession'])

    #idx should be either 0 or 1
    incorrect_idx = (~dataset['idx'].isin([0.0,1.0])).sum()
    logging.info('There are %s idx which are not 0 or 1 in the log file', incorrect_idx)

    #norefer should be either 0 or 1
    incorrect_norefer = (~dataset['norefer'].isin([0.0,1.0])).sum()
    logging.info('There are %s norefer which are not 0 or 1 in the log file', incorrect_norefer)

    #noagent should be either 0 or 1
    incorrect_noagent = (~dataset['noagent'].isin([0.0,1.0])).sum()
    logging.info('There are %s noagent which are not 0 or 1 in the log file ', incorrect_noagent)

    #crawling should be either 0 or 1
    incorrect_crawler = (~dataset['crawler'].isin([0.0,1.0])).sum()
    logging.info('There are %s crawler which are not 0 or 1 in the log file ', incorrect_crawler)

    #dropping empty column browser

    try:
        #max_code = pd.DataFrame(dataset.groupby('browser').size().rename('NotMentioned'))
        dataset['browser'] = dataset['browser'].fillna('NAV')
        logging.info('NaN values in browser replaced with maximum count browser.')
        print("START")


    except:
        dataset= dataset.dropna(axis='columns',how='all')
        logging.info('All the values in browser are NaN so column browser is deleted.')

    #replace nan idx with max idx
    max_idx = pd.DataFrame(dataset.groupby('idx').size().rename('cty')).idxmax()[0]
    dataset['idx'] = dataset['idx'].fillna(max_idx)


    #replace nan code with max code
    max_code = pd.DataFrame(dataset.groupby('code').size().rename('cty')).idxmax()[0]
    dataset['code'] = dataset['code'].fillna(max_idx)


    #replace nan norefer with one
    dataset['norefer'] = dataset['norefer'].fillna('1')

    #replace nan noagent with one
    dataset['noagent'] = dataset['noagent'].fillna('1')


    #replace nan find with min find
    min_find = pd.DataFrame(dataset.groupby('find').size().rename('cty')).idxmax()[0]
    dataset['find'] = dataset['find'].fillna(min_find)

    #replace nan crawler with zero
    dataset['crawler'] = dataset['crawler'].fillna('0')

    #replace nan extension with max extension
    max_extention = pd.DataFrame(dataset.groupby('extention').size().rename('cty')).idxmax()[0]
    dataset['extention'] = dataset['extention'].fillna(max_extention)

    #replace null values of size with mean of the size
    dataset['size']=dataset['size'].fillna(dataset['size'].mean(axis = 0))

    #replace nan zone with max zone
    max_zone = pd.DataFrame(dataset.groupby('zone').size().rename('cty')).idxmax()[0]
    dataset['zone'] = dataset['zone'].fillna(max_zone)

    ##### Summary Metrics #####
    #Compute mean size
    dataset['size_mean'] = dataset['size'].mean(axis=0)

    #Compute maximum used browser
    dataset['max_browser'] = pd.DataFrame(dataset.groupby('browser').size().rename('cty')).idxmax()[0]

    #Compute distinct count of ip per month
    dataset['ip_count'] = dataset['ip'].nunique()

    #Compute distinct count of cik per month
    dataset['cik_count'] = dataset['cik'].nunique()

    logging.info('Rows removed where ip, date, time, cik or accession were null.')
    logging.info('NaN values in idx replaced with maximum idx.')
    logging.info('NaN values in code replaced with maximum code.')
    logging.info('NaN values in norefer replaced')
    logging.info('NaN values in noagent replaced')
    logging.info('NaN values in find replaced with minimum find.')
    logging.info('NaN values in crawler replaced')
    logging.info('NaN values in extension replaced with maximum extension.')
    logging.info('NaN values in size replaced with mean value of size.')
    logging.info('NaN values in zone replaced with maximum zone.')
    logging.info(dataset.head())
except Exception as e:
    logging.error(str(e))
    exit()


# Combining all dataframe to master_csv

logging.info('Start of conversion of csv files')
dataset.to_csv('master_csv.csv')
logging.info('All dataframes of csvs are combined and exported as csv: master_csv.csv.')



# zip the csvs and log files
def zipdir(path,ziph):
    ziph.write(os.path.join('master_csv.csv'))
    ziph.write(os.path.join('problem2_log.log'))

zipf = zipfile.ZipFile('Assignment_Part2.zip','w',zipfile.ZIP_DEFLATED)
zipdir('/',zipf)
zipf.close()
logging.info('Compiled csv and log file zipped')





############### Upload the zip to AWS S3 ###############
############### Fetch the location argument if provided, else user's system location is taken ###############
loc = ''
try:
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
        bucket_name = AWS_ACCESS_KEY_ID.lower()+str(st).replace(" ", "").replace("-", "").replace(":","").replace(".","")
        bucket = conn.create_bucket(bucket_name, location=loc)
        print("bucket created")
        zipfile = 'Assignment_Part2.zip'
        print ("Uploading %s to Amazon S3 bucket %s", zipfile, bucket_name)

        def percent_cb(complete, total):
            sys.stdout.write('.')
            sys.stdout.flush()

        k = Key(bucket)
        k.key = 'Problem2'
        k.set_contents_from_filename(zipfile,cb=percent_cb, num_cb=10)
        print("Zip File successfully uploaded to S3")
    except:
        logging.info("Amazon keys are invalid!!")
        print("Amazon keys are invalid!!")
        exit()
    else:
        logging.info("It was nice serving you :)")
        exit()
except Exception as e:
    logging.error(str(e))
    logging.info("Good Bye !!! ")
    quit()
