#libraries imported
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import re
import os
import logging
import sys
import zipfile,io
import shutil
import json
import boto.s3
from boto.s3.key import Key


##log file initialization
root = logging.getLogger()
root.setLevel(logging.DEBUG)

##making directory
try:
    if not os.path.exists('Data_Scrapping'):
        os.makedirs('Data_Scrapping', mode=0o777)
        logging.info("Data_Scrapping folder created")
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__),'Data_Scrapping'),ignore_errors=False)
        os.makedirs('Data_Scrapping', mode=0o777)
    logging.info('Data_Scrapping folder cleanup completed')
    if not os.path.exists('Data_Scrapping/extracted_csv'):
        os.makedirs('Data_Scrapping/extracted_csv', mode=0o777)
        logging.info("Data_Scrapping/extracted_csv folder created")
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__),'Data_Scrapping/extracted_csv'),ignore_errors=False)
        os.makedirs('Data_Scrapping/extracted_csv', mode=0o777)
        logging.info("Data_Scrapping/extracted_csv folder created")
except Exception as e:
    logging.error(str(e))
    exit()

##output the  log to a file
log = logging.FileHandler('Data_Scrapping/problem1_log.log')
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





# taking input from commandline
with open('config.json') as json_data:
    a=json.load(json_data)
    print(a)
    argLen=len(a)
    cik=a['args'][0]
    accession_no=a['args'][1]
    years=a['args'][2]
    AWS_ACCESS_KEY_ID=a['args'][3]
    inputLocation=a['args'][4]
    '''
    for i3 in range(1,argLen):
        val=sys.argv[i3]
        if val.startswith('cik='):
            pos=val.index("=")
            cik=val[pos+1:len(val)]
            continue
        elif val.startswith('accessionNumber='):
            pos=val.index("=")
            accession_no=val[pos+1:len(val)]
            continue
        elif val.startswith('year='):
            pos=val.index("=")
            years=val[pos+1:len(val)]
            continue
        elif val.startswith('AWS_ACCESS_KEY_ID='):
            pos=val.index("=")
            AWS_ACCESS_KEY_ID=val[pos+1:len(val)]
            continue
        elif val.startswith('location='):
            pos=val.index("=")
            inputLocation=val[pos+1:len(val)]
            continue
        '''

print("CIK=",cik)
print("Accession Number=",accession_no)
print("Access Key=",AWS_ACCESS_KEY_ID)
print("Location=",inputLocation)


def incheck(cik,acc_no,yr):
    try:
        if (re.match('[0-9]+',cik) and re.match('[0-9]+',acc_no) and re.match('[0-9]{4}',yr)):
            cik_strip = cik.lstrip('0')
            y = yr[-2:]
            url = 'http://www.sec.gov/Archives/edgar/data/'
            link = url+cik_strip+'/'+cik+'-'+y+'-'+acc_no+'/'+acc_no+'-index.html'
            return link
    except Exception as e:
        print("Check entered CIK,ACC_NO,year")
        logging.info(str(e))
        sys.exit()

try:
    link = incheck(cik,accession_no,years)
    request = requests.get(link)
except Exception as e:
    logging.error(str(e))
    logging.info("Check the entered details")
    quit()


# Generation of URL using acc_no
try:
    html_doc = request.text

    #Getting 10q tag#
    soup = BeautifulSoup(html_doc,'lxml')
    a_tags = soup.find_all('a')
    table_tags = soup.find_all('table',class_='tableFile')
    chk = 0
    for table_tag in table_tags:
        for tr in table_tag.find_all('tr'):
                td = tr.find_all('td')
                for row in td:
                    if (str(row.string) == '10-Q'):
                        tr_crt = tr
                        chk = 1
                    if chk == 1:
                        a_tag = tr_crt.find('a')
                        linkq = a_tag.get('href')
                        link10q = ('https://www.sec.gov'+linkq)

    request1 = requests.get(link10q)
    html_doc1 = request1.text
    soup1 = BeautifulSoup(html_doc1,'lxml')
    logging.info("10q link founded %s",link10q)


    #Getting table data using table tag
    table_tags = soup1.find_all('table')
    clean_tables=[]
    for table in table_tags:
        for tr in table.find_all('tr'):
            flag = 0
            for td in tr.find_all('td'):
                if('$' in td.get_text() or '%' in td.get_text()):
                    clean_tables.append(table)
                    flag = 1
                    break
            if(flag == 1):
                break
    logging.info("Data Scrapped from tables")
    i =0
    # Transfering data to csv file

    pd.set_option('display.max_colwidth', -1)

    clean_tables_df = pd.DataFrame(clean_tables)
    for row in clean_tables_df.itertuples():
        calls_df = pd.read_html(str(row),header = 0)
        my_df = pd.DataFrame(calls_df)
        file = cik+years+accession_no+str(i)+'.csv'
        with open(os.path.join('Data_Scrapping/extracted_csv', file),'w') as outfile:
            my_df.to_csv(outfile,index = False, header = 0,sep =',',encoding = 'utf-8')
        i = i + 1
    logging.info("Data transfered to csv files")

    #zipfile creation
    zf = zipfile.ZipFile("Data_Scrapping.zip", "w")
    for dirname, subdirs, files in os.walk("Data_Scrapping"):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

    logging.info('Compiled csv and log file zipped')

except Exception as e:
    logging.error(str(e))
    logging.info("The program terminated")
    quit()

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
        zipfile = 'Problem1.zip'
        print ("Uploading %s to Amazon S3 bucket %s", zipfile, bucket_name)

        def percent_cb(complete, total):
            sys.stdout.write('.')
            sys.stdout.flush()

        k = Key(bucket)
        k.key = 'Problem1'
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
