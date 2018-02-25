import re
from bs4 import BeautifulSoup
import urllib.request
import zipfile
import datetime
import csv
import sys
import os
import logging
import time
import boto3
import threading
import boto.s3


def get_url(cik, accession):
    logging.debug('In the function : get_url')
    cik = str(cik)
    accession = str(accession)
    cik = cik.lstrip('0')
    acc = re.sub(r'[-]', r'', accession)
    url = 'https://www.sec.gov/Archives/edgar/data/' + cik + '/' + acc + '/' + accession + '/-index.htm'
    logging.debug('Calling the initial URL for CIK {} & Accession Number {} to open URL {}'.format(cik, acc, url))
    try:
        page_open = urllib.request.urlopen(url)
        if page_open.code == 200:
            logging.debug("URL Exisits")
            return get_final_url(url)
    except:
        logging.debug("Invalid URL. Please re-validate".format(url))
        print("Invalid URL. Please re-validate".format(url))


def get_final_url(url):
    final_url = ""
    logging.debug('In the function : get_final_url')
    html = urllib.request.urlopen(url)
    soup = BeautifulSoup(html, "html.parser")
    all_tables = soup.find('table', class_='tableFile')
    tr = all_tables.find_all('tr')
    for row in tr:
        final_url = row.findNext("a").attrs['href']
        break
    next_url = "https://www.sec.gov" + final_url
    logging.debug("Final URL {}:".format(next_url))
    print(next_url)
    return get_soup(next_url)
    #return (next_url)


def get_soup(url):
    try:
        logging.debug('In the function : get_soup')
        htmlpage = urllib.request.urlopen(url)
        page = BeautifulSoup(htmlpage, "html.parser")
        return find_all_tables(page)
    except:
        return None


def find_all_tables(page):
    logging.debug('In the function : find_all_tables')
    all_divtables = page.find_all('table')
    find_all_datatables(page, all_divtables)
    return foldername(page)


def foldername(page):
    title = page.find('filename').contents[0]
    if ".htm" in title:
        foldername = title.split(".htm")
        logging.debug('In the function : foldername{}'.format(foldername[0]))
        return foldername[0]


def zip_dir(path_dir, path_file_zip=''):
    print(os.path.dirname(path_dir))
    if not path_file_zip:
        logging.debug('In a function : zip_dir')
        path_file_zip = os.path.join(
            os.path.dirname(path_dir), os.path.basename(path_dir) + '.zip')
    with zipfile.ZipFile(path_file_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(path_dir):
            for file_or_dir in files + dirs:
                
                zip_file.write(
                    os.path.join(root, file_or_dir),
                    os.path.relpath(os.path.join(root, file_or_dir),
                                    os.path.join(path_dir, os.path.pardir)))


def assure_path_exists(path):
    logging.debug('In a function : assure_path_exists')
    if not os.path.exists(path):
        os.makedirs(path)


def checktag(param):
    setflag = "false"
    datatabletags = ["background", "bgcolor", "background-color"]
    for x in datatabletags:
        if x in param:
            setflag = "true"
    return setflag

def checkheadertag(param):
    logging.debug('In a function : checkheadertag')
    setflag="false"
    datatabletags=["center","bold"]
    for x in datatabletags:
        if x in param:
            setflag="true"
    return setflag


def printtable(table):
    logging.debug('In a function : printtable')
    printtable = []
    printtrs = table.find_all('tr')
    for tr in printtrs:
        data=[]
        pdata=[]
        printtds=tr.find_all('td')
        for elem in printtds:
            x=elem.text;
            x=re.sub(r"['()]","",str(x))
            x=re.sub(r"[$]"," ",str(x))
            if(len(x)>1):
                x=re.sub(r"[—]","",str(x))
                pdata.append(x)
        data=([elem.encode('utf-8') for elem in pdata])
        printtable.append([elem.decode('utf-8').strip() for elem in data])
    return printtable

def find_all_datatables(page, all_divtables):
    logging.debug('In a function : find_all_datatables')
    count = 0
    allheaders=[]
    for table in all_divtables:
        bluetables = []
        trs = table.find_all('tr')
        for tr in trs:
            global flagtr
            if checktag(str(tr.get('style'))) == "true" or checktag(str(tr)) == "true":
                logging.debug('Checking data tables at Row Level')
                bluetables = printtable(tr.find_parent('table'))
                break
            else:
                tds = tr.find_all('td')
                for td in tds:
                    if checktag(str(td.get('style'))) == "true" or checktag(str(td)) == "true":
                        logging.debug('Checking data tables at Column Level')
                        bluetables = printtable(td.find_parent('table'))
                        break
            if not len(bluetables) == 0:
                break
        if not len(bluetables) == 0:
            logging.debug('Total Number of data tables to be created {}'.format(len(bluetables)))
            count += 1
            ptag=table.find_previous('p');
            while ptag is not None and checkheadertag(ptag.get('style'))=="false" and len(ptag.text)<=1:
                ptag=ptag.find_previous('p')
                if checkheadertag(ptag.get('style'))=="true" and len(ptag.text)>=2:
                    global name
                    name=re.sub(r"[^A-Za-z0-9]+","",ptag.text)
                    if name in allheaders:
                        hrcount+=1
                        hrname=name+"_"+str(hrcount)
                        allheaders.append(hrname)
                    else:
                        hrname=name
                        allheaders.append(hrname)
                        break
            folder_name = foldername(page)
            logging.debug('folder created with folder Name{}'.format(folder_name))
            path = str(os.getcwd()) + "/" + folder_name
            logging.debug('Path for csv creation {}'.format(path))
            assure_path_exists(path)
            if(len(allheaders)==0):
                filename=folder_name+"-"+str(count)
            else:
                filename=allheaders.pop()
            csvname=filename+".csv"
            logging.debug('file creation Name{}'.format(csvname))
            csvpath = path + "/" + csvname
            os.path.abspath(csvpath)
            print(csvpath)
            logging.debug('CSV Path for the file creation {}'.format(csvpath))
            with open(csvpath, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(bluetables)
            zip_dir(path)
        

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
        bucket_name = 'adsassignment1part1'+str(st).replace(" ", "").replace("-", "").replace(":","").replace(".","")
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
    cik =''
    accessionNumber=''
    accessKey=''
    secretAccessKey=''
    inputLocation=''
  
    for i in range(1,argLen):
        val=sys.argv[i]
        if val.startswith('cik='):
            pos=val.index("=")
            cik=val[pos+1:len(val)]
            continue
        elif val.startswith('accessionNumber='):
            pos=val.index("=")
            accessionNumber=val[pos+1:len(val)]
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
    
    print("CIK=",cik)
    print("Accession Number=",accessionNumber)
    print("Access Key=",accessKey)
    print("Secret Access Key=",secretAccessKey)
    print("Location=",inputLocation)
          
    logfilename = 'log_Edgar_'+ cik + '_' + st + '.txt' 
    logging.basicConfig(filename=logfilename, level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Program Start')
    logging.debug('*************')
    logging.debug('Calling the initial URL with CIK Number {} and Accession number {}'.format(cik, accessionNumber))
    nameOfFolder=get_url(cik, accessionNumber)
    
    filepaths = []
    filepaths.append(os.path.join(logfilename))
    filepaths.append(os.path.join(nameOfFolder + '.zip'))
    
    logging.info('Compiled csv and log file zipped')
    
    upload_to_s3(accessKey,secretAccessKey,inputLocation,filepaths)


if __name__ == '__main__':
    main()