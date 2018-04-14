import pandas as pd
import numpy as np
import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection
import time
import datetime
import os
import boto.s3
import sys
import boto3
import threading
import logging


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

def upload_to_s3(Inputlocation,Access_key,Secret_key):            
    print("Uploading files to amazon")
    try:

        buck_name="ads-assignment3-models"

        S3_client = boto3.client('s3',Inputlocation,aws_access_key_id= Access_key, aws_secret_access_key= Secret_key)
    
        if Inputlocation == 'us-east-1':
            S3_client.create_bucket(Bucket=buck_name)
        else:
            S3_client.create_bucket(Bucket=buck_name,CreateBucketConfiguration={'LocationConstraint': Inputlocation})

        print("connection successful")
        S3_client.upload_file("AllModels.zip", buck_name,"AllModels.zip"),
        Callback=ProgressPercentage("AllModels.zip")
    
        print("Files uploaded successfully")
    
    except Exception as e:
        print("Error uploading files to Amazon s3" + str(e))
        
def main():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    
    argLen=len(sys.argv)
    Access_key=''
    Secret_key=''
    Inputlocation=''

    for i in range(1,argLen):
        val=sys.argv[i]
        if val.startswith('accessKey='):
            pos=val.index("=")
            Access_key=val[pos+1:len(val)]
            continue
        elif val.startswith('secretKey='):
            pos=val.index("=")
            Secret_key=val[pos+1:len(val)]
            continue
        elif val.startswith('location='):
            pos=val.index("=")
            Inputlocation=val[pos+1:len(val)]
            continue

    print("Access Key=",Access_key)
    print("Secret Access Key=",Secret_key)
    print("Location=",Inputlocation)        

    upload_to_s3(Inputlocation,Access_key,Secret_key)
    print('files uploaded')

if __name__ == '__main__':
    main()

