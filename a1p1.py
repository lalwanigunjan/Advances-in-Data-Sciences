#importing libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import re
import os

#input
def incheck(arg):
    default = '0000051143-13-000007'
    if re.match('[0-9]{10}-[0-9]{2}-[0-9]{6}',arg):
        return inp
    else:
        return default

avalid = input("Enter a value:")
acc_no = incheck(avalid)

# Generation of URL using acc_no

url = 'http://www.sec.gov/Archives/edgar/data/'
parts = acc_no.split('-')
cik = parts[0]
year = parts[1]
no = parts[2]
CIK = cik.lstrip('0')
link = url+CIK+'/'+cik+year+acc_no+'/'+acc_no+'-index.html'
print(link)

#Getting Html Document list
request = requests.get(link)
html_doc = request.text

#Getting 10q tag#
soup = BeautifulSoup(html_doc,'lxml')
a_tags = soup.find_all('a')

for a_tag in a_tags:
    if a_tag.get('href').endswith('_10q.htm'):
        print(a_tag.get('href'))
        link10q = ('https://www.sec.gov'+ a_tag.get('href'))
request1 = requests.get(link10q)
html_doc1 = request1.text
soup1 = BeautifulSoup(html_doc1,'lxml')


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


#make directory
if not os.path.exists('extracted_csvs'):
    os.makedirs('extracted_csvs')


i =0
# Transfering data to csv file

pd.set_option('display.max_colwidth', -1)

clean_tables_df = pd.DataFrame(clean_tables)
for row in clean_tables_df.itertuples():
    print (str(row) )
    calls_df = pd.read_html(str(row),header = 0)
    my_df = pd.DataFrame(calls_df)
    file = cik+year+acc_no+str(i)+'.csv'
    with open(os.path.join('extracted_csvs', file),'w') as outfile:
        my_df.to_csv(outfile,index = False, header = 0,sep =',',encoding = 'utf-8')
    i = i + 1
