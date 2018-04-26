# -*- coding: utf-8 -*-
"""
Author: Ashkan Ebadi
Date:   April 25, 2018

Description: Exploratory Analysis on Flight Global Dataset
"""

### Libraries
import sys
sys.path.append(r'C:/Anaconda/Lib/site-packages') # add the path to anaconda libraries

import os
os.chdir('C:/Global Sales/Datasets/fg_data_set')  # set the default working directory

import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime

import matplotlib.pyplot as plt
###

########## Functions
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print '\r%s |%s| %s%% \tScore: %s\r' % (prefix, bar, percent, suffix),
    
########## 
    
    
    
################### Loading Datasets ##################     336,383 x 298▌
start_time = time.time()
fg_data = pd.read_csv('flight_global_aircraft_details.csv', sep=',', header = 0, low_memory=False, error_bad_lines=False, quoting=csv.QUOTE_ALL, index_col=False, encoding = 'latin-1')

print("======== Flight Global loaded in %.2f seconds =========" %(time.time() - start_time))
#######################################################


#pd.DataFrame(fg_data.describe()).to_csv('fg_data_summary.csv')

#### renaming the column names
col_names = []
for item in list(fg_data.columns.values):
    item = item.replace("flight_global_aircraft_details.", "")
    print(item)
    col_names.append(item)

fg_data.columns = col_names
####


############################### Date Fields Treatment ###########################################
#### Aircraft Status, remove rows for which status is either 'cancelled' or 'LOI to order'      --->  285,068 x 298
fg_data.loc[:, 'status'].unique()
fg_data = fg_data.loc[~fg_data['status'].isin(["Cancelled", "LOI to Order"]) ]
####

#### Field names on focus: 'build_year', 'in_service_date', 'delivery_date', 'order_date'
nat = np.datetime64('NaT')

fg_data['in_service_date'].dtype
len(fg_data['in_service_date'].unique())

fg_data = fg_data.loc[fg_data['build_year'] >= 1960]       ### buil_year >= 1960    --->  259,723
fg_data = fg_data.loc[fg_data['build_year'] <= 2018]       ### buil_year <= 2018    --->  235,516

fg_data['in_service_date'] = pd.to_datetime(fg_data['in_service_date'], format='%d-%m-%Y', errors = 'coerce')
fg_data['in_service_date'] = fg_data['in_service_date'].dt.date
fg_data['in_service_date'][1:3]
fg_data = fg_data.loc[(fg_data['in_service_date'] == nat) | (fg_data['in_service_date'] >= datetime(1960, 1, 1).date())]       ### in_service_date >= 1960    --->  216,756
fg_data = fg_data.loc[(fg_data['in_service_date'] == nat) | (fg_data['in_service_date'] <= datetime(2020, 1, 1).date())]       ### in_service_date <= 2020    --->  216,239


fg_data['delivery_date'] = pd.to_datetime(fg_data['delivery_date'], format='%d-%m-%Y', errors = 'coerce')
fg_data['delivery_date'] = fg_data['delivery_date'].dt.date
fg_data['delivery_date'][1:3]
fg_data = fg_data.loc[(fg_data['delivery_date'] == nat) | (fg_data['delivery_date'] >= datetime(1960, 1, 1).date())]       ### delivery_date >= 1960    --->  206,213
fg_data = fg_data.loc[(fg_data['delivery_date'] == nat) | (fg_data['delivery_date'] <= datetime(2020, 1, 1).date())]       ### delivery_date <= 2020    --->  206,213


fg_data['order_date'] = pd.to_datetime(fg_data['order_date'], format='%d-%m-%Y', errors = 'coerce')
fg_data['order_date'] = fg_data['order_date'].dt.date
fg_data['order_date'][1:3]
fg_data = fg_data.loc[(fg_data['order_date'] == nat) | (fg_data['order_date'] >= datetime(1955, 1, 1).date())]       ### order_date >= 1955    --->  204,394
fg_data = fg_data.loc[(fg_data['order_date'] == nat) | (fg_data['order_date'] < datetime(2019, 1, 1).date())]       ### order_date < 2019    --->  204,394
###################################################################################################



#################################### Only Pratt & Whitney Engines #################################
# Field name on focus: 'engine_manufacturer'

# explore the engine manufacturers 
len(fg_data['engine_manufacturer'].unique())  ## 55 distinct engine manufacturers
#for item in fg_data['engine_manufacturer'].unique():
#    print(item)
    
oem_count_df = fg_data[['manufacturer', 'type', 'registration', 'engine_family', 'engine_manufacturer']].groupby(['engine_manufacturer']).agg(['count'])
oem_count_df.columns = ['rows_count', 'type_count', 'registration_count', 'engine_family_count']

oem_count_df = oem_count_df.sort_values('rows_count', ascending = False)
oem_count_df = oem_count_df[0:15]

ax = oem_count_df.plot(kind='bar', title ="Flight Global - OEM Counts", figsize=(15, 10), legend=True, fontsize=12, color = ['orangered', 'navy', 'grey', 'wheat'])
#ax.set_xlabel("OEMs", fontsize=12)
#ax.set_ylabel("V", fontsize=12)
plt.grid()
plt.show()
#

# filter flight global data to contain only "Pratt & Whitney Canada" engines
fg_data = fg_data.loc[fg_data['engine_manufacturer'] == 'Pratt & Whitney Canada']       ###  engine_manufacturer == PWC --->  36,356

oem_count_df = fg_data[['manufacturer', 'type', 'registration', 'engine_family']].groupby(['manufacturer']).agg(['count'])
oem_count_df.columns = ['type_count', 'registration_count', 'engine_family_count']

oem_count_df = oem_count_df.sort_values('registration_count', ascending = False)
oem_count_df = oem_count_df[0:15]

ax = oem_count_df.plot(kind='bar', title ="Flight Global - Manufacturers, PWC Engines", figsize=(15, 10), legend=True, fontsize=12, color = ['orangered', 'navy', 'wheat'])
ax.set_xlabel("Manufacturers - PWC Engines", fontsize=12)
plt.grid()
plt.show()
###################################################################################################











#fg_data.groupby('build_year').age.plot(kind='kde')
fg_data.build_year.plot(kind='kde')
fg_data.build_year.hist()





#### Operators
len(fg_data.loc[:, 'serial_number'].unique())







fg_data.iloc[:,1:5].describe()
print(fg_data.shape)
fg_data.info()        ### memory usage: 777.4 MB

fg_data.describe(include=['float64'])


fg_data[(fg_data['Churn'] == 0) & (df['International plan'] == 'No')]['average_annual_hours'].max()







