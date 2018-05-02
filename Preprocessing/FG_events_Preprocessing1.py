# -*- coding: utf-8 -*-
"""
Author: Ashkan Ebadi
Date:   April 27, 2018

Description: Preprocessing and minor Exploratory Analysis on Flight Global Events Dataset
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
import seaborn as sns
###

########## Functions
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% \tScore: %s\r' % (prefix, bar, percent, suffix),)    

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

def null_cols(df):
    # gives some infos on columns types and number of null values
    cols_null_summary = pd.DataFrame(df.dtypes).T.rename(index={0:'Column Type'})
    cols_null_summary = cols_null_summary.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'Count of Null Values'}))
    cols_null_summary = cols_null_summary.append(pd.DataFrame(df.isnull().sum() / df.shape[0])
                             .T.rename(index={0:'Percentage of Null Values'}))
    cols_null_summary.loc['Percentage of Null Values'] = (cols_null_summary.loc['Percentage of Null Values'].apply('{:.2%}'.format))
    
    cols_null_summary = cols_null_summary.sort_values('Count of Null Values', ascending = False, axis = 1)

    return cols_null_summary
########## 
    
##############################################################################
######################### RUN if the first time! #############################
################# Reading in chunks ##########################################
tp = pd.read_csv(filepath_or_buffer='flight_global_aircraft_events_details.csv',sep=',', nrows = 1) 
header_row = list(tp.columns)
#header_row

## names=header_row,
### gives TextFileReader, which is iteratable with chunks of 1000 rows.
#tp = pd.read_csv(filepath_or_buffer='flight_global_aircraft_events_details.csv',names=header_row, sep=',',na_values='.',header=None, iterator=True, chunksize=1000) 
### df is DataFrame. If error do list(tp)
#df = pd.concat(list(tp), ignore_index=True) 
### if version 3.4, use tp
## df = pd.concat(tp, ignore_index=True)


tp = pd.read_csv(filepath_or_buffer='flight_global_aircraft_events_details.csv',sep=',', nrows = 1) 
header_row = list(tp.columns)

reader = pd.read_csv(filepath_or_buffer='flight_global_aircraft_events_details.csv',names=header_row, sep=',',na_values='.',header=None, iterator=True, chunksize=1000)
progress = 0
lines_number = 1986503
#lines_number = sum(1 for line in open('flight_global_aircraft_events_details.csv'))
for chunk in reader:
    if progress == 0:
        result = chunk.loc[chunk['flight_global_aircraft_events_details.current_engine_manufacturer'] == "Pratt & Whitney Canada"]
    else:
        temp = chunk.loc[chunk['flight_global_aircraft_events_details.current_engine_manufacturer'] == "Pratt & Whitney Canada"]
        result = result.append(temp, ignore_index=True)
    progress += 1000
    print(int(round(float(progress)/lines_number * 100)), "%")


result['flight_global_aircraft_events_details.current_engine_manufacturer'].head()    
    

#################### writing the filtered fg_data to the disk #•############ 357,783 X 455 #######################
# result.to_csv('C:/Global Sales/Datasets/fg_data_set/flight_global_PWC_aircraft_events_details.csv', index = False)
#################################################################################################################
##############################################################################
##############################################################################

############### Do this instead of reading in chunks, if not the first time ########################
result = pd.read_csv('C:/Global Sales/Datasets/fg_data_set/flight_global_PWC_aircraft_events_details.csv', sep=',', header = 0, low_memory=False, error_bad_lines=False, index_col=False)
##############################################################################


#### renaming the column names
col_names = []
for item in list(result.columns.values):
    item = item.replace("flight_global_aircraft_events_details.", "")
    print(item)
    col_names.append(item)

result.columns = col_names
####


################################ Check the percentage of NULL values and remove columns that are 100% NULL ##############################
cols_null_summary = null_cols(result)

null_cols = []
for i in range(0, len(result.columns)):
    if (cols_null_summary.iloc[2, i] == '100.00%'):
        null_cols.append(cols_null_summary.columns[i])

# write to csv
# cols_null_summary.to_csv('C:/Global Sales/Results/FG/FG_events_null_summary.csv', index = False)             # write the result dataframe

len(null_cols)    ### 50 column that are 100% NULL
result = result[result.columns[~result.columns.isin(null_cols)]]   ### drop NULL columns --->  357,783 x 405
##########################################################################################################################################



############################### Date Fields Treatment ###########################################
## list the date fields
date_cols = []
for col in result.columns:
    if col.find("date") > 0:
        date_cols.append(col)
        
for col in result.columns:
    if col.find("year") > 0:
        print(col)#date_cols.append(col)
##
        
#### Aircraft Status, remove rows for which status is either 'cancelled' or 'LOI to order' or 'LOI to Option'     --->  334,003 x 405
result.loc[:, 'current_status'].unique()
result = result.loc[~result['current_status'].isin(["Cancelled", "LOI to Order", "LOI to Option"]) ]
####

#### Field names on focus: 'build_year', 'event_date', 'delivery_date', 'first_flight_date', 'order_date', 'current_hours_and_cycles_date'
nat = np.datetime64('NaT')

result['build_year'].dtype
len(result['build_year'].unique())  # 111 unique build years! including nan

result['build_year'] = pd.to_numeric(result['build_year'], errors = 'coerce')
result = result.loc[result['build_year'] >= 1960]       ### build_year >= 1960    --->  331,887 X 405
result = result.loc[result['build_year'] <= 2018]       ### build_year <= 2018    --->  330,168 X 405

result['event_date'] = pd.to_datetime(result['event_date'], format='%d/%m/%Y', errors = 'coerce')
result['event_date'] = result['event_date'].dt.date
# result['event_date'].min()   max()        event_date range is fine     01/01/1962 to 31/12/2016

result['delivery_date'] = pd.to_datetime(result['delivery_date'], format='%d/%m/%Y', errors = 'coerce')
result['delivery_date'] = result['delivery_date'].dt.date
result['delivery_date'][1:3]
result = result.loc[(result['delivery_date'] == nat) | (result['delivery_date'] >= datetime(1960, 1, 1).date())]       ### delivery_date >= 1960    --->  325,348 X 405
result = result.loc[(result['delivery_date'] == nat) | (result['delivery_date'] <= datetime(2020, 1, 1).date())]       ### delivery_date <= 2020    --->  325,326 X 405

result['first_flight_date'] = pd.to_datetime(result['first_flight_date'], format='%d/%m/%Y', errors = 'coerce')
# result['first_flight_date'].max()      it's fine, 1960-10-21 to 2017-08-28
result['first_flight_date'] = result['first_flight_date'].dt.date
result['first_flight_date'][1:30]

result['current_hours_and_cycles_date'] = pd.to_datetime(result['current_hours_and_cycles_date'], format='%d/%m/%Y', errors = 'coerce')
result['current_hours_and_cycles_date'] = result['current_hours_and_cycles_date'].dt.date
result['current_hours_and_cycles_date'][1:30]


result['order_date'] = pd.to_datetime(result['order_date'], format='%d/%m/%Y', errors = 'coerce')
result['order_date'] = result['order_date'].dt.date
result['order_date'][1:3]
result = result.loc[(result['order_date'] == nat) | (result['order_date'] >= datetime(1955, 1, 1).date())]       ### order_date >= 1955    --->  322,633 X 405
result = result.loc[(result['order_date'] == nat) | (result['order_date'] < datetime(2019, 1, 1).date())]       ### order_date < 2019      --->  322,633 X 405
###################################################################################################



#################################### PWC - OEM registration distribution #################################
oem_count_df = result[['current_manufacturer', 'current_engine_master_series', 'registration', 'current_engine_family']].groupby(['current_manufacturer']).agg(['count'])
oem_count_df.columns = ['master_series_count', 'registration_count', 'engine_family_count']

oem_count_df = oem_count_df.sort_values('registration_count', ascending = False)
oem_count_df = oem_count_df[0:15]

ax = oem_count_df.plot(kind='bar', title ="Flight Global Events (PWC) - Registration Counts", figsize=(15, 10), legend=True, fontsize=12, color = ['orangered', 'navy', 'grey', 'wheat'])
plt.grid()
##################################################################################################


################################ engine_family vs build_year #####################################
fg_pwc_engfamily_buildyr = result.groupby(["current_engine_family", "build_year"]).size().reset_index(name="Count")
print("Number of Distinct PWC's Engine Families: %d" %len(result['current_engine_family'].unique()))        # 9 distinct PWC engine families

fg_pwc_engfamily_buildyr = fg_pwc_engfamily_buildyr.pivot("current_engine_family", "build_year", "Count")
fg_pwc_engfamily_buildyr = fg_pwc_engfamily_buildyr.fillna(0)

# including all 9 engine_families
ax = sns.heatmap(fg_pwc_engfamily_buildyr, linewidths=.5, cmap="BuPu")
##################################################################################################


####################################################################################
################################ Keep only PT6 #####################################
result['current_engine_family'].unique()
result = result.loc[result['current_engine_family'] == 'PT6']             # only PT6   --->   195,340 x 405

fg_pwc_engseries_buildyr = result.groupby(["current_engine_series", "build_year"]).size().reset_index(name="Count")
print("Number of Distinct PWC-PT6 Engine Series: %d" %len(result['current_engine_series'].unique()))        # 36 distinct PWC-PT6 engine series

fg_pwc_engseries_buildyr = fg_pwc_engseries_buildyr.pivot("current_engine_series", "build_year", "Count")
fg_pwc_engseries_buildyr = fg_pwc_engseries_buildyr.fillna(0)

ax = sns.heatmap(fg_pwc_engseries_buildyr, linewidths=.5, cmap="BuPu")
####################################################################################


######################### Stats on PT6 #################################
# Creation of a dataframe with statitical infos on each airline:
simple_stats = result['build_year'].groupby(result['current_engine_series']).apply(get_stats).unstack()
simple_stats = simple_stats.sort_values('count', ascending = 0)
simple_stats.to_csv('C:/Global Sales/Results/FG/FG_events_PT6_summary.csv', index = False)             # write the result dataframe

#------------------------------------------------------
# striplot with all the values reported for the delays
#------------------------------------------------------
# I redefine the colors for correspondance with the pie charts
colors = ['firebrick', 'gold', 'lightcoral', 'aquamarine', 'c', 'yellowgreen', 'grey',
          'seagreen', 'tomato', 'violet', 'wheat', 'chartreuse', 'lightskyblue', 'royalblue']
fig = plt.figure(1, figsize=(10,7))
ax3 = sns.stripplot(y="current_engine_series", x="build_year", size = 4, palette = colors,
                    data=result, linewidth = 0.5,  jitter=True)
#------------------------------------------------------
# Barplot for PT6-EngineSeries, Status Count
#------------------------------------------------------
fig = plt.figure(1, figsize=(10,7))
colors = ["red", "aquamarine", "yellowgreen", 'grey', 'firebrick'] 
ax = sns.countplot(y="engine_series", hue='status', data=fg_data, palette=colors)
plt.xlabel('PT6 - Engine Series, Status Count', fontsize=16, weight = 'bold', labelpad=10)
#------------------------------------------------------
# Distribution plot of PT6 over time
#------------------------------------------------------
fig = plt.figure(1, figsize=(10,7))
fg_data.build_year.plot(kind='kde')
plt.xlabel('PT6 Distribution over Time', fontsize=16, weight = 'bold', labelpad=10)
#------------------------------------------------------
# PT6 build_year histogram
#------------------------------------------------------
fig = plt.figure(1, figsize=(10,7))
fg_data.build_year.hist()
plt.xlabel('PT6 Histogram', fontsize=16, weight = 'bold', labelpad=10)
########################################################################


#################### writing the filtered fg_data to the disk #####################################
fg_data.to_csv('C:/Global Sales/Datasets/fg_data_set/FG_aircraft_details_filtered.csv', index = False)
###################################################################################################














################################ Check columns wigh more 50%  values ##############################
percent50_cols = []
for i in range(0, len(result.columns)):
    if (cols_null_summary.iloc[1, i] <= 178892):
        percent50_cols.append(cols_null_summary.columns[i])

cols_null_summary.to_csv('C:/Global Sales/Results/FG/FG_events_null_summary.csv', index = False)             # write the result dataframe

len(percent50_cols)    ### 101 column that have more than 50% values

for item in percent50_cols:
    print(item)
   
##########################################################################################################################################




    


#filename = "flight_global_aircraft_events_details.csv"
#
#lines_number = sum(1 for line in open(filename))
#print lines_number
#lines_in_chunk = 5000 # I don't know what size is better
#counter = 0
#completed = 0
#reader = pd.read_csv(filename, chunksize=lines_in_chunk, sep=',', header = 0, engine='c', error_bad_lines=False, quoting=csv.QUOTE_ALL, index_col=False, encoding = 'latin-1')
#for chunk in reader:
#    if counter == 0:
#        result = chunk.loc[chunk['flight_global_aircraft_events_details.current_engine_manufacturer'] == "Pratt & Whitney Canada"]
#    else:
#        temp = chunk.loc[chunk['flight_global_aircraft_events_details.current_engine_manufacturer'] == "Pratt & Whitney Canada"]
#        result = result.append(temp, ignore_index=True)
#        
#    #result.to_csv('./DSVM/fg_data_set/chunk_fg.csv', index=False, header=False, mode='a')
#    #result_df = pd.concat([x,y], ignore_index=True)
#    # showing progress:
#    counter += lines_in_chunk
#    new_completed = int(round(float(counter)/lines_number * 100))
#    if (new_completed > completed): 
#        completed = new_completed
#        print "Completed", completed, "%"


    
    
    





###################################################################################################
################################ Exploratory Analysis #############################################
###################################################################################################












###### memory usage #########################
fg_data.iloc[:,1:5].describe()
print(fg_data.shape)
fg_data.info()        ### memory usage: 777.4 MB dropped to 44.8+ MB
############################################
#fg_data.describe(include=['float64'])







