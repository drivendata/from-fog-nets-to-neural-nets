# Date: May, 2016
# DrivenData.org Fog Net competition solution

import pandas
import numpy as np
import datetime


# -------------------------------------------------------------------------------------------------------------- #
# Python newbie apologies for duplicating training & testing code
# -------------------------------------------------------------------------------------------------------------- #


# ---------------------------------------- Read the raw data files --------------------------------------------- #

Raw_Micro_5min_train = pandas.read_csv('./data/raw/Training set Microclimate (5 minute intervals).csv',index_col=False)
Raw_Micro_2hr_train = pandas.read_csv('./data/raw/Training set Microclimate (2 hour intervals).csv',index_col=False)
Raw_Micro_5min_test = pandas.read_csv('./data/raw/Test set Microclimate (5 min intervals).csv',index_col=False)
Raw_Micro_2hr_test = pandas.read_csv('./data/raw/Test set Microclimate (2 hour intervals).csv',index_col=False)
Raw_Yield = pandas.read_csv('./data/raw/Target Variable Water Yield.csv',index_col=False)
Raw_Si = pandas.read_csv('./data/raw/Macrocliamte Sidi Ifni Weather Station.csv', index_col=False)
Raw_Ag = pandas.read_csv('./data/raw/Macroclimate Agadir Airport.csv', index_col=False)
Raw_Gu = pandas.read_csv('./data/raw/Macroclimate Guelmim Airport.csv', index_col=False)
submission_format = pandas.read_csv('./data/raw/submission_format.csv', index_col=False)

DateFormat='%Y-%m-%d %H:%M:%S'
Raw_Micro_5min_train.rename(columns={Raw_Micro_5min_train.columns[0]:'DateTime'},inplace=True)
Raw_Micro_5min_train.DateTime = pandas.to_datetime(Raw_Micro_5min_train.DateTime, format=DateFormat)
Raw_Micro_5min_test.rename(columns={Raw_Micro_5min_test.columns[0]:'DateTime'},inplace=True)
Raw_Micro_5min_test.DateTime = pandas.to_datetime(Raw_Micro_5min_test.DateTime, format=DateFormat)
Raw_Si.rename(columns={Raw_Si.columns[0]:'DateTime'},inplace=True)
Raw_Si.DateTime = pandas.to_datetime(Raw_Si.DateTime, format=DateFormat)
Raw_Ag.rename(columns={Raw_Ag.columns[0]:'DateTime'},inplace=True)
Raw_Ag.DateTime = pandas.to_datetime(Raw_Ag.DateTime, format=DateFormat)
Raw_Gu.rename(columns={Raw_Gu.columns[0]:'DateTime'},inplace=True)
Raw_Gu.DateTime = pandas.to_datetime(Raw_Gu.DateTime, format=DateFormat)
submission_format.rename(columns={submission_format.columns[0]:'DateTime'},inplace=True)
submission_format.DateTime = pandas.to_datetime(submission_format.DateTime, format=DateFormat)

Raw_Micro_2hr_train.rename(columns={Raw_Micro_2hr_train.columns[0]:'DateTime'},inplace=True)
Raw_Micro_2hr_train.DateTime = pandas.to_datetime(Raw_Micro_2hr_train.DateTime, format=DateFormat)
Raw_Micro_2hr_test.rename(columns={Raw_Micro_2hr_test.columns[0]:'DateTime'},inplace=True)
Raw_Micro_2hr_test.DateTime = pandas.to_datetime(Raw_Micro_2hr_test.DateTime, format=DateFormat)
#Raw_Si.rename(columns={Raw_Si.columns[0]:'DateTime'},inplace=True)
#Raw_Si.DateTime = pandas.to_datetime(Raw_Si.DateTime, format=DateFormat)
#Raw_Ag.rename(columns={Raw_Ag.columns[0]:'DateTime'},inplace=True)
#Raw_Ag.DateTime = pandas.to_datetime(Raw_Ag.DateTime, format=DateFormat)
#Raw_Gu.rename(columns={Raw_Gu.columns[0]:'DateTime'},inplace=True)
#Raw_Gu.DateTime = pandas.to_datetime(Raw_Gu.DateTime, format=DateFormat)

Raw_Yield.rename(columns={Raw_Yield.columns[0]:'DateTime'},inplace=True)
Raw_Yield.DateTime = pandas.to_datetime(Raw_Yield.DateTime, format=DateFormat)

# --------------------------------- Limit to the columns I used ------------------------------------------------- #

cols_micro = ['DateTime', 'humidity', 'temp', 'leafwet_lwscnt', 'wind_dir', 'wind_ms']
cols_Si = ['DateTime', 'T', 'Td', 'DD', 'Nh', 'RRR', 'E']
cols_Ag = ['DateTime', 'T', 'Td', 'P0', 'c']
cols_Gu = ['DateTime', 'T', 'Td', 'DD', 'Ff', 'c']

# --------------------------------- Join yield to training data ------------------------------------------------- #
# Inner join on micro to keep just the good stuff
Micro_2hr_train = pandas.merge(Raw_Yield, Raw_Micro_2hr_train[cols_micro], on='DateTime', how='inner').sort_values('DateTime')
# Outer join on macro for a lot of imputing
SA = pandas.merge(Raw_Si[cols_Si], Raw_Ag[cols_Ag], on='DateTime', how='outer', suffixes=('_Si','_Ag')).sort_values('DateTime')
SAG = pandas.merge(SA, Raw_Gu[cols_Gu], on='DateTime', how='outer', suffixes=('','_Gu')).sort_values('DateTime')
SAG.rename(columns={'DD':'DD_Si','Nh':'Nh_Si','RRR':'RRR_Si','E':'E_Si','P0':'P0_Ag','c':'c_Ag','T':'T_Gu','Td':'Td_Gu','Ff':'Ff_Gu'},inplace=True)
Raw_SAG = pandas.merge(Raw_Yield, SAG, on='DateTime', how='outer').sort_values('DateTime')


# --------------------------------- Omit missing values, outliers ----------------------------------------------- #
# I didn't train on leafwet_lwscnt > 1000, or humidity == 0
Micro_2hr_train = Micro_2hr_train[Micro_2hr_train['leafwet_lwscnt']<1001]
Micro_2hr_train = Micro_2hr_train[Micro_2hr_train['humidity']>0]

# --------------------------------- Move 9 input rows from 5-min training to 2-hour testing --------------------- #
# Use 5-minute stale training data at beginning of each test interval (rather than macro data)
# How to handle this operationally would depend on how often switching occurs
dates = ['2014-01-22 23:55:00', '2014-03-23 23:55:00', '2014-05-22 23:55:00',
        '2014-07-21 23:55:00', '2014-09-19 23:55:00', '2014-12-28 23:55:00',
        '2015-02-26 23:55:00', '2015-08-25 23:55:00', '2015-10-24 23:55:00']
PreMidnight = [datetime.datetime.strptime(date, DateFormat) for date in dates]
dates = ['2014-01-23 00:00:00', '2014-03-24 00:00:00', '2014-05-23 00:00:00',
        '2014-07-22 00:00:00', '2014-09-20 00:00:00', '2014-12-29 00:00:00',
        '2015-02-27 00:00:00', '2015-08-26 00:00:00', '2015-10-25 00:00:00']
Midnight = [datetime.datetime.strptime(date, DateFormat) for date in dates]
MidnightHours = pandas.DataFrame({'Midnight':Midnight, 'PreMidnight':PreMidnight})

merged_9 = pandas.merge(MidnightHours, Raw_Micro_5min_train[cols_micro], left_on='PreMidnight', right_on='DateTime')
merged_9.drop(['PreMidnight','DateTime'], axis=1, inplace=True)
merged_9.rename(columns={'Midnight':'DateTime'},inplace=True)
Micro_test_1119 = pandas.concat([merged_9, Raw_Micro_2hr_test[cols_micro]]).sort_values('DateTime')

# Save Raw_Micro_5min_train[Raw_Micro_5min_train.leafwet_lwscnt.notnull()]


# --------------------------------- Derive some columns, save --------------------------------------------------- #
# These cuts were selected by look at scatterplots, especially leafwet_lwscnt x humidity, colored by yield
Micro_2hr_train['THL'] = 'M'
Micro_2hr_train.loc[(Micro_2hr_train['temp']>18) | (Micro_2hr_train['temp']<2) | (Micro_2hr_train['leafwet_lwscnt']>680) | (Micro_2hr_train['humidity']<0.6), 'THL'] = 'dry'
Micro_2hr_train.loc[(Micro_2hr_train['THL']=='M') & (Micro_2hr_train['humidity']>0.953), 'THL'] = 'VH'
Micro_2hr_train.loc[(Micro_2hr_train['THL']=='M') & (Micro_2hr_train['humidity']>0.9), 'THL'] = 'H'
Micro_2hr_train['WindDir_shifted'] = (Micro_2hr_train['wind_dir'] + 270) % 360 # rotate the compass

Micro_test_1119['THL'] = 'M'
Micro_test_1119.loc[(Micro_test_1119['temp']>18) | (Micro_test_1119['temp']<2) | (Micro_test_1119['leafwet_lwscnt']>680) | (Micro_test_1119['humidity']<0.6), 'THL'] = 'dry'
Micro_test_1119.loc[(Micro_test_1119['THL']=='M') & (Micro_test_1119['humidity']>0.953), 'THL'] = 'VH'
Micro_test_1119.loc[(Micro_test_1119['THL']=='M') & (Micro_test_1119['humidity']>0.9), 'THL'] = 'H'
Micro_test_1119['WindDir_shifted'] = (Micro_test_1119['wind_dir'] + 270) % 360 # rotate the compass

Micro_2hr_train.to_csv('./data/interim/Micro_2hr_train.csv', index=False)
Micro_test_1119.to_csv('./data/interim/Micro_test_1119.csv', index=False)

# Add rows for submission times and any with yield
Extras = pandas.merge(submission_format[['DateTime']], Micro_2hr_train[['DateTime']], on='DateTime', how='outer').sort_values('DateTime')
Raw_SAG_extended = pandas.merge(Raw_SAG, Extras, on='DateTime', how='outer').sort_values('DateTime')
Raw_SAG_extended.to_csv('./data/interim/Raw_SAG.csv', index=False)
