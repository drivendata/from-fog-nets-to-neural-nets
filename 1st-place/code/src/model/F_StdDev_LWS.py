# Date: May, 2016
# DrivenData.org Fog Net competition solution

import pandas as pd
import numpy as np
import datetime

# -------------------------------------------------------------------------------------------------------------- #
# There were several irregularities in how the std. dev. of leafwet_lwscnt was computed. It's all historical
# noise and should be simplified to using the last 2-hour window of 5-minute data (except when that data is not
# available).

# Python newbie apologies for all transgressions, like handling of index columns
# -------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------ Read the raw data files ------------------------------------------- #

Raw_Micro_5min_train = pd.read_csv('./data/raw/Training set Microclimate (5 minute intervals).csv',index_col=False)
Raw_Micro_5min_test = pd.read_csv('./data/raw/Test set Microclimate (5 min intervals).csv',index_col=False)
Raw_Micro_5min_train.rename(columns={Raw_Micro_5min_train.columns[0]:'DateTime'},inplace=True)
Raw_Micro_5min_test.rename(columns={Raw_Micro_5min_test.columns[0]:'DateTime'},inplace=True)

DateFormat='%Y-%m-%d %H:%M:%S'
Raw_Micro_5min_train.DateTime = pd.to_datetime(Raw_Micro_5min_train.DateTime, format=DateFormat)
Raw_Micro_5min_test.DateTime = pd.to_datetime(Raw_Micro_5min_test.DateTime, format=DateFormat)

Micro_2hr_train = Raw_Micro_5min_train

Micro_2hr_train['Train'] = True
Raw_Micro_5min_test['Train'] = False
Micro_5min_both = pd.concat([Micro_2hr_train, Raw_Micro_5min_test], axis=0).sort_values('DateTime')

Micro_5min_both.reset_index(drop=True, inplace=True)
Micro_5min_both.DateTime = pd.to_datetime(Micro_5min_both.DateTime, format=DateFormat) # necessary?

# ------------------------------------------ Derive some columns ----------------------------------------------- #
Micro_5min_both['Hour'] = Micro_5min_both['DateTime'].dt.hour

Micro_5min_both['Year'] = Micro_5min_both['DateTime'].dt.year
Micro_5min_both['Month'] = Micro_5min_both['DateTime'].dt.month
Micro_5min_both['Day'] = map(lambda x: int(x), Micro_5min_both['DateTime'].dt.strftime('%j'))
Micro_5min_both.loc[(Micro_5min_both['DateTime'].dt.hour==0) & (Micro_5min_both['DateTime'].dt.minute==0),'Day'] -= 1

# Take one minute off the time, then round up to next 2-hour event
time_1 = (Micro_5min_both['DateTime'] - datetime.timedelta(minutes=1)).dt.hour
Micro_5min_both['Next_2hr'] = time_1 + (time_1+1)%2+1

######## BUGS faithfully ported #########
# This code should be deleted, but is needed to match submission
# In the two-hour period just before the first 2-hour of a Micro interval, standard deviation was computed on only half of the preceeding two hours
for i in range(len(Micro_5min_both)-12):
    if ((Micro_5min_both.Train[i]) & (not Micro_5min_both.Train[i+11])):
        Micro_5min_both.loc[i-1:i,'leafwet_lwscnt'] = np.nan

# Somehow the loop is catching most, but not all of them (and I'm really frustrated learning python)
NukeMe = (Micro_5min_both['Hour']==23) & (Micro_5min_both['Day']==22) & (Micro_5min_both['Year']==2014)
Micro_5min_both.loc[NukeMe,'leafwet_lwscnt'] = np.nan
NukeMe = (Micro_5min_both['Hour']==23) & (Micro_5min_both['Day']==142) & (Micro_5min_both['Year']==2014)
Micro_5min_both.loc[NukeMe,'leafwet_lwscnt'] = np.nan
NukeMe = (Micro_5min_both['Hour']==23) & (Micro_5min_both['Day']==237) & (Micro_5min_both['Year']==2015)
Micro_5min_both.loc[NukeMe,'leafwet_lwscnt'] = np.nan
######## End BUG replication ############

# Apologies for another bad hack, but I'm exhausted:
# I had two-hour intervals like 4:00AM - 5:55AM, but apparently my submission used 4:05AM - 6:00AM.
Micro_5min_both['LWS'] = Micro_5min_both['leafwet_lwscnt'].shift(1)
# End bad hack ##########################

Micro_5min_both.loc[Micro_5min_both['LWS']<2,'LWS'] = np.nan # Ick, go away (14 rows of outlier data)
Micro_5min_both['LWS'].fillna(method='ffill')
Micro_5min_both.loc[(Micro_5min_both['wind_ms']).isnull(),'LWS'] = np.nan # Ick, go away

################ GROUP BY to get StdDev(leafwet_lwscnt) and max(datetime) for each 2-hour interval ###############

Micro_5min_both['DateTime'] = Micro_5min_both.DateTime
grouped_train = Micro_5min_both.groupby(['Year','Month','Day','Next_2hr'])

maxDate_train = grouped_train.agg((np.max))['DateTime']
fixedDate_train = maxDate_train.apply(lambda x: x + datetime.timedelta(minutes=5) if x.minute==55 else x)

lws_sd_both1 = grouped_train.agg((np.std))['LWS']
lws_sd_both = pd.concat([fixedDate_train, lws_sd_both1], axis=1)
lws_sd_both.rename(columns={'LWS':'lws_std'},inplace=True)

pd.DataFrame.to_csv(lws_sd_both,'./data/interim/lws_std_both.csv', columns={'DateTime','lws_std'}, index=False)
Train = grouped_train.agg((np.min))['Train']

# Sad hack: those 9 rows that start Micro intervals are in Training, but I want them in test, so now they're in both
pd.DataFrame.to_csv(lws_sd_both,'./data/interim/lws_std_train.csv', columns={'DateTime','lws_std'}, index=False)
pd.DataFrame.to_csv(lws_sd_both,'./data/interim/lws_std_test.csv', columns={'DateTime','lws_std'}, index=False)
#pd.DataFrame.to_csv(lws_sd_both[Train],'./data/interim/lws_std_train.csv', columns={'DateTime','lws_std'}, index=False)
#pd.DataFrame.to_csv(lws_sd_both[~Train],'./data/interim/lws_std_test.csv', columns={'DateTime','lws_std'}, index=False)
