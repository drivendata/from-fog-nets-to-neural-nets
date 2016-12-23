# Date: May, 2016
# DrivenData.org Fog Net competition solution

import pandas as pd
import datetime

# "Step" refers to the change in leafwet_lwscnt over the last two hours
#
# Python newbie bad.  TBD: fix "copy of a slice" problem by using df.loc

# ---------------------------------------- Read in data -------------------------------------------------------- #

Micro_2hr_train = pd.read_csv('./data/interim/Micro_2hr_train.csv',index_col=False)
Micro_2hr_test = pd.read_csv('./data/interim/Micro_test_1119.csv',index_col=False)

DateFormat='%Y-%m-%d %H:%M:%S'
Micro_2hr_train.rename(columns={Micro_2hr_train.columns[0]:'DateTime','leafwet_lwscnt':'LWS'},inplace=True)
Micro_2hr_train.DateTime = pd.to_datetime(Micro_2hr_train.DateTime, format=DateFormat)
Micro_2hr_test.rename(columns={Micro_2hr_test.columns[0]:'DateTime','leafwet_lwscnt':'LWS'},inplace=True)
Micro_2hr_test.DateTime = pd.to_datetime(Micro_2hr_test.DateTime, format=DateFormat)

# -------------------------------------------------------------------------------------------------------------- #

Micro_train = Micro_2hr_train
Micro_test = Micro_2hr_test

Micro_both = pd.concat([Micro_train, Micro_test], axis=0).sort_values('DateTime')
Micro_both.reset_index(drop=True, inplace=True)
Micro_both['Step'] = Micro_both.LWS - Micro_both.LWS.shift()
elapsed = Micro_both['DateTime'] - Micro_both.shift()['DateTime']
Micro_both['Step'][0] = 0
for i in range(1,len(Micro_both)):
    if ((elapsed[i].seconds//60)>240) | (elapsed[i].days>1): # 120 (2 hours) should work better
        Micro_both['Step'][i] = 0

Micro_train = pd.merge(Micro_train, Micro_both[['DateTime','Step']], on='DateTime', how='left')
Micro_test = pd.merge(Micro_test, Micro_both[['DateTime','Step']], on='DateTime', how='left')

# Save LWS Steps
micro_cols = ['DateTime','Step','LWS']
Micro_train[micro_cols].to_csv('./data/interim/Step_train.csv', index=False, float_format='%.6f')
Micro_test[micro_cols].to_csv('./data/interim/Step_test.csv', index=False, float_format='%.6f')
