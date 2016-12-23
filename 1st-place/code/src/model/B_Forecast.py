# Date: May, 2016
# DrivenData.org Fog Net competition solution

import pandas
import numpy as np
import datetime
import math


# -------------------------------------------------------------------------------------------------------------- #
# Python newbie apologies for transgressions
# -------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------- #
#   Forecasting the weather station data is done in order to have input for predicting the fog yield.
#     Some key inputs are not reported through the night.
#
#   Typical example forecasting strategy: T_Si_fc from T_Si (temperature at Sidi Ifni)
#     Determine how many rows back to a non-null T_Si value
#     Determine how old that data is
#     Note T_Si[6AM on the previous day]
#     Fill forward through the night using simple logic based on the last evening temperature and yesterday's 6AM temperature
#   Fill-forward logic depends on the specific variable


# ---------------------------------------- Read the data file -------------------------------------------------- #

SAG = pandas.read_csv('./data/interim/Raw_SAG.csv', index_col=False)
DateFormat='%Y-%m-%d %H:%M:%S'
SAG.rename(columns={SAG.columns[0]:'DateTime'},inplace=True)
SAG.DateTime = pandas.to_datetime(SAG.DateTime, format=DateFormat)

# -------------------------------------------------------------------------------------------------------------- #

# How many rows back is the last reported datum?
def x_last(aList):
    gap = aList.apply(lambda x: True if np.isnan(x) else False)
    counts = []
    for i in range(len(aList)):
        if not gap[i]:
            counter = 0
        elif gap[i] and (i==0 or counter == None):
            counter = None
        elif gap[i]:
            counter += 1
        else:
            print "TBD"
        counts.append(counter)
    return(counts)

# ---------------------------------------- T_Ag ---------------------------------------------------------------- #
T_Ag_cols = ['DateTime', 'T_Ag', 'T_Ag_fc']

DD_int = SAG['T_Ag'].apply(lambda x: x if pandas.isnull(x) else 1) # x_last wants numeric
SAG['T_Ag_last'] = x_last(DD_int)          # How many rows back
myI = range(len(SAG)) - SAG['T_Ag_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['T_Ag_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

T_Ag_fc = []
fc = None
for i in range(len(SAG)):
    if SAG['T_Ag_age'][i]==0:              # Actual value when available
        fc = SAG['T_Ag'][i]
        T_Ag_fc.append(fc)
    elif ((SAG['T_Ag_age'][i]<800)):       # Stutter last value unless old
        T_Ag_fc.append(fc)
    else:
        T_Ag_fc.append(None)
SAG['T_Ag_fc'] = T_Ag_fc


# ---------------------------------------- Td_Ag ---------------------------------------------------------------- #
Td_Ag_cols = ['DateTime', 'Td_Ag', 'Td_Ag_fc']

DD_int = SAG['Td_Ag'].apply(lambda x: x if pandas.isnull(x) else 1) # x_last wants numeric
SAG['Td_Ag_last'] = x_last(DD_int)          # How many rows back
myI = range(len(SAG)) - SAG['Td_Ag_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['Td_Ag_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

Td_Ag_fc = []
fc = None
for i in range(len(SAG)):
    if SAG['Td_Ag_age'][i]==0:              # Actual value when available
        fc = SAG['Td_Ag'][i]
        Td_Ag_fc.append(fc)
    elif ((SAG['Td_Ag_age'][i]<800)):       # Stutter last value unless old
        Td_Ag_fc.append(fc)
    else:
        Td_Ag_fc.append(None)
SAG['Td_Ag_fc'] = Td_Ag_fc


# ---------------------------------------- P0_Ag ---------------------------------------------------------------- #
P0_Ag_cols = ['DateTime', 'P0_Ag', 'P0_Ag_fc']

DD_int = SAG['P0_Ag'].apply(lambda x: x if pandas.isnull(x) else 1) # x_last wants numeric
SAG['P0_Ag_last'] = x_last(DD_int)          # How many rows back
myI = range(len(SAG)) - SAG['P0_Ag_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['P0_Ag_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

P0_Ag_fc = []
fc = None
for i in range(len(SAG)):
    if SAG['P0_Ag_age'][i]==0:              # Actual value when available
        fc = SAG['P0_Ag'][i]
        P0_Ag_fc.append(fc)
    elif ((SAG['P0_Ag_age'][i]<1200)):      # Stutter last value unless old
        P0_Ag_fc.append(fc)
    else:
        P0_Ag_fc.append(None)
SAG['P0_Ag_fc'] = P0_Ag_fc


# ---------------------------------------- c_Ag ---------------------------------------------------------------- #
c_Ag_cols = ['DateTime', 'c_Ag', 'c_Ag_last', 'c_Ag_age', 'c_Ag_fc']

c_Agt = SAG['c_Ag'].apply(lambda x: x if pandas.isnull(x) else 1) # x_last wants numeric
SAG['c_Ag_last'] = x_last(c_Agt)          # How many rows back
myI = range(len(SAG)) - SAG['c_Ag_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['c_Ag_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

c_Ag_fc = []
fc = None
for i in range(len(SAG)):
    if SAG['c_Ag_age'][i]==0:               # Actual value when available
        fc = SAG['c_Ag'][i]
        c_Ag_fc.append(fc)
    elif ((SAG['c_Ag_age'][i]<800)):	    # Stutter last value unless old
        c_Ag_fc.append(fc)
    else:
        c_Ag_fc.append(None)
SAG['c_Ag_fc'] = c_Ag_fc


# ---------------------------------------- Ff_Gu --------------------------------------------------------------- #
Ff_Gu_cols = ['DateTime', 'Ff_Gu', 'Ff_Gu_last', 'Ff_Gu_age', 'Ff_Gu_6AM', 'Ff_Gu_fc']

SAG['Ff_Gu_last'] = x_last(SAG['Ff_Gu'])    # How many rows back
myI = range(len(SAG)) - SAG['Ff_Gu_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['Ff_Gu_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

sixAM = SAG.DateTime.apply(lambda x: True if x.hour==6 else False)
SAG['Ff_Gu_6AM'] = SAG['Ff_Gu'][sixAM]
SAG['Ff_Gu_6AM'] = SAG['Ff_Gu_6AM'].fillna(method='ffill')

Ff_Gu_fc = []
for i in range(len(SAG)):
    if np.isfinite(SAG['Ff_Gu'][i]):	    # Actual value when available
        Ff_Gu_fc.append(SAG['Ff_Gu'][i])
    elif not np.isfinite(myI[i]):
        Ff_Gu_fc.append(None)
    elif SAG['Ff_Gu_age'][i]<250:	    # Stutter last value if recent
        Ff_Gu_fc.append(SAG['Ff_Gu'][myI[i]])
    else:				    # Taper down over longer stretches
        Ff_Gu_fc.append(SAG['Ff_Gu'][myI[i]]*0.9**((SAG['Ff_Gu_age'][i])/60.0))
SAG['Ff_Gu_fc'] = Ff_Gu_fc


# ---------------------------------------- T_Gu --------------------------------------------------------------- #
T_Gu_cols = ['DateTime', 'T_Gu', 'T_Gu_last', 'T_Gu_age', 'T_Gu_6AM', 'T_Gu_fc']

SAG['T_Gu_last'] = x_last(SAG['T_Gu'])      # How many rows back
myI = range(len(SAG)) - SAG['T_Gu_last']    # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['T_Gu_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

sixAM = SAG.DateTime.apply(lambda x: True if (x.hour + x.minute/60.0)==6 else False)
SAG['T_Gu_6AM'] = SAG['T_Gu'][sixAM]
SAG['T_Gu_6AM'] = SAG['T_Gu_6AM'].fillna(method='ffill')

T_Gu_fc = []
for i in range(len(SAG)):
    if SAG['T_Gu_age'][i]==0:		    # Actual value when available
        T_Gu_fc.append(SAG['T_Gu'][i])
    elif not np.isfinite(myI[i]):
        T_Gu_fc.append(None)		    # Quick hack
    else:				    # Taper down over night
        last_t = SAG.DateTime[myI[i]]
        T_Gu_fc.append(SAG['T_Gu'][myI[i]] + ((SAG['T_Gu'][myI[i]] - SAG['T_Gu_6AM'][i])*((SAG['T_Gu_age'][i])/60.0))/(6-last_t.time().hour))
SAG['T_Gu_fc'] = T_Gu_fc

# ---------------------------------------- Td_Gu --------------------------------------------------------------- #
Td_Gu_cols = ['DateTime', 'Td_Gu', 'Td_Gu_last', 'Td_Gu_age', 'Td_Gu_6AM', 'Td_Gu_fc']

SAG['Td_Gu_last'] = x_last(SAG['Td_Gu'])    # How many rows back
myI = range(len(SAG)) - SAG['Td_Gu_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['Td_Gu_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

sixAM = SAG.DateTime.apply(lambda x: True if (x.hour + x.minute/60.0)==6 else False)
SAG['Td_Gu_6AM'] = SAG['Td_Gu'][sixAM]
SAG['Td_Gu_6AM'] = SAG['Td_Gu_6AM'].fillna(method='ffill')

Td_Gu_fc = []
for i in range(len(SAG)):
    if SAG['Td_Gu_age'][i]==0:		    # Actual value when available
        Td_Gu_fc.append(SAG['Td_Gu'][i])
    elif not np.isfinite(myI[i]):
        Td_Gu_fc.append(None)		    # Quick hack
    else:				    # Taper down over night
        last_t = SAG.DateTime[myI[i]]
        Td_Gu_fc.append(SAG['Td_Gu'][myI[i]] + ((SAG['Td_Gu'][myI[i]] - SAG['Td_Gu_6AM'][i])*((SAG['Td_Gu_age'][i])/60.0))/(6-last_t.time().hour))
SAG['Td_Gu_fc'] = Td_Gu_fc


# ---------------------------------------- c_Gu --------------------------------------------------------------- #
SAG['c_Gu'] = SAG['c_Gu'].fillna(method='ffill')


# ---------------------------------------- DD_Gu -------------------------------------------------------------- #
SAG['DD_Gu_fc'] = SAG['DD_Gu'].fillna(method='ffill')



# ---------------------------------------- T_Si ---------------------------------------------------------------- #
T_Si_cols = ['DateTime', 'T_Si', 'T_Si_last', 'T_Si_age', 'T_Si_6AM', 'T_Si_fc']


SAG['T_Si_last'] = x_last(SAG['T_Si'])      # How many rows back
myI = range(len(SAG)) - SAG['T_Si_last']    # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['T_Si_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

sixAM = SAG.DateTime.apply(lambda x: True if x.hour==6 else False)
SAG['T_Si_6AM'] = SAG['T_Si'][sixAM]
SAG['T_Si_6AM'] = SAG['T_Si_6AM'].fillna(method='ffill')

T_Si_fc = []
fc = None
for i in range(len(SAG)):
    t = SAG.DateTime[i]
    daytime = (t.time() > datetime.time(6)) & (t.time() < datetime.time(18))
    if SAG['T_Si_age'][i]==0:               # Actual value when available
        fc = SAG['T_Si'][i]
        T_Si_fc.append(fc)
    elif not np.isfinite(myI[i]):
        T_Si_fc.append(None)                # Quick hack
    elif ((SAG['T_Si_age'][i]>720) | daytime): # Stutter last value if old or daytime
        T_Si_fc.append(fc)
    else:                                   # Taper down over night
        last_t = SAG.DateTime[myI[i]]
        fc = SAG['T_Si'][myI[i]] + ((SAG['T_Si'][myI[i]] - SAG['T_Si_6AM'][i])*((SAG['T_Si_age'][i])/60.0))/(6-last_t.time().hour)
        T_Si_fc.append(fc)
SAG['T_Si_fc'] = T_Si_fc


# ---------------------------------------- Td_Si ---------------------------------------------------------------- #
Td_Si_cols = ['DateTime', 'Td_Si', 'Td_Si_last', 'Td_Si_age', 'Td_Si_6AM', 'Td_Si_fc']

SAG['Td_Si_last'] = x_last(SAG['Td_Si'])    # How many rows back
myI = range(len(SAG)) - SAG['Td_Si_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['Td_Si_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

sixAM = SAG.DateTime.apply(lambda x: True if (x.hour + x.minute/60.0)==6 else False)
SAG['Td_Si_6AM'] = SAG['Td_Si'][sixAM]
SAG['Td_Si_6AM'] = SAG['Td_Si_6AM'].fillna(method='ffill')

Td_Si_fc = []
fc = None
for i in range(len(SAG)):
    t = SAG.DateTime[i]
    daytime = (t.time() > datetime.time(6)) & (t.time() < datetime.time(18))
    if SAG['Td_Si_age'][i]==0:		    # Actual value when available
        fc = SAG['Td_Si'][i]
        Td_Si_fc.append(fc)
    elif not np.isfinite(myI[i]):
        Td_Si_fc.append(None)		    # Quick hack
    elif ((SAG['Td_Si_age'][i]>720) | daytime): # Stutter last value if old or daytime
        Td_Si_fc.append(fc)
    else:				    # Taper down over night
        last_t = SAG.DateTime[myI[i]]
        fc = SAG['Td_Si'][myI[i]] + ((SAG['Td_Si'][myI[i]] - SAG['Td_Si_6AM'][i])*((SAG['Td_Si_age'][i])/60.0))/(6-last_t.time().hour)
        Td_Si_fc.append(fc)
SAG['Td_Si_fc'] = Td_Si_fc


# ---------------------------------------- DD_Si --------------------------------------------------------------- #
DD_Si_cols = ['DateTime', 'DD_Si', 'DD_Si_last', 'DD_Si_age', 'DD_Si_fc']

DD_int = SAG['DD_Si'].apply(lambda x: x if pandas.isnull(x) else 1) # x_last wants numeric
SAG['DD_Si_last'] = x_last(DD_int)          # How many rows back
myI = range(len(SAG)) - SAG['DD_Si_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['DD_Si_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

DD_Si_fc = []
fc = None
for i in range(len(SAG)):
    if SAG['DD_Si_age'][i]==0:		    # Actual value when available
        fc = SAG['DD_Si'][i]
        DD_Si_fc.append(fc)
    elif ((SAG['DD_Si_age'][i]<400)):	    # Stutter last value unless old
        DD_Si_fc.append(fc)
    else:
        DD_Si_fc.append(None)
SAG['DD_Si_fc'] = DD_Si_fc


# ---------------------------------------- RRR_Si --------------------------------------------------------------- #
RRR_Si_cols = ['DateTime', 'RRR_Si', 'RRR_Si_last', 'RRR_Si_age', 'RRR_Si_fc']

RRR_int = SAG['RRR_Si'].apply(lambda x: x if pandas.isnull(x) else 1) # x_last wants numeric
SAG['RRR_Si_last'] = x_last(RRR_int)        # How many rows back
myI = range(len(SAG)) - SAG['RRR_Si_last']  # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['RRR_Si_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

RRR_Si_fc = []
fc = None
for i in range(len(SAG)):
    if SAG['RRR_Si_age'][i]==0:		    # Actual value when available
        fc = SAG['RRR_Si'][i]
        RRR_Si_fc.append(fc)
    elif ((SAG['RRR_Si_age'][i]<400)):	    # Stutter last value unless old
        RRR_Si_fc.append(fc)
    else:
        RRR_Si_fc.append(None)
SAG['RRR_Si_fc'] = RRR_Si_fc


# ---------------------------------------- E_Si --------------------------------------------------------------- #
E_Si_cols = ['DateTime', 'E_Si', 'E_Si_last', 'E_Si_age', 'E_Si_fc_verbose', 'E_Si_fc']

E_int = SAG['E_Si'].apply(lambda x: x if pandas.isnull(x) else 1) # x_last wants numeric
SAG['E_Si_last'] = x_last(E_int)           # How many rows back
myI = range(len(SAG)) - SAG['E_Si_last']   # Row number of the datetime

prevTime = myI.apply(lambda x: SAG.DateTime[1] if math.isnan(x) else SAG.DateTime[int(x)]) # SAG.DateTime[1] isn't right, but might do. Should be a missing value.
elapsedTime = SAG.DateTime[range(len(SAG))] - pandas.to_datetime(prevTime, format=DateFormat)

SAG['E_Si_age'] = elapsedTime.apply(lambda x: x.days*24*60 + x.seconds // 60) # elapsed time in minutes

E_Si_fc_verbose = []
fc = None
for i in range(len(SAG)):
    if SAG['E_Si_age'][i]==0:		   # Actual value when available
        fc = SAG['E_Si'][i]
        E_Si_fc_verbose.append(fc)
    elif ((SAG['E_Si_age'][i]<24*60)):	   # Stutter last value unless old
        E_Si_fc_verbose.append(fc)
    else:
        E_Si_fc_verbose.append(None)
SAG['E_Si_fc_verbose'] = E_Si_fc_verbose
SAG['E_Si_fc'] = SAG['E_Si_fc_verbose'].apply(lambda x: '' if pandas.isnull(x) else 'Dry' if 'Dry' in x else 'Flooded' if 'Flooded' in x else 'moist' if 'moist' in x else 'wet' if 'wet' in x else '')

# ---------------------------------------- Nh_Si -------------------------------------------------------------- #
SAG['Nh_Si'] = SAG['Nh_Si'].fillna(method='ffill')


SAG.drop(['E_Si_last', 'E_Si_age', 'E_Si_fc_verbose'], axis=1, inplace=True)
SAG.drop(['E_Si'], axis=1, inplace=True) # not needed and I confused myself by reusing this name
SAG.drop(['RRR_Si', 'RRR_Si_last', 'RRR_Si_age'], axis=1, inplace=True)
SAG.drop(['DD_Si', 'DD_Si_last', 'DD_Si_age'], axis=1, inplace=True)
SAG.drop(['Td_Si', 'Td_Si_last', 'Td_Si_6AM', 'Td_Si_age'], axis=1, inplace=True)
SAG.drop(['T_Si', 'T_Si_last', 'T_Si_6AM', 'T_Si_age'], axis=1, inplace=True)
SAG.drop(['T_Gu', 'T_Gu_last', 'T_Gu_6AM', 'T_Gu_age'], axis=1, inplace=True)
SAG.drop(['Td_Gu', 'Td_Gu_last', 'Td_Gu_6AM', 'Td_Gu_age'], axis=1, inplace=True)
SAG.drop(['Ff_Gu', 'Ff_Gu_last', 'Ff_Gu_6AM', 'Ff_Gu_age'], axis=1, inplace=True)
SAG.drop(['DD_Gu'], axis=1, inplace=True)
#SAG.drop(['DD_Gu_last', 'DD_Gu_age'], axis=1, inplace=True)
SAG.drop(['T_Ag', 'T_Ag_last', 'T_Ag_age'], axis=1, inplace=True)
SAG.drop(['Td_Ag', 'Td_Ag_last', 'Td_Ag_age'], axis=1, inplace=True)
SAG.drop(['P0_Ag', 'P0_Ag_last', 'P0_Ag_age'], axis=1, inplace=True)
SAG.drop(['c_Ag', 'c_Ag_last', 'c_Ag_age'], axis=1, inplace=True)


SAG.to_csv('./data/interim/Imputed_SAG.csv', index=False, float_format='%.6f')
