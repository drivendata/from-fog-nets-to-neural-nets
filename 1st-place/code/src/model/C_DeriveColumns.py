# Date: May, 2016
# DrivenData.org Fog Net competition solution

import pandas
import numpy as np
import datetime
import math
import re


# -------------------------------------------------------------------------------------------------------------- #
# Get wind directions, altitudes, etc. Define categorical variables.
# Many of these categories were learned manually, looking at the data. The strategy was to identify a few nominal
# variables that would strongly signify wetness or dryness, then either include these terms in the regression, or
# split the problem into two regressions, one wet, one dry.
#
# Python newbie apologies for all transgressions
# -------------------------------------------------------------------------------------------------------------- #


# ---------------------------------------- Read the data file -------------------------------------------------- #

# Imputed_SAG.csv contains the raw data plus imputed values (fancy fillna-like in columns xx_yy_fc)
Imp_SAG = pandas.read_csv('./data/interim/Imputed_SAG.csv', index_col=False)
DateFormat='%Y-%m-%d %H:%M:%S'
Imp_SAG.rename(columns={Imp_SAG.columns[0]:'DateTime'},inplace=True)
Imp_SAG.DateTime = pandas.to_datetime(Imp_SAG.DateTime, format=DateFormat)

# Select from Imp_SAG those rows with yield
Subm_Form = pandas.read_csv('./data/raw/submission_format.csv',index_col=False)
Subm_Form.rename(columns={Subm_Form.columns[0]:'DateTime'},inplace=True)
Subm_Form.DateTime = pandas.to_datetime(Subm_Form.DateTime, format=DateFormat)
Subm_Form.drop('yield', axis=1, inplace=True)

Raw_Yield = pandas.read_csv('./data/raw/Target Variable Water Yield.csv',index_col=False)
Raw_Yield.rename(columns={Raw_Yield.columns[0]:'DateTime'},inplace=True)
Raw_Yield.DateTime = pandas.to_datetime(Raw_Yield.DateTime, format=DateFormat)
Raw_Yield.drop('yield', axis=1, inplace=True)
Keepers = pandas.concat([Raw_Yield, Subm_Form],axis=0).sort_values('DateTime')
SAG = pandas.merge(Keepers, Imp_SAG, on='DateTime', how='inner').sort_values('DateTime')


SAG['DayOfYear'] = SAG['DateTime'].apply(lambda x: x.timetuple().tm_yday)
SAG['Hour'] = pandas.DatetimeIndex(SAG.DateTime).hour


########################################## Sidi Ifni ########################################################### #

SAG['TTd_Si'] = SAG['T_Si_fc'] - SAG['Td_Si_fc']
SAG['RRR_S_any'] = SAG['RRR_Si_fc'].apply(lambda x: "dry" if str(x)=="nan" else "wet")

# ---------------------------------------- DD_Dir_Si ----------------------------------------------------------- #
# Map wind direction text to angles
windDir = []
for i in range(len(SAG)):
    s = str(SAG['DD_Si_fc'][i])
    if "Calm" in s:
        windDir.append(130)
    elif "variable" in s:
        windDir.append(265)
    elif "east-northeast" in s:
        windDir.append(67.5)
    elif "east-southeast" in s:
        windDir.append(112.5)
    elif "north-northeast" in s:
        windDir.append(22.5)
    elif "north-northwest" in s:
        windDir.append(337.5)
    elif "south-southeast" in s:
        windDir.append(157.5)
    elif "south-southwest" in s:
        windDir.append(202.5)
    elif "west-northwest" in s:
        windDir.append(292.5)
    elif "west-southwest" in s:
        windDir.append(247.5)
    elif "north-east" in s:
        windDir.append(45)
    elif "north-west" in s:
        windDir.append(315)
    elif "south-east" in s:
        windDir.append(135)
    elif "south-west" in s:
        windDir.append(225)
    elif "east" in s:
        windDir.append(90)
    elif "north" in s:
        windDir.append(0)
    elif "south" in s:
        windDir.append(180)
    elif "west" in s:
        windDir.append(270)
    else:
        windDir.append(310)
# Regression curve shape is simpler if I rotate the compass
SAG['Sidi_winds_shifted'] = map(lambda x: (x + 270) % 360, windDir)


# ---------------------------------------- Nh_S_real ----------------------------------------------------------- #
# Bigger numbers for more clouds, with tweaks
# This term proved very linear and strong in the models
Nh_S_real = []
for i in range(len(SAG)):
    Nh = SAG['Nh_Si'][i]
    if not Nh:
        Nh_S_real.append(0)
    elif Nh == "no clouds":
        Nh_S_real.append(0)
    elif Nh == "10%  or less, but not 0":
        Nh_S_real.append(1)
    elif Nh == "20-30%":
        Nh_S_real.append(2)
    elif Nh == "40%":
        Nh_S_real.append(3)
    elif Nh == "50%":
        Nh_S_real.append(4)
    elif Nh == "60%":
        Nh_S_real.append(4)
    elif Nh == "70 - 80%":
        Nh_S_real.append(5)
    elif Nh == "90  or more, but not 100%":
        Nh_S_real.append(6)
    elif Nh == "100%":
        Nh_S_real.append(5)
SAG['Nh_S_real'] = Nh_S_real


# ---------------------------------------- E_Si ---------------------------------------------------------------- #
# Sloppy: similar code produced E_Si_fc
E_Si2 = []       # used in DrySplit, below
E_Si3 = []       # used as a modeling term
for i in range(len(SAG)):
    E = str(SAG['E_Si_fc'][i])
    if E=='nan': # I don't understand python nulls...
        E_Si2.append("na")
        E_Si3.append("Other")
    elif E=="Dry":
        E_Si2.append("damp")
        E_Si3.append("Other")
    elif E=="Flooded":
        E_Si2.append("wet")
        E_Si3.append("Wet")
    elif E=="moist":
        E_Si2.append("moist")
        E_Si3.append("Other")
    else:
        E_Si2.append("wet")
        E_Si3.append("Wet")
SAG['E_Si2'] = E_Si2
SAG['E_Si3'] = E_Si3


########################################## Agadir ############################################################## #

# ---------------------------------------- sqrt_P0_Ag----------------------------------------------------------- #
SAG['sqrt_P0_Ag'] = SAG['P0_Ag_fc'].apply(lambda x: math.sqrt(x-746))


# ---------------------------------------- clouds_Ag ----------------------------------------------------------- #
clouds_Ag = []
for i in range(len(SAG)):
    c = str(SAG['c_Ag_fc'][i])
    if (c=="na") | ("Scattered" in c) | ("Few" in c) | ("Broken" in c):
        clouds_Ag.append("wet")
    else:
        clouds_Ag.append("dry")
SAG['clouds_Ag'] = clouds_Ag


########################################## Guelmim ############################################################# #

SAG['TTd_Gu'] = SAG['T_Gu_fc'] - SAG['Td_Gu_fc']

# ---------------------------------------- clouds_G ------------------------------------------------------------ #
clouds_G = []
for i in range(len(SAG)):
    c = str(SAG['c_Gu'][i])
    if not c:
        clouds_G.append("med")
    elif ("Few" in c) | ("Scattered" in c):
        clouds_G.append("med")
    elif "Broken" in c:
        clouds_G.append("wet")
    else:
        clouds_G.append("dry")
SAG['clouds_G'] = clouds_G

# ---------------------------------------- clouds_G ------------------------------------------------------------ #
# BUG required to match winning entry. Should be three OR MORE digits: '([0-9][0-9][0-9][0-9]*)'

c_meters_Gu = []
for i in range(len(SAG)):
    c = str(SAG['c_Gu'][i])
    if not c:
        c_meters_Gu.append(np.nan)
    elif ("No Significant" in c):
        c_meters_Gu.append(950)
    else:
        cleaner = re.sub('[)] 000','[)]', c)
        found = re.search('([0-9][0-9][0-9])',cleaner) # Bug (see above)
        if found:
            c_meters_Gu.append(int(found.group(1)))
        else:
            c_meters_Gu.append(np.nan)

SAG['cm_Gu_600'] = map(lambda x: 1 if x > 600 else 0, c_meters_Gu)


# ---------------------------------------- DrySplit ------------------------------------------------------------ #

# THIS IS AN UNFORTUNATE LEGACY AND COULD BE VASTLY SIMPLIFIED
# It should be sufficient to define two classes for DrySplit:
#    I doubt the distinction between "dry" and "dryish" is important.
#    Just work through a truth table to identify DrySplit=="other"

DryDry = []
for i in range(len(SAG)):
    if (SAG['clouds_G'][i]=="dry") & (SAG['clouds_Ag'][i]=="dry"):
        DryDry.append("DryDry")
    else:
        DryDry.append("Other")

DD_Gu_NEW = []
for i in range(len(SAG)):
    dd = str(SAG['DD_Gu_fc'][i])
    if dd=="nan":
        DD_Gu_NEW.append("NA")
    elif "west-northwest" in dd:
        DD_Gu_NEW.append("dry wind")
    elif ("north" in dd) | ("Calm" in dd):
        DD_Gu_NEW.append("north")
    else:
        DD_Gu_NEW.append("dry wind")


DrySplit = []
for i in range(len(SAG)):
    ttd_Gu = SAG['TTd_Gu'][i]
    drydry = DryDry[i]
    DD_Gu  = DD_Gu_NEW[i]
    E_Si2   = str(SAG['E_Si2'][i])
    c_G    = SAG['clouds_G'][i]
    if np.isnan(ttd_Gu):
        DrySplit.append("dry")
    elif ttd_Gu > 10.8:
        if drydry=="DryDry":
            DrySplit.append("dry")
        else:
            if E_Si2=="na":
                DrySplit.append("dry")
            elif E_Si2=="damp":
                DrySplit.append("dryish")
            elif c_G=="dry":
                DrySplit.append("dryish")
            else:
                DrySplit.append("other")
    else:
        if (drydry=="DryDry") & (DD_Gu=="north"):
            DrySplit.append("dryish")
        elif (drydry=="DryDry"):
            DrySplit.append("dry")
        elif DD_Gu=="dry wind":
            if (E_Si2=="na"):
               DrySplit.append("dry")
            elif (E_Si2=="damp") | (c_G=="dry"):
                DrySplit.append("dryish")
            else:
                DrySplit.append("other")
        elif (c_G=="dry"):
            DrySplit.append("dryish")
        else:
            DrySplit.append("other")

SAG['DrySplit'] = DrySplit


#new_cols = ['DayOfYear', 'Sidi_winds_shifted', 'Nh_S_real', 'E_Si2', 'E_Si3', 'sqrt_P0_Ag', 'clouds_Ag', 'clouds_G', 'TTd_Gu', 'DrySplit']

TRAIN = pandas.merge(Raw_Yield, SAG, on='DateTime', how='inner').sort_values('DateTime')
TEST = pandas.merge(Subm_Form, SAG, on='DateTime', how='inner').sort_values('DateTime')

TRAIN.to_csv('./data/interim/Derived_SAG_train.csv', index=False, float_format='%.9f')
TEST.to_csv('./data/interim/Derived_SAG_test.csv', index=False, float_format='%.9f')
