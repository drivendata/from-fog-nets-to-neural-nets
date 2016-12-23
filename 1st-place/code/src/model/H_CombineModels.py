# Date: May, 2016
# DrivenData.org Fog Net competition solution

import pandas as pd
import numpy as np
import datetime

# -------------------------------------------------------------------------------------------------------------- #
# Combine predictions from the Micro and Macro models
# -------------------------------------------------------------------------------------------------------------- #


# ---------------------------------------- Read in data -------------------------------------------------------- #

#Micro_test = pd.read_csv('./data/interim/Micro_L4_test.csv')
Micro_test = pd.read_csv('./data/interim/Micro_L4_test.csv')
Macro_test = pd.read_csv('./data/interim/Layer8_test.csv',index_col=False)

DateFormat='%Y-%m-%d %H:%M:%S'

Submission = pd.merge(Micro_test[['DateTime','L3','L4']], Macro_test[['DateTime','L8_lag']],on='DateTime',how='outer').sort_values('DateTime')
Submission.reset_index(drop=True, inplace=True)
FogYield = []
for i in range(len(Submission)):
    y = Submission['L4'][i] if np.isfinite(Submission['L4'][i]) else Submission['L8_lag'][i]
    z = y if y>0 else 0
    FogYield.append(z)
Submission['yield'] = FogYield

Submission[['DateTime','yield']].to_csv('./data/Submission.csv', index=False, float_format='%.6f')
