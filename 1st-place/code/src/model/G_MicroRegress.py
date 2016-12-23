# Date: May, 2016
# DrivenData.org Fog Net competition solution

# -------------------------------------------------------------------------------------------------------------- #
# To understand the code, it may help to read Regress.py first. It is similar and has more explanatory comments.
# -------------------------------------------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from patsy import dmatrices, dmatrix


# -------------------------------------------- Define functions ------------------------------------------------ #

# X must be a DataFrame containing a column named x
# This assumes 3 knots!
def k3_basis(X,pre,knots):
    k = 3
    ts = []
    for i in range(0,k):
        t = np.maximum((X.x - knots[i]),0)**3
        ts.append(t)
    d3 = {"Intercept" : np.linspace(1,1,len(X.x)),
            "Lin" : X.x,
            pre + "1" : ts[0] - 2*ts[1] + 1*ts[2]}
    label = "Intercept + Lin + " + pre + "1"
    df3 = pd.DataFrame(d3)
    return({"knots" : knots, "df" : df3})



# X must be a DataFrame containing a column named x
# This assumes 5 knots!
def k5_basis(X,pre,knots):
    k = 5
    ts = []
    for i in range(0,k):
        t = np.maximum((X.x - knots[i]),0)**3
        ts.append(t)
    d5 = {"Intercept" : np.linspace(1,1,len(X.x)),
            "Lin" : X.x,
            pre + "1" : ts[0] - 4*ts[3] + 3*ts[4],
            pre + "2" : ts[1] - 3*ts[3] + 2*ts[4],
            pre + "3" : ts[2] - 2*ts[3] + 1*ts[4]}
    label = "Intercept + Lin + " + pre + "1 + " + pre + "2 + " + pre + "3"
    df5 = pd.DataFrame(d5)
    return({"knots" : knots, "df" : df5})
# -------------------------------------------------------------------------------------------------------------- #


# ---------------------------------------- Read in data -------------------------------------------------------- #
#    Micro_2hr_train, Micro_test_1119       -- LWS, THL, temperature, windspeed, windDir shifted
#    lws_std_train, lws_std_test            -- stdDev (derived from 5min micro)
#    Derived_SAG_train, Derived_SAG_test    -- Nh_S (from Sidi Ifni)

Micro_2hr_train = pd.read_csv('./data/interim/Micro_2hr_train.csv',index_col=False)
Micro_2hr_test = pd.read_csv('./data/interim/Micro_test_1119.csv',index_col=False)
lws_std_train = pd.read_csv('./data/interim/lws_std_train.csv', index_col=False).dropna()
lws_std_test = pd.read_csv('./data/interim/lws_std_test.csv', index_col=False).dropna()
step_train = pd.read_csv('./data/interim/Step_train.csv', index_col=False)
step_test = pd.read_csv('./data/interim/Step_test.csv', index_col=False)
SAG_train = pd.read_csv('./data/interim/Derived_SAG_train.csv', index_col=False)
SAG_test = pd.read_csv('./data/interim/Derived_SAG_test.csv', index_col=False)

DateFormat='%Y-%m-%d %H:%M:%S'
Micro_2hr_train.rename(columns={Micro_2hr_train.columns[0]:'DateTime','leafwet_lwscnt':'LWS'},inplace=True)
Micro_2hr_train.DateTime = pd.to_datetime(Micro_2hr_train.DateTime, format=DateFormat)
Micro_2hr_test.rename(columns={Micro_2hr_test.columns[0]:'DateTime','leafwet_lwscnt':'LWS'},inplace=True)
Micro_2hr_test.DateTime = pd.to_datetime(Micro_2hr_test.DateTime, format=DateFormat)
lws_std_train.rename(columns={lws_std_train.columns[0]:'DateTime'},inplace=True)
lws_std_train.DateTime = pd.to_datetime(lws_std_train.DateTime, format=DateFormat)
lws_std_test.rename(columns={lws_std_test.columns[0]:'DateTime'},inplace=True)
lws_std_test.DateTime = pd.to_datetime(lws_std_test.DateTime, format=DateFormat)
step_train.rename(columns={step_train.columns[0]:'DateTime'},inplace=True)
step_train.DateTime = pd.to_datetime(step_train.DateTime, format=DateFormat)
step_test.rename(columns={step_test.columns[0]:'DateTime'},inplace=True)
step_test.DateTime = pd.to_datetime(step_test.DateTime, format=DateFormat)
SAG_train.rename(columns={SAG_train.columns[0]:'DateTime'},inplace=True)
SAG_train.DateTime = pd.to_datetime(SAG_train.DateTime, format=DateFormat)
SAG_test.rename(columns={SAG_test.columns[0]:'DateTime'},inplace=True)
SAG_test.DateTime = pd.to_datetime(SAG_test.DateTime, format=DateFormat)

# -------------------------------------------------------------------------------------------------------------- #

# ---------------------------------------- Gather metrics ------------------------------------------------------ #

# Get lws_std
Micro_train = pd.merge(Micro_2hr_train, lws_std_train, on='DateTime', how='left').sort_values('DateTime')
Micro_train['sqrtStdDev'] = map(lambda x: np.sqrt(x) if np.isfinite(x) else 0, Micro_train['lws_std']) # filling with 0 may not be the best choice
Micro_test = pd.merge(Micro_2hr_test, lws_std_test, on='DateTime', how='left').sort_values('DateTime')
Micro_test['sqrtStdDev'] = map(lambda x: np.sqrt(x) if np.isfinite(x) else 0, Micro_test['lws_std'])

# Get Step
Micro_train = pd.merge(Micro_train, step_train[['DateTime','Step']], on='DateTime', how='left').sort_values('DateTime')
Micro_test = pd.merge(Micro_test, step_test[['DateTime','Step']], on='DateTime', how='left').sort_values('DateTime')

# Get Nh_S_real from SAG
Micro_train = pd.merge(Micro_train, SAG_train[['DateTime','Nh_S_real']], on='DateTime', how='left').sort_values('DateTime')
Micro_test = pd.merge(Micro_test, SAG_test[['DateTime','Nh_S_real']], on='DateTime', how='left').sort_values('DateTime')
Micro_train.rename(columns={'yield':'FogYield'},inplace=True) # patsy seemed to have trouble with overloading this name

# Redefine north
Micro_train['wind_dir_shifted'] = map(lambda x: (x + 270) % 360, Micro_train['wind_dir'])
Micro_test['wind_dir_shifted'] = map(lambda x: (x + 270) % 360, Micro_test['wind_dir'])

# -------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------- Layer 1: LWS, Nh_S_real, Step ----------------------------------- #
# The same formula is fit to each of four THL splits (THL refers to temperature, humidity, leafwetness):
#   'LWS + Step + Nh_S_real + LWS * Nh_S_real'
# However, after knotting, it looks like this:
f = 'LWS + LWS1 + LWS2 + LWS3 + Step + Step1 + Step2 + Step3 + Nh_S_real + LWS * Nh_S_real'

# Select rows THL=="dry", or M or H or VH:
THL_dry_train = Micro_train[Micro_train.THL=='dry']
THL_dry_test  = Micro_test[Micro_test.THL=='dry']
THL_M_train   = Micro_train[Micro_train.THL=='M']
THL_M_test    = Micro_test[Micro_test.THL=='M']
THL_H_train   = Micro_train[Micro_train.THL=='H']
THL_H_test    = Micro_test[Micro_test.THL=='H']
THL_VH_train  = Micro_train[Micro_train.THL=='VH']
THL_VH_test   = Micro_test[Micro_test.THL=='VH']

# Construct 16 sets of bases (for LWS and Step, training and test, each THL split)
# THL=dry
knots_dry_LWS = [ 435, 437.42708333325, 439.8541666665, 442.28124999975, 444.708333333]
knots_dry_Step = [ -2.04166666700002, -1.15625000025001, -0.270833333500008, 0.614583333249996, 1.5]

training_dry_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_dry_train.LWS}), 'LWS', knots_dry_LWS)
training_dry_LWS_K5['df'].rename(columns={'Intercept':'LWS_Intercept','Lin':'LWS'},inplace=True)
test_dry_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_dry_test.LWS}), 'LWS', knots_dry_LWS)
test_dry_LWS_K5['df'].rename(columns={'Intercept':'LWS_Intercept','Lin':'LWS'},inplace=True)

training_dry_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_dry_train.Step}), 'Step', knots_dry_Step)
training_dry_Step_K5['df'].rename(columns={'Intercept':'Step_Intercept','Lin':'Step'},inplace=True)
test_dry_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_dry_test.Step}), 'Step', knots_dry_Step)
test_dry_Step_K5['df'].rename(columns={'Intercept':'Step_Intercept','Lin':'Step'},inplace=True)

# THL=M
knots_M_LWS = [437.9624999997, 446.9437499997, 455.9249999997, 464.9062499997, 473.8874999997]
knots_M_Step = [-36.4333333339, -22.539583333775, -8.64583333365003, 5.24791666647497, 19.1416666666]

training_M_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_M_train.LWS}), 'LWS', knots_M_LWS)
training_M_LWS_K5['df'].rename(columns={'Lin':'LWS'},inplace=True)
test_M_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_M_test.LWS}), 'LWS', knots_M_LWS)
test_M_LWS_K5['df'].rename(columns={'Lin':'LWS'},inplace=True)

training_M_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_M_train.Step}), 'Step', knots_M_Step)
training_M_Step_K5['df'].rename(columns={'Lin':'Step'},inplace=True)
test_M_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_M_test.Step}), 'Step', knots_M_Step)
test_M_Step_K5['df'].rename(columns={'Lin':'Step'},inplace=True)

# THL=H
knots_H_LWS = [440.71874999975, 474.132812499875, 507.546875, 540.960937500125, 574.37500000025]
knots_H_Step = [-59.4895833335, -25.7161458334375, 8.05729166662496, 41.8307291666874, 75.60416666675]

training_H_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_H_train.LWS}), 'LWS', knots_H_LWS)
training_H_LWS_K5['df'].rename(columns={'Lin':'LWS'},inplace=True)
test_H_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_H_test.LWS}), 'LWS', knots_H_LWS)
test_H_LWS_K5['df'].rename(columns={'Lin':'LWS'},inplace=True)

training_H_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_H_train.Step}), 'Step', knots_H_Step)
training_H_Step_K5['df'].rename(columns={'Lin':'Step'},inplace=True)
test_H_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_H_test.Step}), 'Step', knots_H_Step)
test_H_Step_K5['df'].rename(columns={'Lin':'Step'},inplace=True)

# THL=VH
knots_VH_LWS = [445.7166666668, 487.63750000015, 529.5583333335, 571.47916666685, 613.4000000002]
knots_VH_Step = [-79.0083333338, -38.48750000045, 2.03333333290001, 42.55416666625, 83.0749999996]

training_VH_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_train.LWS}), 'LWS', knots_VH_LWS)
training_VH_LWS_K5['df'].rename(columns={'Lin':'LWS'},inplace=True)
test_VH_LWS_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_test.LWS}), 'LWS', knots_VH_LWS)
test_VH_LWS_K5['df'].rename(columns={'Lin':'LWS'},inplace=True)

training_VH_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_train.Step}), 'Step', knots_VH_Step)
training_VH_Step_K5['df'].rename(columns={'Lin':'Step'},inplace=True)
test_VH_Step_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_test.Step}), 'Step', knots_VH_Step)
test_VH_Step_K5['df'].rename(columns={'Lin':'Step'},inplace=True)

# Make design matrices, regress, and save predictions for each split
training_dry_bases = pd.concat([training_dry_LWS_K5['df'],training_dry_Step_K5['df'], THL_dry_train[['Nh_S_real','FogYield']]],axis=1)
test_dry_bases = pd.concat([test_dry_LWS_K5['df'],test_dry_Step_K5['df'], THL_dry_test[['Nh_S_real']]],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f, data=training_dry_bases, return_type='dataframe')
X_test = dmatrix(f, data=test_dry_bases, return_type='dataframe')
Dry_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='dry','L1'] = Dry_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='dry','L1'] = Dry_res.predict(X_test)

training_M_bases = pd.concat([training_M_LWS_K5['df'],training_M_Step_K5['df'], THL_M_train[['Nh_S_real','FogYield']]],axis=1)
test_M_bases = pd.concat([test_M_LWS_K5['df'],test_M_Step_K5['df'], THL_M_test['Nh_S_real']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f, data=training_M_bases, return_type='dataframe')
X_test = dmatrix(f, data=test_M_bases, return_type='dataframe')
M_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='M','L1'] = M_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='M','L1'] = M_res.predict(X_test)

training_H_bases = pd.concat([training_H_LWS_K5['df'],training_H_Step_K5['df'], THL_H_train[['Nh_S_real','FogYield']]],axis=1)
test_H_bases = pd.concat([test_H_LWS_K5['df'],test_H_Step_K5['df'], THL_H_test['Nh_S_real']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f, data=training_H_bases, return_type='dataframe')
X_test = dmatrix(f, data=test_H_bases, return_type='dataframe')
H_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='H','L1'] = H_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='H','L1'] = H_res.predict(X_test)

training_VH_bases = pd.concat([training_VH_LWS_K5['df'],training_VH_Step_K5['df'], THL_VH_train[['Nh_S_real','FogYield']]],axis=1)
test_VH_bases = pd.concat([test_VH_LWS_K5['df'],test_VH_Step_K5['df'], THL_VH_test['Nh_S_real']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f, data=training_VH_bases, return_type='dataframe')
X_test = dmatrix(f, data=test_VH_bases, return_type='dataframe')
VH_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='VH','L1'] = VH_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='VH','L1'] = VH_res.predict(X_test)

# Save predictions (Micro layer 1)
micro_cols = ['DateTime','Nh_S_real','Step','LWS','wind_dir_shifted','THL','sqrtStdDev','L1']
#Micro_train[micro_cols].to_csv('./data/interim/Micro_L1_train.csv', index=False, float_format='%.6f')
#Micro_test[micro_cols].to_csv('./data/interim/Micro_L1_test.csv', index=False, float_format='%.6f')

# ---------------------------------------- Level 2: Temperature ------------------------------------------------ #
f5 = 'temp + temp1 + temp2 + temp3 + L1'
f3 = 'temp + temp1 + L1 + temp * L1'

# Temperature knots (K5,K3,K3,K3)
knots_VH_L2 = [4.508369557239, 6.98908967389425, 9.4698097905495, 11.9505299072047, 14.43125002386]
knots_H_L2 = [3.721666666668, 8.999166657919, 14.27666664917]
knots_M_L2 = [5.685000000001, 11.5792708504205, 17.47354170084]
knots_dry_L2 = [9.1875, 19.2104166746, 29.2333333492]

# Select rows THL=="dry", or M or H or VH:
THL_dry_train = Micro_train[Micro_train.THL=='dry']
THL_dry_test  = Micro_test[Micro_test.THL=='dry']
THL_M_train   = Micro_train[Micro_train.THL=='M']
THL_M_test    = Micro_test[Micro_test.THL=='M']
THL_H_train   = Micro_train[Micro_train.THL=='H']
THL_H_test    = Micro_test[Micro_test.THL=='H']
THL_VH_train  = Micro_train[Micro_train.THL=='VH']
THL_VH_test   = Micro_test[Micro_test.THL=='VH']

# THL==VH
training_VH_temp_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_train.temp}), 'temp', knots_VH_L2)
training_VH_temp_K5['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)
test_VH_temp_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_test.temp}), 'temp', knots_VH_L2)
test_VH_temp_K5['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)

training_VH_bases = pd.concat([training_VH_temp_K5['df'], THL_VH_train[['L1','FogYield']]],axis=1)
test_VH_bases = pd.concat([test_VH_temp_K5['df'], THL_VH_test['L1']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f5, data=training_VH_bases, return_type='dataframe')
X_test = dmatrix(f5, data=test_VH_bases, return_type='dataframe')
VH_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='VH','L2'] = VH_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='VH','L2'] = VH_res.predict(X_test)

# THL==H
training_H_temp_K3 = k3_basis(pd.DataFrame({"x" : THL_H_train.temp}), 'temp', knots_H_L2)
training_H_temp_K3['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)
test_H_temp_K3 = k3_basis(pd.DataFrame({"x" : THL_H_test.temp}), 'temp', knots_H_L2)
test_H_temp_K3['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)

training_H_bases = pd.concat([training_H_temp_K3['df'], THL_H_train[['L1','FogYield']]],axis=1)
test_H_bases = pd.concat([test_H_temp_K3['df'], THL_H_test['L1']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f3, data=training_H_bases, return_type='dataframe')
X_test = dmatrix(f3, data=test_H_bases, return_type='dataframe')
H_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='H','L2'] = H_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='H','L2'] = H_res.predict(X_test)

# THL==M
training_M_temp_K3 = k3_basis(pd.DataFrame({"x" : THL_M_train.temp}), 'temp', knots_M_L2)
training_M_temp_K3['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)
test_M_temp_K3 = k3_basis(pd.DataFrame({"x" : THL_M_test.temp}), 'temp', knots_M_L2)
test_M_temp_K3['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)

training_M_bases = pd.concat([training_M_temp_K3['df'], THL_M_train[['L1','FogYield']]],axis=1)
test_M_bases = pd.concat([test_M_temp_K3['df'], THL_M_test['L1']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f3, data=training_M_bases, return_type='dataframe')
X_test = dmatrix(f3, data=test_M_bases, return_type='dataframe')
M_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='M','L2'] = M_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='M','L2'] = M_res.predict(X_test)

# THL==dry
training_dry_temp_K3 = k3_basis(pd.DataFrame({"x" : THL_dry_train.temp}), 'temp', knots_dry_L2)
training_dry_temp_K3['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)
test_dry_temp_K3 = k3_basis(pd.DataFrame({"x" : THL_dry_test.temp}), 'temp', knots_dry_L2)
test_dry_temp_K3['df'].rename(columns={'Intercept':'temp_Intercept','Lin':'temp'},inplace=True)

training_dry_bases = pd.concat([training_dry_temp_K3['df'], THL_dry_train[['L1','FogYield']]],axis=1)
test_dry_bases = pd.concat([test_dry_temp_K3['df'], THL_dry_test['L1']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f3, data=training_dry_bases, return_type='dataframe')
X_test = dmatrix(f3, data=test_dry_bases, return_type='dataframe')
dry_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='dry','L2'] = dry_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='dry','L2'] = dry_res.predict(X_test)

# Save predictions (Micro layer 1)
micro_cols = ['DateTime','temp','THL','L1','L2']
#Micro_train[micro_cols].to_csv('./data/interim/Micro_L2_train.csv', index=False, float_format='%.6f')
#Micro_test[micro_cols].to_csv('./data/interim/Micro_L2_test.csv', index=False, float_format='%.6f')

# ---------------------------------------- Level 3: sqrt_stdDev ------------------------------------------------ #
# SqrtStdDev + L2
# VH (K5), H (K5), M (linear), dry (K3)
fVH = 'sqrtStdDev + sqrtStdDev1 + sqrtStdDev2 + sqrtStdDev3 + L2'
fH = 'sqrtStdDev + sqrtStdDev1 + sqrtStdDev2 + sqrtStdDev3 + L2 + sqrtStdDev * L2'
fM = 'sqrtStdDev * L2 + L2'
fdry = 'sqrtStdDev + sqrtStdDev1 + L2 + sqrtStdDev * L2'

knots_VH_L3 = [0.899010144293307, 2.64452049778702, 4.39003085128073, 6.13554120477444, 7.88105155826814]
knots_H_L3 = [0.70965495502062, 2.32594936596958, 3.94224377691853, 5.55853818786749, 7.17483259881645]
knots_dry_L3 = [0, 0.520249061055932, 1.04049812211186]

# Select rows THL=="dry", or M or H or VH:
THL_dry_train = Micro_train[Micro_train.THL=='dry']
THL_dry_test  = Micro_test[Micro_test.THL=='dry']
THL_M_train   = Micro_train[Micro_train.THL=='M']
THL_M_test    = Micro_test[Micro_test.THL=='M']
THL_H_train   = Micro_train[Micro_train.THL=='H']
THL_H_test    = Micro_test[Micro_test.THL=='H']
THL_VH_train  = Micro_train[Micro_train.THL=='VH']
THL_VH_test   = Micro_test[Micro_test.THL=='VH']


# THL==VH
training_VH_sd_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_train.sqrtStdDev}), 'sqrtStdDev', knots_VH_L3)
training_VH_sd_K5['df'].rename(columns={'Intercept':'sqrtStdDev_Intercept','Lin':'sqrtStdDev'},inplace=True)
test_VH_sd_K5 = k5_basis(pd.DataFrame({"x" : THL_VH_test.sqrtStdDev}), 'sqrtStdDev', knots_VH_L3)
test_VH_sd_K5['df'].rename(columns={'Intercept':'sqrtStdDev_Intercept','Lin':'sqrtStdDev'},inplace=True)

training_VH_bases = pd.concat([training_VH_sd_K5['df'], THL_VH_train[['L2','FogYield']]],axis=1)
test_VH_bases = pd.concat([test_VH_sd_K5['df'], THL_VH_test['L2']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + fVH, data=training_VH_bases, return_type='dataframe')
X_test = dmatrix(fVH, data=test_VH_bases, return_type='dataframe')
VH_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='VH','L3'] = VH_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='VH','L3'] = VH_res.predict(X_test)

# THL==H
training_H_sd_K5 = k5_basis(pd.DataFrame({"x" : THL_H_train.sqrtStdDev}), 'sqrtStdDev', knots_H_L3)
training_H_sd_K5['df'].rename(columns={'Intercept':'sqrtStdDev_Intercept','Lin':'sqrtStdDev'},inplace=True)
test_H_sd_K5 = k5_basis(pd.DataFrame({"x" : THL_H_test.sqrtStdDev}), 'sqrtStdDev', knots_H_L3)
test_H_sd_K5['df'].rename(columns={'Intercept':'sqrtStdDev_Intercept','Lin':'sqrtStdDev'},inplace=True)

training_H_bases = pd.concat([training_H_sd_K5['df'], THL_H_train[['L2','FogYield']]],axis=1)
test_H_bases = pd.concat([test_H_sd_K5['df'], THL_H_test['L2']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + fH, data=training_H_bases, return_type='dataframe')
X_test = dmatrix(fH, data=test_H_bases, return_type='dataframe')
H_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='H','L3'] = H_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='H','L3'] = H_res.predict(X_test)

# THL==M (linear)
training_M_bases = THL_M_train[['sqrtStdDev','FogYield','L2']]
test_M_bases = THL_M_test[['sqrtStdDev','L2']]
y_train,X_train = dmatrices('FogYield ~ ' + fM, data=training_M_bases, return_type='dataframe')
X_test = dmatrix(fM, data=test_M_bases, return_type='dataframe')
M_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='M','L3'] = M_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='M','L3'] = M_res.predict(X_test)

# THL==dry
training_dry_sd_K3 = k3_basis(pd.DataFrame({"x" : THL_dry_train.sqrtStdDev}), 'sqrtStdDev', knots_dry_L3)
training_dry_sd_K3['df'].rename(columns={'Intercept':'sqrtStdDev_Intercept','Lin':'sqrtStdDev'},inplace=True)
test_dry_sd_K3 = k3_basis(pd.DataFrame({"x" : THL_dry_test.sqrtStdDev}), 'sqrtStdDev', knots_dry_L3)
test_dry_sd_K3['df'].rename(columns={'Intercept':'sqrtStdDev_Intercept','Lin':'sqrtStdDev'},inplace=True)

training_dry_bases = pd.concat([training_dry_sd_K3['df'], THL_dry_train[['L2','FogYield']]],axis=1)
test_dry_bases = pd.concat([test_dry_sd_K3['df'], THL_dry_test['L2']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + fdry, data=training_dry_bases, return_type='dataframe')
X_test = dmatrix(fdry, data=test_dry_bases, return_type='dataframe')
dry_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='dry','L3'] = dry_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='dry','L3'] = dry_res.predict(X_test)

# Save predictions (Micro layer 1)
micro_cols = ['DateTime','sqrtStdDev','THL','L2','L3']
#Micro_train[micro_cols].to_csv('./data/interim/Micro_L3_train.csv', index=False, float_format='%.6f')
#Micro_test[micro_cols].to_csv('./data/interim/Micro_L3_test.csv', index=False, float_format='%.6f')

# ---------------------------------------- Level 4: WindSpeed -------------------------------------------------- #
# VH (pass through L3), H (K3), M (K3), dry (linear)
# 'WindSpeed + L3 + WindSpeed * L3'
fVH = 'WindDir_shifted + WindDir_shifted1 + WindDir_shifted2 + WindDir_shifted3 + L3 + WindDir_shifted * L3'
f3 = 'wind_ms + wind_ms1 + wind_ms * L3 + L3'
fdry = 'wind_ms * L3 + L3'

knots_VH_L4 = [22.2875, 102.29583333334, 182.30416666668, 262.31250000002, 342.32083333336]
knots_H_L4 = [1.472054732222, 3.8225611161105, 6.173067499999]
knots_M_L4 = [0.5367178089634, 3.22644776191395, 5.9161777148645]

Micro_train.WindDir_shifted.fillna(method='ffill', inplace=True) # Missed a few

# Select rows THL=="dry", or M or H or VH:
THL_dry_train = Micro_train[Micro_train.THL=='dry']
THL_dry_test  = Micro_test[Micro_test.THL=='dry']
THL_M_train   = Micro_train[Micro_train.THL=='M']
THL_M_test    = Micro_test[Micro_test.THL=='M']
THL_H_train   = Micro_train[Micro_train.THL=='H']
THL_H_test    = Micro_test[Micro_test.THL=='H']
THL_VH_train  = Micro_train[Micro_train.THL=='VH']
THL_VH_test   = Micro_test[Micro_test.THL=='VH']

# THL==VH
Micro_train.loc[Micro_train.THL=='VH','L4'] = THL_VH_train['L3']
Micro_test.loc[Micro_test.THL=='VH','L4'] = THL_VH_test['L3']

THL_dry_train.wind_ms.fillna(method='ffill', inplace=True)

# THL==H
training_H_wd_K3 = k3_basis(pd.DataFrame({"x" : THL_H_train.wind_ms}), 'wind_ms', knots_H_L4)
training_H_wd_K3['df'].rename(columns={'Intercept':'wind_ms','Lin':'wind_ms'},inplace=True)
test_H_wd_K3 = k3_basis(pd.DataFrame({"x" : THL_H_test.wind_ms}), 'wind_ms', knots_H_L4)
test_H_wd_K3['df'].rename(columns={'Intercept':'wind_ms','Lin':'wind_ms'},inplace=True)

training_H_bases = pd.concat([training_H_wd_K3['df'], THL_H_train[['L3','FogYield']]],axis=1)
test_H_bases = pd.concat([test_H_wd_K3['df'], THL_H_test['L3']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f3, data=training_H_bases, return_type='dataframe')
X_test = dmatrix(f3, data=test_H_bases, return_type='dataframe')
H_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='H','L4'] = H_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='H','L4'] = H_res.predict(X_test)

# THL==M
training_M_wd_K3 = k3_basis(pd.DataFrame({"x" : THL_M_train.wind_ms}), 'wind_ms', knots_M_L4)
training_M_wd_K3['df'].rename(columns={'Intercept':'wind_ms','Lin':'wind_ms'},inplace=True)
test_M_wd_K3 = k3_basis(pd.DataFrame({"x" : THL_M_test.wind_ms}), 'wind_ms', knots_M_L4)
test_M_wd_K3['df'].rename(columns={'Intercept':'wind_ms','Lin':'wind_ms'},inplace=True)

training_M_bases = pd.concat([training_M_wd_K3['df'], THL_M_train[['L3','FogYield']]],axis=1)
test_M_bases = pd.concat([test_M_wd_K3['df'], THL_M_test['L3']],axis=1)
y_train,X_train = dmatrices('FogYield ~ ' + f3, data=training_M_bases, return_type='dataframe')
X_test = dmatrix(f3, data=test_M_bases, return_type='dataframe')
M_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='M','L4'] = M_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='M','L4'] = M_res.predict(X_test)

# THL==dry (linear)
training_dry_bases = THL_dry_train[['wind_ms','FogYield','L3']]
test_dry_bases = THL_dry_test[['wind_ms','L3']]
y_train,X_train = dmatrices('FogYield ~ ' + fdry, data=training_dry_bases, return_type='dataframe')
X_test = dmatrix(fdry, data=test_dry_bases, return_type='dataframe')
dry_res = sm.OLS(y_train, X_train, missing='drop').fit()
Micro_train.loc[Micro_train.THL=='dry','L4'] = dry_res.predict(X_train)
Micro_test.loc[Micro_test.THL=='dry','L4'] = dry_res.predict(X_test)

# Save predictions (Micro layer 1)
micro_cols = ['DateTime','wind_ms','WindDir_shifted','THL','L3','L4']
Micro_train[micro_cols].to_csv('./data/interim/Micro_L4_train.csv', index=False, float_format='%.6f')
Micro_test[micro_cols].to_csv('./data/interim/Micro_L4_test.csv', index=False, float_format='%.6f')
