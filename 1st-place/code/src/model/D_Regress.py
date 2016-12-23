# Date: May, 2016
# DrivenData.org Fog Net competition solution

from patsy import dmatrices, dmatrix
import pandas as pd
import numpy as np
import statsmodels.api as sm

# I've long been a fan of JMP(TM), but this exercise revealed that JMP 12 miscomputes the quantiles that define where to
# place the knots.  The rule JMP actually followed in selecting the knots was based on quantiles, but of the wrong data.
# JMP included rows in the calculation of the quantiles that were not among the subset selected for the regression.
#
# So the final rule for selecting knots was this:
#  1. Select the number of knots manually (default 5), by inspection of the fit (shape and stats)
#  2. Nominally, for K knots, select the 5th and 95th percentile, and also select K-2 equidistantly spaced intermediate locations.
#  3. Actually, consider the selection to have been manual, approximating that rule.
#     The difference is sometimes quite substantial and may account for some of my modeling frustrations!
#
# The model could probably be improved by setting the knots manually, not at regular intervals near where the curvature changes.
# This could all be replaced by the patsy cr() and cc() implementations of natural cubic regression splines, but
# that would require  some modeling effort, not just reprogramming. The result would be cleaner and more flexible.

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
# This assumes 4 knots!
def k4_basis(X,pre,knots):
    k = 4
    ts = []
    for i in range(0,k):
        t = np.maximum((X.x - knots[i]),0)**3
        ts.append(t)
    d4 = {"Intercept" : np.linspace(1,1,len(X.x)),
            "Lin" : X.x,
            pre + "1" : ts[0] - 3*ts[2] + 2*ts[3],
            pre + "2" : ts[1] - 2*ts[2] + 1*ts[3]}
    label = "Intercept + Lin + " + pre + "1 + " + pre + "2"
    df4 = pd.DataFrame(d4)
    return({"knots" : knots, "df" : df4})


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


# X must be a DataFrame containing a column named x
# This assumes 7 knots!
def k7_basis(X,pre,knots):
    k = 7
    ts = []
    for i in range(0,k):
        t = np.maximum((X.x - knots[i]),0)**3
        ts.append(t)
    d7 = {"Intercept" : np.linspace(1,1,len(X.x)),
            "Lin" : X.x,
            pre + "1" : ts[0] - 6*ts[5] + 5*ts[6],
            pre + "2" : ts[1] - 5*ts[5] + 4*ts[6],
            pre + "3" : ts[2] - 4*ts[5] + 3*ts[6],
            pre + "4" : ts[3] - 3*ts[5] + 2*ts[6],
            pre + "5" : ts[4] - 2*ts[5] + 1*ts[6]}
    label = "Intercept + Lin + " + pre + "1 + " + pre + "2 + " + pre + "3" + pre + "4" + pre + "5"
    df7 = pd.DataFrame(d7)
    return({"knots" : knots, "df" : df7})


# X must be a DataFrame containing a column named x
# This assumes 9 knots!
def k9_basis(X,pre,knots):
    k = 9
    ts = []
    for i in range(0,k):
        t = np.maximum((X.x - knots[i]),0)**3
        ts.append(t)
    d9 = {"Intercept" : np.linspace(1,1,len(X.x)),
            "Lin" : X.x,
            pre + "1" : ts[0] - 8*ts[7] + 7*ts[8],
            pre + "2" : ts[1] - 7*ts[7] + 6*ts[8],
            pre + "3" : ts[2] - 6*ts[7] + 5*ts[8],
            pre + "4" : ts[3] - 5*ts[7] + 4*ts[8],
            pre + "5" : ts[4] - 4*ts[7] + 3*ts[8],
            pre + "6" : ts[5] - 3*ts[7] + 2*ts[8],
            pre + "7" : ts[6] - 2*ts[7] + 1*ts[8]}
    label = "Intercept + Lin + " + pre + "1 + " + pre + "2 + " + pre + "3" + pre + "4" + pre + "5" + pre + "6" + pre + "7"
    df9 = pd.DataFrame(d9)
    return({"knots" : knots, "df" : df9})


# Select knot locations for a list -- per JMP documentation (see note at top of file)
def Select_knots(x,k):
    df = pd.DataFrame({"x" : x})
    return(np.linspace( df.x.quantile(.05), df.x.quantile(.95), k))



# -------------------------------------------- Read in data ---------------------------------------------------- #

# Training
SAG_train = pd.read_csv('./data/interim/Derived_SAG_train.csv', index_col=False)
DateFormat='%Y-%m-%d %H:%M:%S'
SAG_train.rename(columns={SAG_train.columns[0]:'DateTime'},inplace=True)
SAG_train.DateTime = pd.to_datetime(SAG_train.DateTime, format=DateFormat)

# Test
SAG_test = pd.read_csv('./data/interim/Derived_SAG_test.csv', index_col=False)
DateFormat='%Y-%m-%d %H:%M:%S'
SAG_test.rename(columns={SAG_test.columns[0]:'DateTime'},inplace=True)
SAG_test.DateTime = pd.to_datetime(SAG_test.DateTime, format=DateFormat)


# -------------------------------------------- Layer 0: "Diurnal" ---------------------------------------------- #
# This model lays down a simple oscillating function. Three actually. One is flat.
# One function is defined for each of three splits on SAG.DrySplit
# For dryish and other, knotted spline regressions. For dry, just the mean value.
# Splines for DayOfYear and Hour; 5 knots each.

####### First regression: DrySplit == "other" ########
DrySplit_other_train = SAG_train[SAG_train.DrySplit=='other'] # Select rows where DrySplit == "other"
DrySplit_other_test = SAG_test[SAG_test.DrySplit=='other']    # Select rows where DrySplit == "other"
knots_day = Select_knots(DrySplit_other_train.DayOfYear, 5)   # Define knots for DayOfYear
knots_hour = Select_knots(DrySplit_other_train.Hour, 5)       # Define knots for Hour

# Bug insertion. See note at top of file.
# In this case, the 5th quantile shifted from Day of year 21 to Day of year 22
knots_day[0] = knots_day[0]+1
knots_day[1] = knots_day[1]+0.75
knots_day[2] = knots_day[2]+0.5
knots_day[3] = knots_day[3]+0.25
# End bug insertion

# Construct training bases
training_day = pd.DataFrame({"x" : DrySplit_other_train.DayOfYear})
training_day_basis = k5_basis(training_day, "DayOfYear", knots_day)
training_hour = pd.DataFrame({"x" : DrySplit_other_train.Hour})
training_hour_basis = k5_basis(training_hour, "Hour", knots_hour)

# Repackage the two spline bases as one (df means degrees of freedom, not dataframe)
# (Note: redundant intercept column doesn't have any effect)
training_hour_basis['df'].rename(columns={'Intercept':'Hr_Intercept','Lin':'Hr_Lin'},inplace=True)
training_day_basis['df'].rename(columns={'Intercept':'Day_Intercept','Lin':'Day_Lin'},inplace=True)
DH_df = pd.concat([training_day_basis['df'],training_hour_basis['df']],axis=1)

# Regress
DH_res = sm.OLS(SAG_train['yield'][SAG_train.DrySplit=='other'],DH_df).fit() # create the OLS model object

# Save trained result for next layer
trained_other = pd.DataFrame({'DateTime':DrySplit_other_train.DateTime, 'Diurnal':DH_res.predict(DH_df)})

# Construct prediction bases
test_day = pd.DataFrame({"x" : DrySplit_other_test.DayOfYear})
test_day_basis = k5_basis(test_day, "DayOfYear", knots_day)
test_hour = pd.DataFrame({"x" : DrySplit_other_test.Hour})
test_hour_basis = k5_basis(test_hour, "Hour", knots_hour)
# Repackage bases
test_hour_basis['df'].rename(columns={'Intercept':'Hr_Intercept','Lin':'Hr_Lin'},inplace=True)
test_day_basis['df'].rename(columns={'Intercept':'Day_Intercept','Lin':'Day_Lin'},inplace=True)

DH_df = pd.concat([test_day_basis['df'],test_hour_basis['df']],axis=1)

# Save test result for next layer
test_other = pd.DataFrame({'DateTime':DrySplit_other_test.DateTime, 'Diurnal':DH_res.predict(DH_df)})


####### Second regression: DrySplit == "dryish" ######

DrySplit_dryish_train = SAG_train[SAG_train.DrySplit=='dryish'] # Select rows where DrySplit == "dryish"
DrySplit_dryish_test = SAG_test[SAG_test.DrySplit=='dryish']    # Select rows where DrySplit == "dryish"
knots_day = Select_knots(DrySplit_dryish_train.DayOfYear, 5)    # Define knots for DayOfYear
knots_hour = Select_knots(DrySplit_dryish_train.Hour, 5)        # Define knots for Hour

# Construct training bases
training_day = pd.DataFrame({"x" : DrySplit_dryish_train.DayOfYear})
training_day_basis = k5_basis(training_day, "DayOfYear", knots_day)
training_hour = pd.DataFrame({"x" : DrySplit_dryish_train.Hour})
training_hour_basis = k5_basis(training_hour, "Hour", knots_hour)

# Repackage the two spline bases as one (df means degrees of freedom, not dataframe)
# (Note: redundant intercept column doesn't have any effect)
training_hour_basis['df'].rename(columns={'Intercept':'Hr_Intercept','Lin':'Hr_Lin'},inplace=True)
training_day_basis['df'].rename(columns={'Intercept':'Day_Intercept','Lin':'Day_Lin'},inplace=True)
DH_df = pd.concat([training_day_basis['df'],training_hour_basis['df']],axis=1)

# Regress
DH_res = sm.OLS(SAG_train['yield'][SAG_train.DrySplit=='dryish'],DH_df).fit() # create the OLS model object

# Save trained result for next layer
trained_dryish = pd.DataFrame({'DateTime':DrySplit_dryish_train.DateTime, 'Diurnal':DH_res.predict(DH_df)})

# Construct prediction bases
test_day = pd.DataFrame({"x" : DrySplit_dryish_test.DayOfYear})
test_day_basis = k5_basis(test_day, "DayOfYear", knots_day)
test_hour = pd.DataFrame({"x" : DrySplit_dryish_test.Hour})
test_hour_basis = k5_basis(test_hour, "Hour", knots_hour)
# Repackage bases
test_hour_basis['df'].rename(columns={'Intercept':'Hr_Intercept','Lin':'Hr_Lin'},inplace=True)
test_day_basis['df'].rename(columns={'Intercept':'Day_Intercept','Lin':'Day_Lin'},inplace=True)

DH_df = pd.concat([test_day_basis['df'],test_hour_basis['df']],axis=1)

# Save test results for next layer
test_dryish = pd.DataFrame({'DateTime':DrySplit_dryish_test.DateTime, 'Diurnal':DH_res.predict(DH_df)})


####### Third split (dry) just predicts mean(yield), which is close to zero

grouped_train = SAG_train.groupby('DrySplit')
dry_mean = grouped_train.agg((np.mean))['yield']
pred_dry = pd.DataFrame({'DateTime':SAG_test.DateTime[SAG_test.DrySplit=="dry"], "pred":dry_mean['dry']})
trained_dry = pd.DataFrame({'DateTime':SAG_train.DateTime[SAG_train.DrySplit=="dry"], "Diurnal":dry_mean['dry']})
test_dry = pd.DataFrame({'DateTime':SAG_test.DateTime[SAG_test.DrySplit=="dry"], "Diurnal":dry_mean['dry']})

# Save predictions (Macro layer 0)
Diurnal = pd.concat([trained_other, trained_dryish, trained_dry],axis=0).sort_values('DateTime')
SAG_train['Diurnal'] = Diurnal.Diurnal
Diurnal = pd.concat([test_other, test_dryish, test_dry],axis=0).sort_values('DateTime')
SAG_test['Diurnal'] = Diurnal.Diurnal
SAG_train.to_csv('./data/interim/Diurnal.csv', index=False, float_format='%.6f')

# -------------------------------------------- END Layer 0: Diurnal -------------------------------------------- #



# -------------------------------------------- Layer 1: TTd_S, TTd_G, and Diurnal ------------------------------ #
# This layer builds on Diurnal by adding T-Td
# Both Sidi Ifni and Guelmim have useful opinions on this, so I fit them separately, then fuse the result with Diurnal.

# Sidi Ifni
# knots = Select_knots(SAG_train.TTd_Si, 7)  # Define knots per JMP documentation (intended)
knots = [1.1, 4.38270833333333, 7.66541666666667, 10.948125, 14.2308333333333, 17.5135416666667, 20.79625]

# Construct bases
training_TTd_S_K7 = k7_basis(pd.DataFrame({"x" : SAG_train.TTd_Si}), 'TTd_Si', knots)
test_TTd_S_K7 = k7_basis(pd.DataFrame({"x" : SAG_test.TTd_Si}), 'TTd_Si', knots)

# Regress
TTd_S_res = sm.OLS(SAG_train['yield'],training_TTd_S_K7['df']).fit() # create the OLS model object

# Save predictions for next layer
SAG_train['L1_TTd_S'] = TTd_S_res.predict(training_TTd_S_K7['df'])
SAG_test['L1_TTd_S'] = TTd_S_res.predict(test_TTd_S_K7['df'])


# Guelmim
#knots = Select_knots(SAG_train_nonans.TTd_Gu, 9)  # Define knots per JMP documentation
knots = [1, 4.95833333333333, 8.91666666666667, 12.875, 16.8333333333333, 20.7916666666667, 24.75, 28.7083333333333, 32.6666666666667]

# Construct bases
training_TTd_G_K9 = k9_basis(pd.DataFrame({"x" : SAG_train.TTd_Gu}), 'TTd_Gu', knots)
test_TTd_G_K9 = k9_basis(pd.DataFrame({"x" : SAG_test.TTd_Gu}), 'TTd_Gu', knots)

# Regress
TTd_G_res = sm.OLS(SAG_train['yield'],training_TTd_G_K9['df'],missing='drop').fit() # create the OLS model object

# Save predictions for next layer
SAG_train['L1_TTd_G'] = TTd_G_res.predict(training_TTd_G_K9['df'])
SAG_test['L1_TTd_G'] = TTd_G_res.predict(test_TTd_G_K9['df'])


# Fuse Sidi Ifni and Guelmim TTd predictions with previous layer (Diurnal)
# Construct bases
SAG_train['FogYield'] = SAG_train['yield'] # patsy bug?? "unexpected EOF while parsing" formula at word "yield"
SAG_test['FogYield'] = SAG_test['yield']

training_TTd_SG = SAG_train[['FogYield','L1_TTd_S','L1_TTd_G','Diurnal']]
test_TTd_SG = SAG_test[['FogYield','L1_TTd_S','L1_TTd_G','Diurnal']]

y_train,X_train = dmatrices('FogYield ~ L1_TTd_S + L1_TTd_G + Diurnal + L1_TTd_S * Diurnal + L1_TTd_G * Diurnal',data=training_TTd_SG, return_type='dataframe')
X_test = dmatrix('L1_TTd_S + L1_TTd_G + Diurnal + L1_TTd_S * Diurnal + L1_TTd_G * Diurnal',data=test_TTd_SG, return_type='dataframe')

# Regress
TTd_SG_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 1)
SAG_train.loc[np.isfinite(SAG_train['TTd_Gu']),'L1_TTd_SG'] = TTd_SG_res.predict(X_train)
SAG_test.loc[np.isfinite(SAG_test['TTd_Gu']),'L1_TTd_SG'] = TTd_SG_res.predict(X_test)
#SAG_train[['DateTime','Diurnal','L1_TTd_S','L1_TTd_G','L1_TTd_SG']].to_csv('./data/interim/Layer1.csv', index=False, float_format='%.6f')
#SAG_test[['DateTime','Diurnal','L1_TTd_S','L1_TTd_G','L1_TTd_SG']].to_csv('./data/interim/Layer1_test.csv', index=False, float_format='%.6f')
# -------------------------------------------- END Layer 1 TTd_SG ---------------------------------------------- #


# -------------------------------------------- Layer 2 RRR_S_any, cm_Gu_600 ------------------------------------ #
# Construct bases
SAG_train['cm_Gu_600_ord'] = map(lambda x: 'low' if x==1 else 'high', SAG_train['cm_Gu_600']) # ugly -- forcing categorical
SAG_test['cm_Gu_600_ord'] = map(lambda x: 'low' if x==1 else 'high', SAG_test['cm_Gu_600'])
training_RRR_cm = SAG_train[['FogYield','RRR_S_any','cm_Gu_600_ord','L1_TTd_SG']]
test_RRR_cm = SAG_test[['FogYield','RRR_S_any','cm_Gu_600_ord','L1_TTd_SG']]
y_train,X_train = dmatrices('FogYield ~ RRR_S_any + cm_Gu_600_ord + L1_TTd_SG + RRR_S_any * L1_TTd_SG + cm_Gu_600_ord * L1_TTd_SG',data=training_RRR_cm, return_type='dataframe')
X_test = dmatrix('RRR_S_any + cm_Gu_600_ord + L1_TTd_SG + RRR_S_any * L1_TTd_SG + cm_Gu_600_ord * L1_TTd_SG',data=test_RRR_cm, return_type='dataframe')

# Regress
RRR_cm_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 2)
SAG_train.loc[np.isfinite(SAG_train['TTd_Gu']),'L2_RRR_cm'] = RRR_cm_res.predict(X_train)
SAG_test.loc[np.isfinite(SAG_test['TTd_Gu']),'L2_RRR_cm'] = RRR_cm_res.predict(X_test)
#SAG_train[['DateTime','RRR_S_any','cm_Gu_600_ord','L2_RRR_cm','L1_TTd_SG']].to_csv('./data/interim/Layer2.csv', index=False, float_format='%.6f')
#SAG_test[['DateTime','RRR_S_any','cm_Gu_600_ord','L2_RRR_cm','L1_TTd_SG']].to_csv('./data/interim/Layer2_test.csv', index=False, float_format='%.6f')
# -------------------------------------------- END Layer 2 RRR_S_any, cm_Gu_600 -------------------------------- #




# -------------------------------------------- Layer 3 Nh_S_real, Sidi winds shifted --------------------------- #
knots = [40, 131.666666666667, 223.333333333333, 315]

# Construct bases
training_winds_K4 = k4_basis(pd.DataFrame({"x" : SAG_train.Sidi_winds_shifted}), 'Sidi_winds_shifted', knots)
test_winds_K4 = k4_basis(pd.DataFrame({"x" : SAG_test.Sidi_winds_shifted}), 'Sidi_winds_shifted', knots)

training_basis_L3 = pd.concat([training_winds_K4['df'], SAG_train[['FogYield','Nh_S_real','L2_RRR_cm']]], axis=1)
test_basis_L3 = pd.concat([test_winds_K4['df'], SAG_test[['FogYield','Nh_S_real','L2_RRR_cm']]], axis=1)

# In this formula, the first 4 terms are the knotted basis for Sidi_winds_shifted (and Lin==Sidi_winds_shifted)
f = 'Lin + Sidi_winds_shifted1 + Sidi_winds_shifted2 + Nh_S_real + L2_RRR_cm + Lin * L2_RRR_cm + Nh_S_real * L2_RRR_cm + Nh_S_real * Lin'

y_train,X_train = dmatrices('FogYield ~ ' + f, data=training_basis_L3, return_type='dataframe')
X_test = dmatrix(f, data=test_basis_L3, return_type='dataframe')

# Regress
winds_Nh_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 3)
SAG_train.loc[np.isfinite(SAG_train['TTd_Gu']),'L3_winds_Nh'] = winds_Nh_res.predict(X_train)
SAG_test.loc[np.isfinite(SAG_test['TTd_Gu']),'L3_winds_Nh'] = winds_Nh_res.predict(X_test)
#SAG_train[['DateTime','Sidi_winds_shifted','Nh_S_real','L3_winds_Nh','L2_RRR_cm']].to_csv('./data/interim/Layer3.csv', index=False, float_format='%.6f')
#SAG_test[['DateTime','Sidi_winds_shifted','Nh_S_real','L3_winds_Nh','L2_RRR_cm']].to_csv('./data/interim/Layer3_test.csv', index=False, float_format='%.6f')
# -------------------------------------------- END Layer 3 Nh_S_real, Sidi winds shifted ----------------------- #




# -------------------------------------------- Layer 4 P0_Ag, T_Ag --------------------------------------------- #

# Construct bases
knots = [2.34520787991171, 2.78664614948851, 3.2280844190653, 3.6695226886421, 4.11096095821889]
training_P0_K5 = k5_basis(pd.DataFrame({"x" : SAG_train.sqrt_P0_Ag}), 'sqrt_P0_Ag', knots)
training_P0_K5['df'].rename(columns={'Lin':'sqrt_P0_Ag'},inplace=True)
test_P0_K5 = k5_basis(pd.DataFrame({"x" : SAG_test.sqrt_P0_Ag}), 'sqrt_P0_Ag', knots)
test_P0_K5['df'].rename(columns={'Lin':'sqrt_P0_Ag'},inplace=True)

knots = [9, 13.75, 18.5, 23.25, 28]
training_T_K5 = k5_basis(pd.DataFrame({"x" : SAG_train.T_Ag_fc}), 'T_Ag_fc', knots)
training_T_K5['df'].rename(columns={'Lin':'T_Ag_fc'},inplace=True)
test_T_K5 = k5_basis(pd.DataFrame({"x" : SAG_test.T_Ag_fc}), 'T_Ag_fc', knots)
test_T_K5['df'].rename(columns={'Lin':'T_Ag_fc'},inplace=True)

training_basis_L5 = pd.concat([training_P0_K5['df'], training_T_K5['df'], SAG_train[['FogYield','L3_winds_Nh']]], axis=1)
test_basis_L5 = pd.concat([test_P0_K5['df'], test_T_K5['df'], SAG_test[['FogYield','L3_winds_Nh']]], axis=1)

# In this formula, the first 4 terms are the knotted basis for Sidi_winds_shifted (and Lin==Sidi_winds_shifted)
f = 'sqrt_P0_Ag + sqrt_P0_Ag1 + sqrt_P0_Ag2 + sqrt_P0_Ag3 + T_Ag_fc + T_Ag_fc1 + T_Ag_fc2 + T_Ag_fc3 + L3_winds_Nh + T_Ag_fc * L3_winds_Nh'

y_train,X_train = dmatrices('FogYield ~ ' + f, data=training_basis_L5, return_type='dataframe')
X_test = dmatrix(f, data=test_basis_L5, return_type='dataframe')

# Regress
PT_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 4)
SAG_train.loc[np.isfinite(SAG_train['TTd_Gu']) & np.isfinite(SAG_train['T_Ag_fc']),'L4_P0_T_Ag'] = PT_res.predict(X_train)
SAG_test.loc[np.isfinite(SAG_test['TTd_Gu']) & np.isfinite(SAG_test['T_Ag_fc']),'L4_P0_T_Ag'] = PT_res.predict(X_test)
#SAG_train[['DateTime','L4_P0_T_Ag']].to_csv('./data/interim/Layer4.csv', index=False, float_format='%.6f')
# -------------------------------------------- END Layer 4 P0_Ag, T_Ag ----------------------------------------- #




# -------------------------------------------- Layer 5 E_Si ---------------------------------------------------- #
# Construct bases
training_E_Si = SAG_train[['FogYield','E_Si3','L4_P0_T_Ag']]
test_E_Si = SAG_test[['FogYield','E_Si3','L4_P0_T_Ag']]
y_train,X_train = dmatrices('FogYield ~ E_Si3 + L4_P0_T_Ag + E_Si3 * L4_P0_T_Ag',data=training_E_Si, return_type='dataframe')
X_test = dmatrix('E_Si3 + L4_P0_T_Ag + E_Si3 * L4_P0_T_Ag',data=test_E_Si, return_type='dataframe')

# Regress
E_Si_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 5)
SAG_train.loc[np.isfinite(SAG_train['L4_P0_T_Ag']),'L5_E_Si'] = E_Si_res.predict(X_train)
SAG_test.loc[np.isfinite(SAG_test['L4_P0_T_Ag']),'L5_E_Si'] = E_Si_res.predict(X_test)
#SAG_train[['DateTime','L5_E_Si']].to_csv('./data/interim/Layer5.csv', index=False, float_format='%.6f')
# -------------------------------------------- END Layer 5 E_Si ------------------------------------------------ #




# -------------------------------------------- Layer 6 Td_Gu --------------------------------------------------- #
# Construct bases
knots = [-4, 1.25, 6.5, 11.75, 17]
training_Td_K5 = k5_basis(pd.DataFrame({"x" : SAG_train.Td_Gu_fc}), 'Td_Gu_fc', knots)
test_Td_K5 = k5_basis(pd.DataFrame({"x" : SAG_test.Td_Gu_fc}), 'Td_Gu_fc', knots)

training_basis_L5 = pd.concat([training_Td_K5['df'], SAG_train[['FogYield','L5_E_Si']]], axis=1)
test_basis_L5 = pd.concat([test_Td_K5['df'], SAG_test[['FogYield','L5_E_Si']]], axis=1)
f = 'Lin + Td_Gu_fc1 + Td_Gu_fc2 + Td_Gu_fc3 + L5_E_Si'

y_train,X_train = dmatrices('FogYield ~ ' + f, data=training_basis_L5, return_type='dataframe')
X_test = dmatrix(f, data=test_basis_L5, return_type='dataframe')

# Regress
Td_G_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 6)
SAG_train.loc[np.isfinite(SAG_train['L5_E_Si']),'L6_Td_G'] = Td_G_res.predict(X_train)
SAG_test.loc[np.isfinite(SAG_test['L5_E_Si']),'L6_Td_G'] = Td_G_res.predict(X_test)
#SAG_train[['DateTime','L6_Td_G']].to_csv('./data/interim/Layer6.csv', index=False, float_format='%.6f')
# -------------------------------------------- END Layer 6 Td_Gu ----------------------------------------------- #


# -------------------------------------------- Layer 7 clouds_G, Ff_G ------------------------------------------ #
# Construct bases
f = 'Ff_Gu_fc + clouds_G + L6_Td_G + Ff_Gu_fc * clouds_G + clouds_G * L6_Td_G'
training_Ff = SAG_train[['FogYield','Ff_Gu_fc','clouds_G','L6_Td_G']]
test_Ff = SAG_test[['FogYield','Ff_Gu_fc','clouds_G','L6_Td_G']]
y_train,X_train = dmatrices('FogYield ~ ' + f,data=training_Ff, return_type='dataframe')
X_test = dmatrix(f,data=test_Ff, return_type='dataframe')

# Regress
Ff_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 7)
SAG_train.loc[np.isfinite(SAG_train['L6_Td_G']),'L7_Ff'] = Ff_res.predict(X_train)
SAG_test.loc[np.isfinite(SAG_test['L6_Td_G']),'L7_Ff'] = Ff_res.predict(X_test)
#SAG_train[['DateTime','L7_Ff']].to_csv('./data/interim/Layer7.csv', index=False, float_format='%.6f')
#SAG_test[['DateTime','L7_Ff']].to_csv('./data/interim/Layer7_test.csv', index=False, float_format='%.6f')
# -------------------------------------------- END Layer 7 clouds_G, Ff_G -------------------------------------- #



# -------------------------------------------- Layer 8 Lag ----------------------------------------------------- #
# This last layer fuses the Layer 7 prediction with its own lag. In the exceptional case of the first test hour
# of each test interval, I use the previous yield rather than the previous prediction. Because of this exception,
# it seemed easier to combine test and training sets to grab those yields from SAG_train.
SAG_both = pd.concat([SAG_train, SAG_test], axis=0).sort_values('DateTime')	# note the "both" naming convention
SAG_both.reset_index(drop=True, inplace=True)
SAG_both.DateTime = pd.to_datetime(SAG_both.DateTime, format=DateFormat)
train = SAG_both.FogYield.apply(lambda x: np.isfinite(x))
test = ~train

# Only use the lag if it's two hours old. Otherwise (infrequently) substitute the current Layer 7 prediction for the lag.
elapsed = SAG_both['DateTime'] - SAG_both.shift()['DateTime']
previous_L7 = SAG_both.shift().L7_Ff
previous_yield = SAG_both.shift().FogYield
lags = []
for i in range(len(SAG_both)):
    if (i>0):
        if elapsed[i].days!=0:
            lags.append(SAG_both.L7_Ff[i])
        elif (elapsed[i].seconds//60)==120:
            if test[i] & np.isfinite(previous_yield[i]):
                lags.append(previous_yield[i])
            else:
                lags.append(previous_L7[i])
        else:
            lags.append(SAG_both.L7_Ff[i])
    else:
        lags.append(SAG_both.L7_Ff[i])
SAG_both['L7_Ff_lag'] = lags
SAG_trn = SAG_both[train]
SAG_tst = SAG_both[test]

# Regress
training_lag = SAG_trn[['FogYield','L7_Ff','L7_Ff_lag']]
y_train,X_train = dmatrices('FogYield ~ L7_Ff + L7_Ff_lag',data=training_lag, return_type='dataframe')
Lag_res = sm.OLS(y_train, X_train, missing='drop').fit()

# Save predictions (Macro layer 8)
both_lag = SAG_both[['FogYield','L7_Ff','L7_Ff_lag']]
X_both = dmatrix('L7_Ff + L7_Ff_lag',data=both_lag, return_type='dataframe')
SAG_both.loc[np.isfinite(SAG_both['L7_Ff']) & np.isfinite(SAG_both['L7_Ff_lag']),'L8_lag'] = Lag_res.predict(X_both)
SAG_both[['DateTime','L7_Ff','L7_Ff_lag','L8_lag']][test].to_csv('./data/interim/Layer8_test.csv', index=False, float_format='%.6f')
SAG_both[['DateTime','L7_Ff','L7_Ff_lag','L8_lag']][train].to_csv('./data/interim/Layer8_train.csv', index=False, float_format='%.6f')

# -------------------------------------------- END Layer 8 Lag ------------------------------------------------- #
