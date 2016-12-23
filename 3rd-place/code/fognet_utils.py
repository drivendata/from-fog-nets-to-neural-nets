#!/usr/bin/python

'''
Based on https://www.kaggle.com/justdoit/rossmann-store-sales/xgboost-in-python-with-rmspe/code
Public Score :  not submitted
Private Validation Score :  0.114779
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
#matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import locale
#locale.setlocale(locale.LC_ALL,'en_US')
import h5py
import cPickle, gzip, os, glob, os.path
import cPickle as pickle
import copy

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import MiniBatchKMeans, KMeans     
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor
import ephem

import sys
from _sqlite3 import Row
np.random.seed(0)


regressors_dict = dict()

def toBinary(featureCol, df):
    values = set(df[featureCol].unique())
    newCol = [ "{}_{}".format(featureCol, val) for val in values]
    for val in values:
        df["{}_{}".format(featureCol, val)] = df[featureCol].map(lambda x: 1 if x == val else 0)
    return newCol

def rmse_scoring(regressor, x2score, y_true):
        y_hat = regressor.predict(x2score)
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        return rmse * -1



def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")      


# compute various sun-related features
# at the approximate fognet location

def sun_at_fognets():
    full_period=pd.date_range('2013-11-23 16:00:00',periods=9262, freq='2H')
    fognets = ephem.Observer()
    fognets.lon = str(-10.0125)
    fognets.lat = str(29.231)
    fognets.elev = 480
    fognets.pressure = 0
    fognets.horizon = 0
    sun_angles = []         # sun angle normalized -1.0 - 1.0
    hours_of_sun = []       # number of hours of sun since sunrise
    hours_of_night = []     # number of hours of night since sunset
    adjusted_hours_sun = [] # hours of sun adjusted with angle
    is_day = []             # True if day
    
    cumulated_hours_of_sun = 0.0
    for ts in full_period:
        fognets.date = ts
        sun = ephem.Sun(fognets)
        sun.compute(fognets)
        sun_angle = float(sun.alt) / (np.pi / 2.0)
        sunrise=fognets.previous_rising(ephem.Sun()).datetime() 
        noon   =fognets.next_transit   (ephem.Sun(), start=sunrise).datetime() 
        sunset =fognets.next_setting   (ephem.Sun(), start=sunrise).datetime()
        sun_angles.append(sun_angle)
        if (ts < sunset): # day
            cumulated_hours_of_sun += ((ts - sunrise).total_seconds()/3600) * sun_angle
            #print ts, (ts - sunrise).total_seconds()/3600,0, (sunset - sunrise).total_seconds()/3600, sun_angle, cumulated_hours_of_sun, "day"
            hours_of_sun.append((ts - sunrise).total_seconds()/3600)
            hours_of_night.append(0.0)
            adjusted_hours_sun.append(cumulated_hours_of_sun)
            is_day.append(True)
        else:
            cumulated_hours_of_sun = 0.0
            #print ts, 0, (ts - sunset).total_seconds()/3600, (sunset - sunrise).total_seconds()/3600, sun_angle, cumulated_hours_of_sun, "night"            
            hours_of_sun.append(0.0)
            hours_of_night.append((ts - sunset).total_seconds()/3600)
            adjusted_hours_sun.append(0.0)
            is_day.append(False)
    df = pd.DataFrame()
    df['utc_time'] = full_period
    df['sun_angle'] = pd.DataFrame(sun_angles)
    df['hours_of_sun'] = pd.DataFrame(hours_of_sun)
    df['hours_of_night'] = pd.DataFrame(hours_of_night)
    df['adjusted_hours_of_sun'] = pd.DataFrame(adjusted_hours_sun)
    df['is_day'] = pd.DataFrame(is_day)
    df = df.set_index('utc_time')
    return df
    


def WindParse(df,colname):
    df.loc[df[colname] == 'Calm, no wind', 'WindDirection'] = -1
    df.loc[df[colname] == 'Wind blowing from the north', 'WindDirection'] = 0
    df.loc[df[colname] == 'Wind blowing from the north-northeast', 'WindDirection'] = 1
    df.loc[df[colname] == 'Wind blowing from the north-east', 'WindDirection'] = 2
    df.loc[df[colname] == 'Wind blowing from the east-northeast', 'WindDirection'] = 3
    df.loc[df[colname] == 'Wind blowing from the east', 'WindDirection'] = 4
    df.loc[df[colname] == 'Wind blowing from the east-southeast', 'WindDirection'] = 5
    df.loc[df[colname] == 'Wind blowing from the south-east', 'WindDirection'] = 6
    df.loc[df[colname] == 'Wind blowing from the south-southeast', 'WindDirection'] = 7
    df.loc[df[colname] == 'Wind blowing from the south', 'WindDirection'] = 8
    df.loc[df[colname] == 'Wind blowing from the south-southwest', 'WindDirection'] = 9
    df.loc[df[colname] == 'Wind blowing from the south-west', 'WindDirection'] = 10
    df.loc[df[colname] == 'Wind blowing from the west-southwest', 'WindDirection'] = 11
    df.loc[df[colname] == 'Wind blowing from the west', 'WindDirection'] = 12
    df.loc[df[colname] == 'Wind blowing from the west-northwest', 'WindDirection'] = 13
    df.loc[df[colname] == 'Wind blowing from the north-west', 'WindDirection'] = 14
    df.loc[df[colname] == 'Wind blowing from the north-northwest', 'WindDirection'] = 15


# Parse wind directions into 3 factors
def WindParse2(df,colname):
    df.loc[df[colname] == 'Calm, no wind', 'WindDirection1'] = 'nowind'
    df.loc[df[colname] == 'Calm, no wind', 'WindDirection2'] = 'nowind'
    df.loc[df[colname] == 'Calm, no wind', 'WindDirection3'] = 'nowind'
    df.loc[df[colname] == 'Wind blowing from the north', 'WindDirection1'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north', 'WindDirection2'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north', 'WindDirection3'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north-northeast', 'WindDirection1'] = 'NNE'
    df.loc[df[colname] == 'Wind blowing from the north-northeast', 'WindDirection2'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north-northeast', 'WindDirection3'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the north-east', 'WindDirection1'] = 'NE'
    df.loc[df[colname] == 'Wind blowing from the north-east', 'WindDirection2'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north-east', 'WindDirection3'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the east-northeast', 'WindDirection1'] = 'ENE'
    df.loc[df[colname] == 'Wind blowing from the east-northeast', 'WindDirection2'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the east-northeast', 'WindDirection3'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the east', 'WindDirection1'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the east', 'WindDirection2'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the east', 'WindDirection3'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the east-southeast', 'WindDirection1'] = 'ESE'
    df.loc[df[colname] == 'Wind blowing from the east-southeast', 'WindDirection2'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the east-southeast', 'WindDirection3'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south-east', 'WindDirection1'] = 'SE'
    df.loc[df[colname] == 'Wind blowing from the south-east', 'WindDirection2'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south-east', 'WindDirection3'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the south-southeast', 'WindDirection1'] = 'SSE'
    df.loc[df[colname] == 'Wind blowing from the south-southeast', 'WindDirection2'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south-southeast', 'WindDirection3'] = 'E'
    df.loc[df[colname] == 'Wind blowing from the south', 'WindDirection1'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south', 'WindDirection2'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south', 'WindDirection3'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south-southwest', 'WindDirection1'] = 'SSW'
    df.loc[df[colname] == 'Wind blowing from the south-southwest', 'WindDirection2'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south-southwest', 'WindDirection3'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the south-west', 'WindDirection1'] = 'SW'
    df.loc[df[colname] == 'Wind blowing from the south-west', 'WindDirection2'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the south-west', 'WindDirection3'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the west-southwest', 'WindDirection1'] = 'WSW'
    df.loc[df[colname] == 'Wind blowing from the west-southwest', 'WindDirection2'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the west-southwest', 'WindDirection3'] = 'S'
    df.loc[df[colname] == 'Wind blowing from the west', 'WindDirection1'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the west', 'WindDirection2'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the west', 'WindDirection3'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the west-northwest', 'WindDirection1'] = 'WNW'
    df.loc[df[colname] == 'Wind blowing from the west-northwest', 'WindDirection2'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the west-northwest', 'WindDirection3'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north-west', 'WindDirection1'] = 'NW'
    df.loc[df[colname] == 'Wind blowing from the north-west', 'WindDirection2'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north-west', 'WindDirection3'] = 'W'
    df.loc[df[colname] == 'Wind blowing from the north-northwest', 'WindDirection1'] = 'NNW'
    df.loc[df[colname] == 'Wind blowing from the north-northwest', 'WindDirection2'] = 'N'
    df.loc[df[colname] == 'Wind blowing from the north-northwest', 'WindDirection3'] = 'W'


# Prase cloud cover as a numeric feature
def CloudCoverParse(df,colname, dest_column = 'CloudCover'):
    df.loc[df[colname] == 'no clouds', dest_column] = 0
    df.loc[df[colname] == '10%  or less, but not 0', dest_column] = 5
    df.loc[df[colname] == '20-30%', dest_column] = 25
    df.loc[df[colname] == '40%', dest_column] = 40
    df.loc[df[colname] == '50%', dest_column] = 50
    df.loc[df[colname] == '60%', dest_column] = 60
    df.loc[df[colname] == '70 - 80%', dest_column] = 75  
    df.loc[df[colname] == '90  or more, but not 100%', dest_column] = 95    
    df.loc[df[colname] == '100%', dest_column] = 100

# Prase cloud density as a numeric feature
def CloudDensityParse(df,colname):
    df.loc[df[colname] == 'Overcast_(100%)', 'CloudDensity'] = 100
    df.loc[df[colname] == 'Broken_clouds_(60-90%)', 'CloudDensity'] = 75
    df.loc[df[colname] == 'Scattered_clouds_(40-50%)', 'CloudDensity'] = 45
    df.loc[df[colname] == 'Few_clouds_(10-30%)', 'CloudDensity'] = 20
    df.loc[df[colname] == 'No_Clouds_(0-0%)', 'CloudDensity'] = 0


# Prase cloud height as a numeric feature
def CloudHeigthParse(df,colname):
    df.loc[df[colname] == '50-100', 'CloudHeight'] = 75
    df.loc[df[colname] == '100-200', 'CloudHeight'] = 150
    df.loc[df[colname] == '200-300', 'CloudHeight'] = 250
    df.loc[df[colname] == '300-600', 'CloudHeight'] = 450
    df.loc[df[colname] == '600-1000', 'CloudHeight'] = 800
    df.loc[df[colname] == '2000-2500', 'CloudHeight'] = 2250
    df.loc[df[colname] == '2500 or more, or no clouds.', 'CloudHeight'] = 2500
    df.loc[df[colname] == '2500 or more, or no clouds', 'CloudHeight'] = 2500


# Prase weather as a numeric feature
# The higher the number the wetter the weather
# the conversion is entirely arbitrary
def WeatherParse(df,colname):
    """
"Light rain, fog"
"Light shower(s), rain"
"Light thunderstorm, rain"
"Mist, light rain"
"Mist, rain"
"Rain, mist"
"Rain, shower(s)"
"Shower(s), rain"
"Thunderstorm, hail"
"Thunderstorm, hail, rain"
"Thunderstorm, light rain"
"Thunderstorm, light shower(s), rain"
"Thunderstorm, rain"
"Thunderstorm, shower(s)"
"Thunderstorm, shower(s), rain"
Drizzle
Duststorm
Fog
Haze
Heavy rain
In the vicinity shower(s)
Light drizzle
Light duststorm
Light rain
Mist
Rain
Sand
Shower(s)
Smoke
Thunderstorm
    """
    df.loc[df[colname].isnull(), 'RainLevel'] = 0
    df.loc[df[colname] == 'Duststorm', 'RainLevel']                = 0
    df.loc[df[colname] == 'Widespread dust', 'RainLevel']          = 0
    df.loc[df[colname] == 'Sand', 'RainLevel']                     = 0
    df.loc[df[colname] == 'Smoke', 'RainLevel']                    = 0
    df.loc[df[colname] == 'Light duststorm', 'RainLevel']          = 0
    df.loc[df[colname] == 'Haze', 'RainLevel']                     = 0
    df.loc[df[colname] == 'Fog', 'RainLevel']                      = 5
    df.loc[df[colname] == 'Light rain, fog', 'RainLevel']          = 10
    df.loc[df[colname] == 'Light drizzle', 'RainLevel']            = 10
    df.loc[df[colname] == 'Mist', 'RainLevel']                     = 10
    df.loc[df[colname] == 'Thunderstorm', 'RainLevel']             = 10
    df.loc[df[colname] == 'Mist, light rain', 'RainLevel']         = 20
    df.loc[df[colname] == 'Light rain', 'RainLevel']               = 20
    df.loc[df[colname] == 'Drizzle', 'RainLevel']                  = 20
    df.loc[df[colname] == 'Mist, rain', 'RainLevel']               = 30
    df.loc[df[colname] == 'Rain, mist', 'RainLevel']               = 30
    df.loc[df[colname] == 'Thunderstorm, light rain', 'RainLevel'] = 30
    df.loc[df[colname] == 'Shower(s)', 'RainLevel']                = 30
    df.loc[df[colname] == 'Rain', 'RainLevel']                     = 40    
    df.loc[df[colname] == 'Light shower(s), rain', 'RainLevel']    = 40
    df.loc[df[colname] == 'In the vicinity shower(s)', 'RainLevel']= 40
    df.loc[df[colname] == 'Thunderstorm, hail', 'RainLevel']       = 40
    df.loc[df[colname] == 'Thunderstorm, light shower(s), rain', 'RainLevel']       = 40
    df.loc[df[colname] == 'Rain, shower(s)', 'RainLevel']          = 45
    df.loc[df[colname] == 'Light thunderstorm, rain', 'RainLevel'] = 50
    df.loc[df[colname] == 'Thunderstorm, hail, rain', 'RainLevel'] = 60
    df.loc[df[colname] == 'Thunderstorm, rain', 'RainLevel']       = 60    
    df.loc[df[colname] == 'Shower(s), rain', 'RainLevel']          = 75
    df.loc[df[colname] == 'Thunderstorm, shower(s)', 'RainLevel']  = 75
    df.loc[df[colname] == 'Thunderstorm, shower(s), rain', 'RainLevel']          = 75
    df.loc[df[colname] == 'Heavy rain', 'RainLevel']               = 90


def TruncateTimeStamp2Hours(ts):
    dt_hour = ts.hour
    dt_minute = ts.minute    
    if ((dt_hour % 2) == 0) and ( dt_minute > 0 ):
        dt_hour_limit = dt_hour + 2
    elif ((dt_hour % 2) == 0) and ( dt_minute == 0 ):
        dt_hour_limit = dt_hour
    else:
        dt_hour_limit = dt_hour + 1
        
    dt_hour_limit = dt_hour_limit % 24
    dt_ts_limit  = ts.replace(hour = dt_hour_limit, minute = 0)
    return dt_ts_limit


def TruncateTimeStamp4Hours(ts):
    dt_hour = ts.hour
    dt_minute = ts.minute
    hour_reminder = dt_hour % 4    
    if (hour_reminder == 0) and ( dt_minute > 0 ):
        dt_hour_limit = dt_hour + 4
    elif (hour_reminder == 0) and ( dt_minute == 0 ):
        dt_hour_limit = dt_hour
    else:
        dt_hour_limit = dt_hour + ( 4 - hour_reminder )
        
    dt_hour_limit = dt_hour_limit % 24
    dt_ts_limit  = ts.replace(hour = dt_hour_limit, minute = 0)
    return dt_ts_limit


def TruncateTimeStampHours(ts, n_hours, hour_offset):
    dt_hour = ts.hour - hour_offset
    dt_minute = ts.minute
    hour_reminder = dt_hour % n_hours    
    if (hour_reminder == 0) and ( dt_minute > 0 ):
        dt_hour_limit = dt_hour + n_hours
    elif (hour_reminder == 0) and ( dt_minute == 0 ):
        dt_hour_limit = dt_hour
    else:
        dt_hour_limit = dt_hour + ( n_hours - hour_reminder )
    
    dt_hour_limit += hour_offset
    dt_hour_limit = dt_hour_limit % 24
    dt_ts_limit  = ts.replace(hour = dt_hour_limit, minute = 0)
    return dt_ts_limit


# the idea was to compute predictions based on aggregated time ranged
# but I realized it is completely because, and it injects future data
# however, this way I had the 3.05 score
# For each aggregate the min + mean + max values are computed
def GenerateGroupsBy(df,df_name, hours_grouped = 4):
    
        # Prepare groups by 2 hours step
        print("Generate Groups for", df_name, "by", hours_grouped,"hours")
        n_offsets = hours_grouped / 2
        print("There are", n_offsets, "offsets")
        grouped_offsets = []
        for hour_offsets in xrange(n_offsets):
            df_group = df.groupby(lambda x: TruncateTimeStampHours(x,hours_grouped,(hour_offsets -1) * 2))
            grouped_offsets.append(df_group)


        grouped_set = []            
        for group_function in ["mean","max","min"]:
            print("Apply functon", group_function, "to each group")
            grouped_results = []
            for df_group in grouped_offsets:
                df_grouped_group = df_group.apply(getattr(np,group_function)).add_prefix('%s_%s' % (df_name, group_function))
                grouped_results.append(df_grouped_group)
                #print(df_grouped_group)
            grouped_set.append(pd.concat(grouped_results).sort_index())



        return grouped_set


def compute_rmse(y_true, y_hat):
    y_hat[ y_hat < 0 ] = 0
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    return rmse
            

def plot_errors(y_true, y_hat, df_model, valid_indices):        
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(y_true, y_hat)
    
    plt.xlabel('actual', fontsize=20)
    plt.ylabel('predicted', fontsize=20)
    plt.plot(np.linspace(0, 35), np.linspace(0, 35), label="$y=x$")
    
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.legend(loc='upper left', fontsize=20)
    plt.show()
    
    
    fig, ax = plt.subplots(figsize=(16, 4))
    err = y_true - y_hat
    
    print(df_model[df_model['is_train'] == True].reset_index().columns)
    valid_dates = df_model[df_model['is_train'] == True].reset_index()['index'][valid_indices]
    ax.plot_date( valid_dates, err, c='r', ms=3)
    ax.set_title('residuals on test data (each)', fontsize=20)
    ax.set_ylabel('error')
    plt.show()
                
    fig, ax = plt.subplots(figsize=(16, 4))
    plt.hist(err, bins=20, normed=True)
    plt.title('residuals on test data (distribution)', fontsize=20)
    plt.xlim(-20, 20)
    
    plt.show()

def plot_importances(evaluation_list, set_to_plot, columns_selection = None):
        
    for evaluation_set in evaluation_list:
        (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model, pred_test, best_params) = evaluation_set
        
        if set_name != set_to_plot:
            continue
        
        if columns_selection is None:
            columns_selection = list(work_model.columns)
            columns_selection.remove('yield')
            if 'yield_%s' % set_name in columns_selection: 
                columns_selection.remove('yield_%s' % set_name)
            columns_selection.remove('is_train')
            columns_selection_nocorr = remove_correlated_features(work_model, columns_selection)
            columns_selection = columns_selection_nocorr
        
        X = work_model.loc[work_model['is_train'] == True, columns_selection ].as_matrix().astype(np.float32)
        y = work_model.loc[work_model['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
        
        x_train = X.copy()
        vec_res = y.copy()
        
        train_indices = np.logical_not(valid_indices)
        
        train_set_x_tot, train_set_y_tot = x_train[train_indices], vec_res[train_indices]
        
        print(work_model.loc[work_model['is_train'] == True, columns_selection ][train_indices].isnull().sum(axis=0))
        
        n_features = train_set_x_tot.shape[1]     
        print "build forest"
        #forest = ExtraTreesClassifier(n_estimators=60, random_state=0, n_jobs = -1,  min_samples_leaf = 2, max_depth= 15, max_features=None, verbose = 1)
        forest = ExtraTreesRegressor(n_estimators=1000, random_state=0, n_jobs = -1,  min_samples_leaf = 2, max_depth= 20, max_features=None, verbose = 1)
        
        print "fit forest"
        forest.fit(train_set_x_tot, train_set_y_tot)
        
        print "compute importances"
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        
        # Print the feature ranking
        print("Feature ranking:")
        
        for f in range(n_features):
            print("%d. feature %s (%f)" % (f + 1, columns_selection[indices[f]], importances[indices[f]]))
        
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(n_features), importances[indices],color="r", yerr=std[indices], align="center")
        plt.xticks(range(n_features), indices)
        plt.xlim([-1, n_features])
        plt.show()

# we have a list of models
# For each one, we compute an rmse on the test set
# on each interval on which 2 models compete, the one with
# the best rmse will be kept
# this function is deprecated in favor of the next one
def compare_evaluations(evaluation_list, sub_form):

    if 0:
        sub_form = lin_model[lin_model['is_train'] == True]
        sub_form = sub_form[ (sub_form.index.day <= 4) & (sub_form.index.day >= 1)]
        
        sub_form = sub_form.join(agadir_model_train.loc[valid_indices,'yield_agadir_predicted'],how='left')
    
    for evaluation_set in evaluation_list:
        (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model, pred_test, best_params) = evaluation_set
        pred_valid[ pred_valid < 0 ] = 0.0
        train_model.loc[valid_indices, 'yield_%s_predicted' % set_name] = pred_valid
        
        y_hat = pred_valid
        rmse = compute_rmse(valid_set_y, y_hat)
        print("rmse %s =" % (set_name), rmse)
        
        sub_form = sub_form.join(train_model.loc[valid_indices,'yield_%s_predicted' % set_name],how='left')


    idx_set_1 = 0
    
    comparison_list = []
    for evaluation_set in evaluation_list:
        (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model, pred_test, best_params) = evaluation_set
        idx_set_2 = 0
            
        for evaluation_set2 in evaluation_list:
            if idx_set_2 <= idx_set_1:
                idx_set_2 += 1
                continue
            
            (pred_valid2, valid_set_y2, train_model2, valid_indices2, set_name2, work_model, pred_test, best_params) = evaluation_set2
        

            
            yield1_name = 'yield_%s_predicted' % set_name
            yield2_name = 'yield_%s_predicted' % set_name2
            sub_form_common = sub_form.loc[np.logical_not(sub_form[yield1_name].isnull()) & np.logical_not(sub_form[yield2_name].isnull()),['yield', yield1_name, yield2_name ] ].copy()
            

            print("%s and %s" % ( set_name, set_name2),"( %i x %i )" % (len(pred_valid), len(pred_valid2)), "%i common" % sub_form_common.shape[0] )
            
            y_hat = sub_form_common[yield1_name]
            rmse1 = compute_rmse(sub_form_common['yield'], y_hat)
            #print("%s x %s: rmse %s =" % (set_name, set_name2, set_name), rmse1)
            

            y_hat = sub_form_common[yield2_name]
            rmse2 = compute_rmse(sub_form_common['yield'], y_hat)
            print("%s x %s => " % (set_name, set_name2), rmse1/rmse2, rmse1, rmse2)
            
            comparison_list.append((set_name, set_name2, rmse1, rmse2, sub_form_common.shape[0]))
            
            idx_set_2 += 1
        idx_set_1 += 1
    return comparison_list


# we have a list of models
# For each one, we compute an rmse on the 4 folds of the cv_list
# on each interval on which 2 models compete, the one with
# the best rmse will be kept
def compare_evaluations2(evaluation_list, sub_form, cv_list):
    from sklearn import clone
    global regressors_dict
    
    
    for evaluation_set in evaluation_list:
        (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model, pred_test, best_params) = evaluation_set
        pred_valid[ pred_valid < 0 ] = 0.0
        train_model.loc[valid_indices, 'yield_%s_predicted' % set_name] = pred_valid
        
        y_hat = pred_valid
        rmse = compute_rmse(valid_set_y, y_hat)
        print("rmse %s =" % (set_name), rmse)
        
        sub_form = sub_form.join(train_model.loc[valid_indices,'yield_%s_predicted' % set_name],how='left')


    idx_set_1 = 0
    
    comparison_list = []
    for evaluation_set in evaluation_list:
        (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model1, pred_test1, best_params1) = evaluation_set
        idx_set_2 = 0

        df_model_train = train_model.copy()
        
        valid_indicesA = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[0]:
            valid_indicesA |= ( df_model_train.index == ts)
    
        cv_fold_1 = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[1]:
            cv_fold_1 |= ( df_model_train.index == ts)
    
        cv_fold_2 = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[2]:
            cv_fold_2 |= ( df_model_train.index == ts)
    
        cv_fold_3 = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[3]:
            cv_fold_3 |= ( df_model_train.index == ts)
        
        custom_cv1 = [ (cv_fold_1,  cv_fold_2 | cv_fold_3 | valid_indicesA), (cv_fold_2, cv_fold_1 | cv_fold_3 | valid_indicesA ), (cv_fold_3, cv_fold_1 | cv_fold_2 | valid_indicesA ), (valid_indicesA,  cv_fold_2 | cv_fold_3 |  cv_fold_1 ) ]


        if set_name in regressors_dict:
            (estimator1_list, custom_cv, columns_selection1) = regressors_dict[set_name]
        else:
            regressors_file = "fognet_%s_regressors.gz" % set_name
            if os.path.isfile(regressors_file):
                print("Loading saved regressors from %s" % regressors_file)
                f = gzip.open( regressors_file,"r")
                (estimator1_list, custom_cv, columns_selection1) = pickle.load(f)
                f.close()
                regressors_dict[set_name] = (estimator1_list, custom_cv, columns_selection1)
            else:
                columns_selection1 = list(work_model1.columns)
                columns_selection1.remove('yield')
                columns_selection1.remove('is_train')                                    
                columns_selection1_nocorr = remove_correlated_features(work_model1, columns_selection1)
                columns_selection1 = columns_selection1_nocorr
                estimator1_list = []
             
        X1 = work_model1.loc[work_model1['is_train'] == True, columns_selection1 ].as_matrix().astype(np.float32)
        X_pred1 = work_model1.loc[work_model1['is_train'] == False, columns_selection1 ].as_matrix().astype(np.float32)
        y1 = work_model1.loc[work_model1['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
        

        
        if len(estimator1_list) == 0:
            regressor_base = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )                    
            for fold_idx in xrange(4):
                print("Fold:", fold_idx + 1, "/4")
    
                
                valid_indices = custom_cv1[fold_idx][0]
                train_indices = custom_cv1[fold_idx][1]
    
                best_params1.pop('best_iteration', None)
                best_estimator = clone(regressor_base).set_params(**best_params1)
    
                train_set_x_tot = X1[train_indices].copy()
                train_set_y_tot = y1[train_indices].copy()
                valid_set_x = X1[valid_indices].copy()
                valid_set_y = y1[valid_indices].copy()
    
                eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]
            
                best_estimator.fit(train_set_x_tot, train_set_y_tot, eval_metric = 'rmse', eval_set = eval_set, early_stopping_rounds=50, verbose = False)
                
                estimator1_list.append(best_estimator)
    
            f = gzip.open( regressors_file,"wb")
            cPickle.dump((estimator1_list, custom_cv1, columns_selection1), f, cPickle.HIGHEST_PROTOCOL)
            f.close()
            regressors_dict[set_name] = (estimator1_list, custom_cv1, columns_selection1)
            
        for evaluation_set2 in evaluation_list:
            if idx_set_2 <= idx_set_1:
                idx_set_2 += 1
                continue
            
            (pred_valid2, valid_set_y2, train_model2, valid_indices2, set_name2, work_model2, pred_test2, best_params2) = evaluation_set2

            yield1_name = 'yield_%s_predicted' % set_name
            yield2_name = 'yield_%s_predicted' % set_name2

            sub_form_common = sub_form.loc[np.logical_not(sub_form[yield1_name].isnull()) & np.logical_not(sub_form[yield2_name].isnull()),['yield', yield1_name, yield2_name ] ].copy()
            

            print("%s and %s" % ( set_name, set_name2),"( %i x %i )" % (len(pred_valid), len(pred_valid2)), "%i common" % sub_form_common.shape[0] )
        

            df_model_train = train_model2.copy()
        
            valid_indicesB = np.zeros(len(train_model2), dtype=bool)
            valid_indicesCommonA = np.zeros(len(train_model), dtype=bool)
            valid_indicesCommonB = np.zeros(len(train_model2), dtype=bool)
            
            for ts in cv_list[0]:
                valid_indicesB |= ( train_model2.index == ts)
                if np.sum( ( train_model2.index == ts) ) == 1:
                    valid_indicesCommonA |= ( train_model.index == ts)
                if np.sum( ( train_model.index == ts) ) == 1:
                    valid_indicesCommonB |= ( train_model2.index == ts)

            if (np.sum(valid_indicesCommonA) != np.sum(valid_indicesCommonB)):
                print("Internal error: valid common indices mismatch",np.sum(valid_indicesCommonA), np.sum(valid_indicesCommonB) )

            cv_fold_1_indicesCommonA = np.zeros(len(train_model), dtype=bool)
            cv_fold_1_indicesCommonB = np.zeros(len(train_model2), dtype=bool)
        
            cv_fold_1 = np.zeros(len(df_model_train), dtype=bool)
            for ts in cv_list[1]:
                cv_fold_1 |= ( df_model_train.index == ts)
                if np.sum( ( train_model2.index == ts) ) == 1:
                    cv_fold_1_indicesCommonA |= ( train_model.index == ts)
                if np.sum( ( train_model.index == ts) ) == 1:
                    cv_fold_1_indicesCommonB |= ( train_model2.index == ts)

            if (np.sum(cv_fold_1_indicesCommonA) != np.sum(cv_fold_1_indicesCommonB)):
                print("Internal error: cv_fold_1 common indices mismatch",np.sum(cv_fold_1_indicesCommonA), np.sum(cv_fold_1_indicesCommonB) )

        
            cv_fold_2_indicesCommonA = np.zeros(len(train_model), dtype=bool)
            cv_fold_2_indicesCommonB = np.zeros(len(train_model2), dtype=bool)
        
            cv_fold_2 = np.zeros(len(df_model_train), dtype=bool)
            for ts in cv_list[2]:
                cv_fold_2 |= ( df_model_train.index == ts)
                if np.sum( ( train_model2.index == ts) ) == 1:
                    cv_fold_2_indicesCommonA |= ( train_model.index == ts)
                if np.sum( ( train_model.index == ts) ) == 1:
                    cv_fold_2_indicesCommonB |= ( train_model2.index == ts)

            if (np.sum(cv_fold_2_indicesCommonA) != np.sum(cv_fold_2_indicesCommonB)):
                print("Internal error: cv_fold_2 common indices mismatch",np.sum(cv_fold_2_indicesCommonA), np.sum(cv_fold_2_indicesCommonB) )
        
            cv_fold_3_indicesCommonA = np.zeros(len(train_model), dtype=bool)
            cv_fold_3_indicesCommonB = np.zeros(len(train_model2), dtype=bool)
        
            cv_fold_3 = np.zeros(len(df_model_train), dtype=bool)
            for ts in cv_list[3]:
                cv_fold_3 |= ( df_model_train.index == ts)
                if np.sum( ( train_model2.index == ts) ) == 1:
                    cv_fold_3_indicesCommonA |= ( train_model.index == ts)
                if np.sum( ( train_model.index == ts) ) == 1:
                    cv_fold_3_indicesCommonB |= ( train_model2.index == ts)

            if (np.sum(cv_fold_3_indicesCommonA) != np.sum(cv_fold_3_indicesCommonB)):
                print("Internal error: cv_fold_3 common indices mismatch",np.sum(cv_fold_3_indicesCommonA), np.sum(cv_fold_3_indicesCommonB) )
                    
            custom_cv2 = [ (cv_fold_1,  cv_fold_2 | cv_fold_3 | valid_indicesB ), (cv_fold_2, cv_fold_1 | cv_fold_3 | valid_indicesB ), (cv_fold_3, cv_fold_1 | cv_fold_2 | valid_indicesB), (valid_indicesB , cv_fold_1 | cv_fold_2 | cv_fold_3) ]
            
            common_indices = [(cv_fold_1_indicesCommonA, cv_fold_1_indicesCommonB), (cv_fold_2_indicesCommonA, cv_fold_2_indicesCommonB), (cv_fold_3_indicesCommonA, cv_fold_3_indicesCommonB), (valid_indicesCommonA, valid_indicesCommonB) ]


            if set_name2 in regressors_dict:
                (estimator2_list, custom_cv, columns_selection2) = regressors_dict[set_name2]
            else:    
                regressors_file = "fognet_%s_regressors.gz" % set_name2
                if os.path.isfile(regressors_file):
                    print("Loading saved regressors from %s"  % regressors_file)
                    f = gzip.open( regressors_file,"r")
                    (estimator2_list, custom_cv, columns_selection2) = pickle.load(f)
                    f.close()
                    regressors_dict[set_name2] = (estimator2_list, custom_cv, columns_selection2)
                else:    
                    columns_selection2 = list(work_model2.columns)
                    columns_selection2.remove('yield')
                    columns_selection2.remove('is_train')                                    
                    columns_selection2_nocorr = remove_correlated_features(work_model2, columns_selection2)
                    columns_selection2 = columns_selection2_nocorr
                    estimator2_list = []
             
            X2 = work_model2.loc[work_model2['is_train'] == True, columns_selection2 ].as_matrix().astype(np.float32)
            X_pred2 = work_model2.loc[work_model2['is_train'] == False, columns_selection2 ].as_matrix().astype(np.float32)
            y2 = work_model2.loc[work_model2['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
            

            if len(estimator2_list) == 0:
                regressor_base = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )
            
            
            y_pred_est1 = []
            valid_pred_est1 = []
            
            y_pred_est2 = []
            valid_pred_est2 = []

                    
            for fold_idx in xrange(4):
                print("Fold:", fold_idx + 1, "/4")
    
                if np.sum(common_indices[fold_idx][0]) == 0:
                    print("Empty intersection. Skip")
                    continue
                
                valid_indices = custom_cv2[fold_idx][0]
                train_indices = custom_cv2[fold_idx][1]                

                if len(estimator2_list) == 4:
                    best_estimator = estimator2_list[fold_idx]
                else:                
                    best_params2.pop('best_iteration', None)
                    best_estimator = clone(regressor_base).set_params(**best_params2)
        
                    train_set_x_tot = X2[train_indices].copy()
                    train_set_y_tot = y2[train_indices].copy()
                    valid_set_x = X2[valid_indices].copy()
                    valid_set_y = y2[valid_indices].copy()
        
                    eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]
                
                    best_estimator.fit(train_set_x_tot, train_set_y_tot, eval_metric = 'rmse', eval_set = eval_set, early_stopping_rounds=50, verbose = False)
                    
                    estimator2_list.append(best_estimator)

                best_estimator1 = estimator1_list[fold_idx]
                
                y_pred1 = best_estimator1.predict(X1[common_indices[fold_idx][0]])
                y_pred1[y_pred1 < 0.0] = 0.0
                y_valid1 = y1[common_indices[fold_idx][0]]
                rmse1 = compute_rmse(y_pred1, y_valid1)
                
                y_pred_est1.append(y_pred1)
                valid_pred_est1.append(y_valid1)
                                
                y_pred2 = best_estimator.predict(X2[common_indices[fold_idx][1]])
                y_pred2[y_pred2 < 0.0] = 0.0
                y_valid2 = y2[common_indices[fold_idx][1]]
                rmse2 = compute_rmse(y_pred2, y_valid2)
                
                y_pred_est2.append(y_pred2)
                valid_pred_est2.append(y_valid2)
                
                print("Rmses on common indices:", rmse1, rmse2)


            if not(os.path.isfile(regressors_file)):
                print("Saving regressors to %s" % regressors_file)
                f = gzip.open( regressors_file,"wb")
                cPickle.dump((estimator2_list, custom_cv2, columns_selection2), f, cPickle.HIGHEST_PROTOCOL)
                f.close()
                if not set_name2 in regressors_dict:
                    regressors_dict[set_name2] = (estimator2_list, custom_cv2, columns_selection2)

                
            y_pred1 = np.concatenate(y_pred_est1)
            y_valid1 = np.concatenate(valid_pred_est1)
            rmse1b = compute_rmse(y_pred1, y_valid1)
            
            y_pred2 = np.concatenate(y_pred_est2)
            y_valid2 = np.concatenate(valid_pred_est2)
            rmse2b = compute_rmse(y_pred2, y_valid2)
            
            
            y_hat = sub_form_common[yield1_name]
            rmse1 = compute_rmse(sub_form_common['yield'], y_hat)
            #print("%s x %s: rmse %s =" % (set_name, set_name2, set_name), rmse1)
            

            y_hat = sub_form_common[yield2_name]
            rmse2 = compute_rmse(sub_form_common['yield'], y_hat)
            print("%s x %s => " % (set_name, set_name2), rmse1/rmse2, rmse1, rmse2, rmse1b, rmse2b, rmse1b/rmse2b)
            
            comparison_list.append((set_name, set_name2, rmse1, rmse2, sub_form_common.shape[0], rmse1b, rmse2b, rmse1b / rmse2b))
            
            idx_set_2 += 1
        idx_set_1 += 1
    return comparison_list

def eval_folded(evaluation_list, cv_list, set_name_param):
    from sklearn import clone

    
    comparison_list = []
    for evaluation_set in evaluation_list:
        (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model1, pred_test1, best_params1) = evaluation_set
        
        if set_name != set_name_param:
            continue

        df_model_train = train_model.copy()
        
        valid_indicesA = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[0]:
            valid_indicesA |= ( df_model_train.index == ts)
    
        cv_fold_1 = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[1]:
            cv_fold_1 |= ( df_model_train.index == ts)
    
        cv_fold_2 = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[2]:
            cv_fold_2 |= ( df_model_train.index == ts)
    
        cv_fold_3 = np.zeros(len(df_model_train), dtype=bool)
        for ts in cv_list[3]:
            cv_fold_3 |= ( df_model_train.index == ts)
        
        custom_cv1 = [ (cv_fold_1,  cv_fold_2 | cv_fold_3 | valid_indicesA), (cv_fold_2, cv_fold_1 | cv_fold_3 | valid_indicesA ), (cv_fold_3, cv_fold_1 | cv_fold_2 | valid_indicesA ), (valid_indicesA,  cv_fold_2 | cv_fold_3 |  cv_fold_1 ) ]


        if set_name in regressors_dict:
            (estimator1_list, custom_cv, columns_selection1) = regressors_dict[set_name]
        else:
            regressors_file = "fognet_%s_regressors.gz" % set_name
            if os.path.isfile(regressors_file):
                print("Loading saved regressors from %s" % regressors_file)
                f = gzip.open( regressors_file,"r")
                (estimator1_list, custom_cv, columns_selection1) = pickle.load(f)
                f.close()
                regressors_dict[set_name] = (estimator1_list, custom_cv, columns_selection1) 
            else:
                columns_selection1 = list(work_model1.columns)
                columns_selection1.remove('yield')
                columns_selection1.remove('is_train')                                    
                columns_selection1_nocorr = remove_correlated_features(work_model1, columns_selection1)
                columns_selection1 = columns_selection1_nocorr
                estimator1_list = []
             
        X1 = work_model1.loc[work_model1['is_train'] == True, columns_selection1 ].as_matrix().astype(np.float32)
        X_pred1 = work_model1.loc[work_model1['is_train'] == False, columns_selection1 ].as_matrix().astype(np.float32)
        y1 = work_model1.loc[work_model1['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
        

        regressor_base = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )

        
        if len(estimator1_list) == 0:
                    
            for fold_idx in xrange(4):
                print("Fold:", fold_idx + 1, "/4")
    
                
                valid_indices = custom_cv1[fold_idx][0]
                train_indices = custom_cv1[fold_idx][1]
    
                best_params1.pop('best_iteration', None)
                best_estimator = clone(regressor_base).set_params(**best_params1)
    
                train_set_x_tot = X1[train_indices].copy()
                train_set_y_tot = y1[train_indices].copy()
                valid_set_x = X1[valid_indices].copy()
                valid_set_y = y1[valid_indices].copy()
    
                eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]
            
                best_estimator.fit(train_set_x_tot, train_set_y_tot, eval_metric = 'rmse', eval_set = eval_set, early_stopping_rounds=50, verbose = False)
                
                estimator1_list.append(best_estimator)
    
            f = gzip.open( regressors_file,"wb")
            cPickle.dump((estimator1_list, custom_cv1, columns_selection1), f, cPickle.HIGHEST_PROTOCOL)
            f.close()
            regressors_dict[set_name] = (estimator1_list, custom_cv1, columns_selection1)

        y_pred = np.zeros(X_pred1.shape[0], dtype=np.float32)
        
        for estimator in estimator1_list:
            y_pred1 = estimator.predict(X_pred1)
            y_pred1[y_pred1 < 0.0] = 0.0
            y_pred = y_pred + y_pred1
        
        y_pred *= 1.0/len(estimator1_list)
        
        return y_pred

# process_comparisons returns the sorted list of models
# according to their rmse
def process_comparisons(comparison_list):
    df_comp = pd.DataFrame(columns=('rmse1div2', 'set1','set2', 'c_size','rmse1','rmse2'))
    i=0
    for ( set_name, set_name2, rmse1, rmse2, c_size) in comparison_list:
        df_comp.loc[i] = [rmse1 * 1.0 / rmse2, set_name, set_name2, c_size, rmse1, rmse2]
        i = i+1
        df_comp.loc[i] = [rmse2 * 1.0 / rmse1, set_name2, set_name, c_size, rmse2, rmse1]
        i = i+1
        
    def set_compare(x,y):
        return int(df_comp[(df_comp['set1'] == y )  & (df_comp['set2'] ==x)]['rmse1div2'].as_matrix()[0]*10) - 10 
    return sorted(df_comp['set1'].unique(), cmp=set_compare)

def process_comparisons2(comparison_list):
    df_comp = pd.DataFrame(columns=('rmse1div2', 'set1','set2', 'c_size','rmse1','rmse2'))
    i=0
    for ( set_name, set_name2, rmse1, rmse2, c_size, rmse1b, rmse2b, rmse12b ) in comparison_list:
        df_comp.loc[i] = [rmse1b * 1.0 / rmse2b, set_name, set_name2, c_size, rmse1b, rmse2b]
        i = i+1
        df_comp.loc[i] = [rmse2b * 1.0 / rmse1b, set_name2, set_name, c_size, rmse2b, rmse1b]
        i = i+1
        
    def set_compare(x,y):
        return int(df_comp[(df_comp['set1'] == y )  & (df_comp['set2'] ==x)]['rmse1div2'].as_matrix()[0]*10) - 10 
    return sorted(df_comp['set1'].unique(), cmp=set_compare)


def generate_valid_sub(evaluation_list, sub_form, cv_list):
    comparison_list = compare_evaluations(evaluation_list, sub_form)
    if 1:
        sub_form = sub_form.copy()
        sub_form_valid = sub_form[sub_form['is_train'] == False].copy()
        sub_form = sub_form[sub_form['is_train'] == True].copy()
        #sub_form = sub_form[ (sub_form.index.day <= 4) & (sub_form.index.day >= 1)]
        valid_indices = np.zeros(len(sub_form), dtype=bool)
        for ts in cv_list[0]:
            valid_indices |= ( sub_form.index == ts)
        sub_form = sub_form[ valid_indices ]

        print("Sub_form shape:", sub_form.shape)
    
    alg_ranking = process_comparisons(comparison_list)
    sub_form['yield_predicted'] = np.nan
    sub_form['yield_source'] = ""
    for best_set_name in alg_ranking:
        print("Before Set:", best_set_name, sub_form['yield_predicted'].isnull().sum(axis=0), "null predictions")
        for evaluation_set in evaluation_list:
            (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model, pred_test, best_params) = evaluation_set
            if best_set_name == set_name:      
                pred_valid[pred_valid < 0] = 0.0
                print("valid_indices shape:", valid_indices.shape)   
                print("valid_indices true:", np.sum(valid_indices))
                print("pred_valid shape:", pred_valid.shape)
                print("sub_form shape:", sub_form.shape)
                train_model[ 'yield_%s' % set_name ] = np.nan
                train_model.loc[valid_indices, 'yield_%s' % set_name ] = pred_valid
                train_model.loc[np.logical_not(valid_indices), 'yield_%s' % set_name ] = np.nan
                
                sub_form = sub_form.join(train_model.loc[valid_indices, 'yield_%s' % set_name], how='left' )
                #sub_form.loc[sub_form['yield_predicted'].isnull() & valid_indices, 'yield_predicted'] = pred_valid
                sub_form.loc[np.logical_not(sub_form['yield_%s' % set_name].isnull()), 'yield_predicted' ] = pred_valid #sub_form.loc[np.logical_not(sub_form['yield_%s' % set_name].isnull()),'yield_%s' % set_name]                
                sub_form.loc[np.logical_not(sub_form['yield_%s' % set_name].isnull()), 'yield_source' ] = set_name
                #print("Before Set:", best_set_name, sub_form['yield_predicted'].isnull().sum(axis=0), "null predictions")
                y_hat = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield_predicted'].as_matrix()
                y = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield'].as_matrix()
            
                rmse = np.sqrt(mean_squared_error(y, y_hat))
                print("rmse=",rmse, "on", sub_form['yield_predicted'].isnull().sum(axis=0) ,"null values")

                pred_test[pred_test < 0] = 0.0                
                work_model.loc[work_model['is_train'] == False, 'yield_%s' % set_name ] = pred_test
                sub_form_valid = sub_form_valid.join(work_model.loc[work_model['is_train'] == False, 'yield_%s' % set_name ], how='left')
                sub_form_valid.loc[np.logical_not(sub_form_valid['yield_%s' % set_name].isnull()), 'yield' ] = sub_form_valid['yield_%s' % set_name]
                sub_form_valid.loc[np.logical_not(sub_form_valid['yield_%s' % set_name].isnull()), 'yield_source' ] = set_name 
                
                break
    print("End loop:",  sub_form['yield_predicted'].isnull().sum(axis=0) )
    y_hat = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield_predicted'].as_matrix()
    y = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield'].as_matrix()

    rmse = np.sqrt(mean_squared_error(y, y_hat))
    print("rmse=",rmse)

    print("Missing predicts for submit:", np.sum(sub_form_valid['yield'].isnull()))
    #sub_form_valid.loc[sub_form['yield'].isnull(), 'yield'] = sub_form_valid['yield_lin']
    
    sub_form_valid[['yield']].to_csv('fognet_sub_%f.csv' % rmse)
    sub_form_valid[['yield','yield_source']].to_csv('fognet_sub_source_%f.csv' % rmse)
    

def generate_valid_sub2(evaluation_list, sub_form, cv_list, comparison_list):
    #comparison_list = compare_evaluations(evaluation_list, sub_form)
    if 1:
        sub_form = sub_form.copy()
        sub_form_valid = sub_form[sub_form['is_train'] == False].copy()
        sub_form = sub_form[sub_form['is_train'] == True].copy()
        #sub_form = sub_form[ (sub_form.index.day <= 4) & (sub_form.index.day >= 1)]
        valid_indices = np.zeros(len(sub_form), dtype=bool)
        for ts in cv_list[0]:
            valid_indices |= ( sub_form.index == ts)
        sub_form = sub_form[ valid_indices ]

        print("Sub_form shape:", sub_form.shape)
    
    alg_ranking = process_comparisons2(comparison_list)
    sub_form['yield_predicted'] = np.nan
    sub_form['yield_source'] = ""
    for best_set_name in alg_ranking:
        print("Before Set:", best_set_name, sub_form['yield_predicted'].isnull().sum(axis=0), "null predictions")
        for evaluation_set in evaluation_list:
            (pred_valid, valid_set_y, train_model, valid_indices, set_name, work_model, pred_test, best_params) = evaluation_set                        
            if best_set_name == set_name: 
                pred_test_folded = eval_folded(evaluation_list, cv_list, set_name)     
                pred_valid[pred_valid < 0] = 0.0
                print("valid_indices shape:", valid_indices.shape)   
                print("valid_indices true:", np.sum(valid_indices))
                print("pred_valid shape:", pred_valid.shape)
                print("sub_form shape:", sub_form.shape)
                train_model[ 'yield_%s' % set_name ] = np.nan
                train_model.loc[valid_indices, 'yield_%s' % set_name ] = pred_valid
                train_model.loc[np.logical_not(valid_indices), 'yield_%s' % set_name ] = np.nan
                
                sub_form = sub_form.join(train_model.loc[valid_indices, 'yield_%s' % set_name], how='left' )
                #sub_form.loc[sub_form['yield_predicted'].isnull() & valid_indices, 'yield_predicted'] = pred_valid
                sub_form.loc[np.logical_not(sub_form['yield_%s' % set_name].isnull()), 'yield_predicted' ] = pred_valid #sub_form.loc[np.logical_not(sub_form['yield_%s' % set_name].isnull()),'yield_%s' % set_name]                
                #print("Before Set:", best_set_name, sub_form['yield_predicted'].isnull().sum(axis=0), "null predictions")
                y_hat = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield_predicted'].as_matrix()
                y = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield'].as_matrix()
            
                rmse = np.sqrt(mean_squared_error(y, y_hat))
                print("rmse=",rmse, "on", sub_form['yield_predicted'].isnull().sum(axis=0) ,"null values")

                pred_test[pred_test < 0] = 0.0   
                pred_test_folded[pred_test_folded < 0] = 0.0             
                work_model.loc[work_model['is_train'] == False, 'yield_%s' % set_name ] = pred_test_folded
                sub_form_valid = sub_form_valid.join(work_model.loc[work_model['is_train'] == False, 'yield_%s' % set_name ], how='left')
                sub_form_valid.loc[np.logical_not(sub_form_valid['yield_%s' % set_name].isnull()), 'yield' ] = sub_form_valid['yield_%s' % set_name]
                sub_form_valid.loc[np.logical_not(sub_form_valid['yield_%s' % set_name].isnull()), 'yield_source' ] = set_name
                
                break
    print("End loop:",  sub_form['yield_predicted'].isnull().sum(axis=0) )
    y_hat = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield_predicted'].as_matrix()
    y = sub_form.loc[np.logical_not(sub_form['yield_predicted'].isnull()),'yield'].as_matrix()

    rmse = np.sqrt(mean_squared_error(y, y_hat))
    print("rmse=",rmse)

    print("Missing predicts for submit:", np.sum(sub_form_valid['yield'].isnull()))
    #sub_form_valid.loc[sub_form['yield'].isnull(), 'yield'] = sub_form_valid['yield_lin']
    
    sub_form_valid[['yield']].to_csv('fognet_sub2_%f.csv' % rmse)
    sub_form_valid[['yield','yield_source']].to_csv('fognet_sub2_source_%f.csv' % rmse)
    

# compute 4 cv_list
# we take 4 days of sequential data for thefirst cv,
# then 4 days for the second cv , etc     
def compute_cv_ranges(lin_model):
    cv_1 = []
    cv_2 = []
    cv_3 = []
    cv_4 = []
    is_contiguous = True
    current_cv = 0
    current_cv_fill = 0
    cv_list = [[] , [], [], [] ]
    for row in lin_model[['is_train']].sort_index().itertuples():
        
        if row[1] == False:
            is_contiguous = False
            continue
        if is_contiguous:
            if current_cv_fill == 4:
                current_cv_fill = 0
                current_cv += 1
                current_cv = current_cv %4            
        else:
            current_cv += 1
            current_cv = current_cv %4
            current_cv_fill = 0
                
        cv_list[current_cv].append(row[0])
        if row[0].hour == 22:
            current_cv_fill += 1        
        is_contiguous = True
            
    return cv_list
        
        
# the following methods construct a model
#     
def process_model(df_model, columns_selection, regressor, cv_list, param_dist, need_scale = False, need_eval_set = False, fit_params=None):
    from sklearn import clone
    
    X = df_model.loc[df_model['is_train'] == True, columns_selection ].as_matrix().astype(np.float32)
    X_pred = df_model.loc[df_model['is_train'] == False, columns_selection ].as_matrix().astype(np.float32)
    y = df_model.loc[df_model['is_train'] == True, "yield" ].as_matrix().astype(np.float32)
    
    df_model_train = df_model.loc[df_model['is_train'] == True].copy()
    
    valid_indices = np.zeros(len(df_model_train), dtype=bool)
    for ts in cv_list[0]:
        valid_indices |= ( df_model_train.index == ts)

    cv_fold_1 = np.zeros(len(df_model_train), dtype=bool)
    for ts in cv_list[1]:
        cv_fold_1 |= ( df_model_train.index == ts)

    cv_fold_2 = np.zeros(len(df_model_train), dtype=bool)
    for ts in cv_list[2]:
        cv_fold_2 |= ( df_model_train.index == ts)

    cv_fold_3 = np.zeros(len(df_model_train), dtype=bool)
    for ts in cv_list[3]:
        cv_fold_3 |= ( df_model_train.index == ts)
    
    custom_cv = [ (cv_fold_1,  cv_fold_2 | cv_fold_3), (cv_fold_2, cv_fold_1 | cv_fold_3), (cv_fold_3, cv_fold_1 | cv_fold_2) ]
    
    train_set_x_tot = X[np.logical_not(valid_indices)].copy()
    train_set_y_tot = y[np.logical_not(valid_indices)].copy()
    valid_set_x = X[valid_indices].copy()
    valid_set_y = y[valid_indices].copy()
    
    eval_set=[(train_set_x_tot,train_set_y_tot),(valid_set_x, valid_set_y)]

    if need_scale:
        scaler = StandardScaler()
        scaler.fit(X)
        
        train_set_x_tot = scaler.transform(train_set_x_tot)
        valid_set_x = scaler.transform(valid_set_x)
        X_pred = scaler.transform(X_pred)

    n_iter_search = 10

    random_search =GridSearchCV(regressor, param_grid=param_dist,n_jobs=1,verbose=1,fit_params={'eval_metric':'rmse','eval_set':eval_set,'early_stopping_rounds':50, 'verbose':False},cv=custom_cv, scoring = rmse_scoring, refit = False )
    random_search.fit(X, y)
    
    print("best params:", random_search.best_params_)
    
    regressor = clone(regressor).set_params(**random_search.best_params_)
    
    if need_eval_set:
        regressor.fit(train_set_x_tot, train_set_y_tot , eval_set = eval_set, verbose=False, **fit_params)
        random_search.best_params_['best_iteration'] = regressor.best_iteration
    else:
        regressor.fit(train_set_x_tot, train_set_y_tot, verbose=False ,  **fit_params)
    
    pred_model_valid = regressor.predict(valid_set_x)
    
    
    y_pred = regressor.predict(X_pred)
    
    return (pred_model_valid, valid_set_y, df_model_train, valid_indices, y_pred, random_search.best_params_) 
    

# adapted from an example on stackoverflow
# this version is actually buggy - it becomes extremely slow with hundreds of features
# the"break" has to be removed, instead we have to take care to not remove twice the same column
def remove_correlated_features(df_model, columns_selection, corr_threshold = 0.99, verbose=True):
    from itertools import chain
    
    columns_selection_nocorr = list(columns_selection)
    
    found_correlations = True
    while(found_correlations):
        found_correlations = False
        cor=df_model[columns_selection_nocorr ].corr()
        cor.loc[:,:] = np.tril(cor, k=-1)
        cor = cor.stack()
        ones = cor[cor > corr_threshold].reset_index().loc[:,['level_0','level_1']]
        ones = ones.query('level_0 not in level_1')
        
        
        for corr_set in ones.groupby('level_0').agg(lambda x: set(chain(x.level_0, x.level_1))).values :
            corr_set = corr_set[0]
            first_elem = True
            for col in corr_set:
                found_correlations = True
                if not(first_elem):
                    if verbose:
                        print("Removing", col)
                    columns_selection_nocorr.remove(col)
                first_elem = False
            break
    return columns_selection_nocorr

def handle_nulls(work_model):
    for c in work_model.columns:
        if c == 'yield':
            continue
        if c == 'is_train':
            continue
        if work_model[c].isnull().any():
            work_model["%s_nulls" % c] = work_model[c].isnull()
            if (np.min(work_model[c]) > 0):
               fillna = 0
            else:
                fillna = int(np.min(work_model[c])) - 1 
            work_model[c].fillna(fillna, inplace = True)
    return work_model

def add_eval_sets(work_model, set_list , set_name, evaluations_list, cv_list, do_dropna = True):
    work_model = work_model.copy()
    print("Adding", set_name)
    for  (df_set, hours_grouped, def_name) in set_list:
        group_list = GenerateGroupsBy(df_set, def_name, hours_grouped = hours_grouped)
        for grouped_result in group_list:
            work_model = work_model.join(grouped_result, how='left')
        #print(work_model.isnull().sum(axis=0))
        #print(work_model.count(axis=0))
    if do_dropna:
        work_model.dropna(inplace=True)
    else:
        for c in work_model.columns:
            if c == 'yield':
                continue
            if c == 'is_train':
                continue
            if work_model[c].isnull().any():
                work_model["%s_nulls" % c] = work_model[c].isnull()
                if (np.min(work_model[c]) > 0):
                    fillna = 0
                else:
                    fillna = int(np.min(work_model[c])) - 1                          
                work_model[c].fillna(fillna, inplace = True) 
            
    
    columns_selection = list(work_model.columns)
    columns_selection.remove('yield')
    columns_selection.remove('is_train')
            

    columns_selection_nocorr = remove_correlated_features(work_model, columns_selection)

    regressor = XGBRegressor(max_depth=3, silent=True, learning_rate= 0.005, n_estimators=10000,objective='reg:linear', subsample=0.65, colsample_bytree=0.8, seed=0, reg_lambda=0.97 , reg_alpha=0.2, gamma=1.0, missing = np.NaN )

    param_dist = {"max_depth": [3,4,5,6,7,8],
                  "subsample" : [0.7,0.8,0.9],
                  "colsample_bytree" : [0.7,0.8,0.9],
                  "n_estimators" : [10000]
                }                  
    
    (pred_model_work_valid, valid_set_work_y, work_model_train, work_valid_indices, work_pred, best_params) = process_model(work_model, columns_selection, regressor, cv_list, param_dist, need_eval_set = True, fit_params={ 'eval_metric':'rmse','early_stopping_rounds':50})
    print("rmse=", compute_rmse(valid_set_work_y, pred_model_work_valid))
    
    work_model_train.loc[work_valid_indices,'yield_%s_predicted' % set_name ] = pred_model_work_valid
    evaluations_list.append((pred_model_work_valid, valid_set_work_y, work_model_train, work_valid_indices, set_name, work_model, work_pred, best_params))
    
    
def doGridSearch(eval_set,X,y):
    
            regressor = XGBRegressor(max_depth=4, silent=True, learning_rate= 0.05, n_estimators=10,objective='reg:linear', subsample=0.9, colsample_bytree=0.9, seed=0, missing = np.NaN )
            # Utility function to report best scores
            param_dist = {"max_depth": [3,5,7,9,11,13],
                          "subsample" : sp_uniform(0.7,0.3),
                          "colsample_bytree" : sp_uniform(0.7,0.3),
                          "gamma" : sp_uniform(0.0,0.2),
                          "reg_lambda" : sp_uniform(0.5,1.0),
                          "reg_alpha" : sp_uniform(0.0,0.5),
                          #"learning_rate" : [0.01,0.001,0.0001,0.00001],
                          "learning_rate" : [0.01],
                          "n_estimators" : [10000]
                        }                  
            n_iter_search = 50

            #eval_set=[(valid_set_x, valid_set_y)]
            random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse','eval_set':eval_set,'early_stopping_rounds':50},cv=2, scoring = rmse_scoring)
            #random_search = RandomizedSearchCV(regressor, param_distributions=param_dist,n_jobs=1,n_iter=n_iter_search,verbose=3,fit_params={'eval_metric':'rmse'},cv=5)
            
            random_search.fit(X, y)
            #random_search.fit(train_set_x_tot, train_set_y_tot) 
            report(random_search.grid_scores_, n_top = 10)

