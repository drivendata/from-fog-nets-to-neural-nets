#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator
import matplotlib
#matplotlib.use("Agg") #Needed to save figures
import matplotlib.pyplot as plt
import seaborn as sns
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

import sys
from psycopg2._psycopg import Column
from numexpr.necompiler import evaluate
np.random.seed(0)

from fognet_utils import toBinary, rmse_scoring, report, WindParse, WindParse2, CloudCoverParse, CloudDensityParse, WeatherParse, TruncateTimeStamp2Hours, TruncateTimeStamp4Hours, TruncateTimeStampHours
  
from fognet_utils import GenerateGroupsBy, compute_rmse, plot_errors,  compare_evaluations, CloudHeigthParse, process_model, remove_correlated_features
from fognet_utils import add_eval_sets
import fognet_utils
    
## Start of main script

if 1:
    
    if 1:
        print("Loading sidi sdata")
        
        sidi = pd.read_csv("fognet_sidi.csv",index_col = 0, parse_dates=[0])
        
        
        
        sidi.loc[sidi['WW'] == 'Fog or ice fog, sky visible, has begun or has become thicker during the preceding hour.', 'RainLevel'] = 'Fog'
        sidi.loc[sidi['WW'] == 'Mist', 'RainLevel'] = 'Mist'
        sidi.loc[sidi['W1'] == 'Drizzle', 'RainLevel'] = 'Drizzle'
        sidi.loc[sidi['W1'] == 'Thunderstorm(s) with or without precipitation.', 'RainLevel'] = 'Thunder'
        sidi.loc[sidi['W1'] == 'Rain', 'RainLevel'] = 'Rain'
        sidi.loc[sidi['W1'] == 'Fog or ice fog or thick haze.', 'RainLevel'] = 'Fog'        
        sidi.loc[sidi['W1'] == 'Shower(s).', 'RainLevel' ] = 'Rain'
        sidi.loc[sidi['W2'] == 'Fog or ice fog or thick haze.', 'RainLevel' ] = 'Fog'
        sidi.loc[sidi['W2'] == 'Shower(s).', 'RainLevel' ] = 'Rain'
        sidi.loc[sidi['W2'] == 'Drizzle', 'RainLevel'] = 'Drizzle'
        sidi.loc[sidi['W2'] == 'Rain', 'RainLevel'] = 'Rain'
        sidi.loc[sidi['RainLevel'].isnull(), 'RainLevel'] = 'Dry'
        
        
        
        toBinary('RainLevel', sidi)
        sidi.drop(['RainLevel'], axis=1,inplace=True)
        
        
        
        # drop garbage columns
        sidi.drop(["WW","W1","W2","E'", "sss","E"],axis=1,inplace = True)
        
        # These one should not be useful
        sidi.drop(['Tn','Tx'],axis=1,inplace = True)
        
        # todo
        #sidi.drop(['Tg'],axis=1,inplace = True)
        
        # Fix cloud cover..
        
        sidi.loc[sidi['N'] == 'no clouds', 'Cl'] = 'no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'Nh'] = 'no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'H'] = '2500 or more, or no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'Cm'] = 'no clouds'
        sidi.loc[sidi['N'] == 'no clouds', 'Ch'] = 'no clouds'
        
        sidi.loc[sidi['Cl'] == 'No Stratocumulus, Stratus, Cumulus or Cumulonimbus.', 'Cl' ] = 'no clouds'
        sidi.loc[sidi['Cm'] == 'No Altocumulus, Altostratus or Nimbostratus.', 'Cm' ] = 'no clouds'
        sidi.loc[sidi['Ch'] == 'No Cirrus, Cirrocumulus or Cirrostratus.', 'Ch' ] = 'no clouds'
        
        
        sidi.loc[(sidi['H'] == '2500 or more, or no clouds.') & sidi['Cl'].isnull(), 'Cl' ] = 'no clouds'
        sidi.loc[(sidi['H'] == '2500 or more, or no clouds.') & sidi['Cm'].isnull(), 'Cm' ] = 'no clouds' 
        sidi.loc[(sidi['H'] == '2500 or more, or no clouds.') & sidi['Ch'].isnull(), 'Ch' ] = 'no clouds'
        
        # We don't see them.. They are not there..
        sidi.loc[(sidi['Nh'] == '100%') & sidi['Ch'].isnull() , 'Ch' ] = 'no visible clouds'
        sidi.loc[(sidi['Nh'] == '90  or more, but not 100%') & sidi['Ch'].isnull(), 'Ch' ] = 'no visible clouds'        
        sidi.loc[(sidi['N'] == '100%') & sidi['Ch'].isnull(), 'Ch' ] = 'no visible clouds'
        sidi.loc[(sidi['N'] == '90  or more, but not 100%') & sidi['Ch'].isnull(), 'Ch' ] = 'no visible clouds'
        
        sidi.loc[(sidi['Nh'] == '100%') & sidi['Cm'].isnull() , 'Cm' ] = 'no visible clouds'
        sidi.loc[(sidi['Nh'] == '90  or more, but not 100%') & sidi['Cm'].isnull(), 'Cm' ] = 'no visible clouds'        
        sidi.loc[(sidi['N'] == '100%') & sidi['Cm'].isnull(), 'Cm' ] = 'no visible clouds'
        
        
        # Fix lazy public servant's work
        sidi.loc[(sidi['N'] == 'no clouds') & sidi['Nh'].isnull(), 'Nh'] = 'no clouds'
        sidi.loc[(sidi['N'] == 'no clouds') & sidi['H'].isnull(), 'H'] = '2500 or more, or no clouds.'
        
        
        
        sidi.loc[sidi['Cl'] == 'Cumulus mediocris or congestus, with or without Cumulus of species fractus or humilis or Stratocumulus, all having their bases at the same level.', 'Cl' ] = 'mediocris'                       
        sidi.loc[sidi['Cl'] == 'Stratocumulus other than Stratocumulus cumulogenitus.', 'Cl'] = 'Stratocumulus'
        sidi.loc[sidi['Cl'] == 'Stratus fractus or Cumulus fractus of bad weather, or both (pannus), usually below Altostratus or Nimbostratus.', 'Cl' ] = 'fractus'
        sidi.loc[sidi['Cl'] == 'Cumulonimbus calvus, with or without Cumulus, Stratocumulus or Stratus.', 'Cl' ] = 'calvus' 
        sidi.loc[sidi['Cl'] == 'Cumulonimbus capillatus (often with an anvil), with or without Cumulonimbus calvus, Cumulus, Stratocumulus, Stratus or pannus.', 'Cl' ] = 'capillatus' 
        sidi.loc[sidi['Cl'] == 'Stratus nebulosus or Stratus fractus other than of bad weather, or both.', 'Cl' ] = 'nebulosus'
        sidi.loc[sidi['Cl'] == 'Cumulus humilis or Cumulus fractus other than of bad weather, or both.', 'Cl' ] = 'humilis'
        sidi.loc[sidi['Cl'] == 'Cumulus and Stratocumulus other than Stratocumulus cumulogenitus, with bases at different levels.', 'Cl' ] = 'cumulus' 
        
        
        sidi.loc[sidi['Cm'] == 'no visible clouds', 'Cm' ] = 'novisible'
        sidi.loc[sidi['Cm'] == 'Altocumulus translucidus at a single level.', 'Cm' ] = 'translucidus'
        sidi.loc[sidi['Cm'] == 'Altocumulus translucidus or opacus in two or more layers, or Altocumulus opacus in a single layer, not progressively invading the sky, or Altocumulus with Altostratus or Nimbostratus.', 'Cm' ] = 'opacus'
        sidi.loc[sidi['Cm'] == 'Altocumulus castellanus or floccus.', 'Cm' ] = 'floccus'
        
        sidi.loc[sidi['Ch'] == 'no visible clouds', 'Ch' ] = 'novisible'
        sidi.loc[sidi['Ch'] == 'Cirrus spissatus, in patches or entangled sheaves, which usually do not increase and sometimes seem to be the remains of the upper part of a Cumulonimbus; or Cirrus castellanus or floccus.', 'Ch' ] = 'spissatus'
        sidi.loc[sidi['Ch'] == 'Cirrus (often in bands) and Cirrostratus, or Cirrostratus alone, progressively invading the sky; they generally thicken as a whole, but the continuous veil does not reach 45 degrees above the horizon.', 'Ch' ] = 'cirrostratus'
        sidi.loc[sidi['Ch'] == 'Cirrus fibratus, sometimes uncinus, not progressively invading the sky.', 'Ch' ] = 'fibratus'
        
        
        # Trace of precipitation is .. small precipitation
        sidi.loc[sidi['RRR'] == "Trace of precipitation", 'RRR' ] = 0.01
        sidi['RRR'] = sidi['RRR'].astype(np.float32)
                                         
        # At  6:00 AM there should be report of rain for the last 24h
        # But maybe there is no data if there was no rain for 24h
        
        # Make sure we have data at 12:00, 18:00, 6:00 with rain for the last 6,12,24 hours
        sidi.loc[sidi['tR'].isnull() & (sidi.index.hour == 6), 'tR' ] = 24
        sidi.loc[sidi['RRR'].isnull() & (sidi.index.hour == 6), 'RRR' ] = 0.0
        
        sidi.loc[sidi['tR'].isnull() & (sidi.index.hour == 18), 'tR' ] = 12
        sidi.loc[sidi['RRR'].isnull() & (sidi.index.hour == 18), 'RRR' ] = 0.0
        
        sidi.loc[sidi['tR'].isnull() & (sidi.index.hour == 12), 'tR' ] = 6
        sidi.loc[sidi['RRR'].isnull() & (sidi.index.hour == 12), 'RRR' ] = 0.0
        
        sidi.loc[sidi['RRR'].isnull(),'RRR'] = 0.0
        
        
        sidi['day'] = sidi.index.values.astype('<M8[D]')
                
        sidi['day_before'] = sidi['day'] - np.timedelta64(1,'D')
        
        # At 6h00 retrieve the volume the previous day at 18h so that
        # We know the qtity of rain during the night
        
        sidi_night_rain = pd.merge(sidi.loc[sidi.index.hour == 6, ['day_before','RRR']].reset_index(), sidi.loc[sidi.index.hour == 18, ['day','RRR']], left_on = 'day_before', right_on = 'day' ).set_index('Local time in Sidi Ifni')
        
        sidi_night_rain['avg_night_rain_3h'] = (sidi_night_rain['RRR_x'] - sidi_night_rain['RRR_y']) / 4.0
        
        # At 18h00 retrieve the volume at 12h so that
        # We know the qtity of rain between 12h and 18h
        
        sidi_18_rain = pd.merge(sidi.loc[sidi.index.hour == 18, ['day','RRR']].reset_index(), sidi.loc[sidi.index.hour == 12, ['day','RRR']], left_on = 'day', right_on = 'day' ).set_index('Local time in Sidi Ifni')
        
        sidi_18_rain['avg_18_rain_3h'] = (sidi_18_rain['RRR_x'] - sidi_18_rain['RRR_y']) / 2.0
        
        sidi = sidi.join(sidi_night_rain['avg_night_rain_3h'], how='left')
        sidi = sidi.join(sidi_18_rain['avg_18_rain_3h'], how='left')
        
        sidi.loc[sidi.index.hour == 6, 'avg_rain_3h'] = sidi.loc[sidi.index.hour == 6, 'avg_night_rain_3h'] 
        sidi.loc[sidi.index.hour == 18, 'avg_rain_3h'] = sidi.loc[sidi.index.hour == 18, 'avg_18_rain_3h']
        sidi.loc[sidi.index.hour == 12, 'avg_rain_3h'] = sidi.loc[sidi.index.hour == 12, 'RRR'] / 2.0
        
        sidi.drop(['avg_night_rain_3h','avg_18_rain_3h','RRR','tR','day','day_before'], axis=1, inplace=True)
        
        col_shift_list = ['T','Po','U','DD','Ff','Cl','Nh','Ch']
        
        print(sidi.isnull().sum(axis=0))
        
        # replace some missing values with the previous one
        for x_offset in xrange(3):
                    sidi = sidi.join(sidi.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
                    
                    
                    for col in col_shift_list:
                        sidi.loc[sidi[col].isnull(), col] = sidi['%s_minus_1' % col]
                    
                    sidi.loc[(sidi["DD"] == 'variable wind direction'), 'DD'] = sidi['%s_minus_1' % 'DD']                    
                    
                    for col in col_shift_list:
                        sidi.drop(['%s_minus_1'%col], axis=1, inplace=True)
        
        
        WindParse2(sidi,"DD")
        CloudCoverParse(sidi,"N")
        CloudCoverParse(sidi,"Nh",'CloudCover2')
        CloudHeigthParse(sidi,'H')

        sidi_wind = sidi[np.logical_not(sidi['Ff'].isnull())][['Ff','WindDirection1','WindDirection2','WindDirection3']].copy()
        
        sidi.drop(['DD','N','Ff','Nh','H','WindDirection1','WindDirection2','WindDirection3'], axis=1, inplace = True)
        
        for col_cat in ['Cl','Cm','Ch']:
            newcols = toBinary(col_cat, sidi)
            sidi.drop([col_cat], axis=1, inplace = True)
        
        for col_cat in ['WindDirection1','WindDirection2','WindDirection3']:
            newcols = toBinary(col_cat, sidi_wind)
            sidi_wind.drop([col_cat], axis=1, inplace = True)
            
        sidi_pa = sidi[np.logical_not(sidi['Pa'].isnull() )][['Pa']].copy()
        
        sidi_Tg = sidi[np.logical_not(sidi['Tg'].isnull())][['Tg']].copy()
        
        sidi_avg_rain = sidi[np.logical_not(sidi['avg_rain_3h'].isnull())][['avg_rain_3h']].copy()
        
        sidi.drop(['Pa','Tg','avg_rain_3h'], axis=1, inplace = True)

        sidi.dropna(inplace=True)
        sidi_wind.dropna(inplace=True)
        
        
        print("Describe sidi",sidi.describe())
        print("Describe sidi_wind",sidi_wind.describe())
        print("Describe sidi_pa",sidi_pa.describe())
        print("Describe sidi_Tg",sidi_Tg.describe())
        print("Describe sidi_avg_rain",sidi_avg_rain.describe())
        
        

    if 1:
        print("Loading guelmin sdata")
        
        guelmin = pd.read_csv("fognet_macro_guelmin.csv",index_col = 0, parse_dates=[0])
        
        
        
        # drop garbage columns
        guelmin.drop(['ff10'], axis=1, inplace = True)
        
        guelmin.loc[guelmin['WW'].isnull(), 'WW'] = guelmin["W'W'"]
        
        WeatherParse(guelmin,"WW")

        guelmin.loc[guelmin["c"] == "29970 m9, few clouds (10-30%) 450 m", "c"] = "Few clouds (10-30%) 450 m"
        guelmin.loc[guelmin["c"] == "29970 m9, few clouds (10-30%) 480 m", "c"] = "Few clouds (10-30%) 480 m"
        guelmin.loc[guelmin["c"] == "29970 m9, few clouds (10-30%) 600 m", "c"] = "Few clouds (10-30%) 600 m"
        guelmin.loc[guelmin["c"] == "Scattered clouds (40-50%)30, cumulonimbus clouds , broken clouds (60-90%) 3000 m","c" ] = "Scattered clouds (40-50%) 30 m, cumulonimbus clouds , broken clouds (60-90%) 3000 m"
        guelmin.loc[guelmin["c"] == "8 less than 30 m , few clouds (10-30%) 480 m , 8 less than 984 feet , few clouds (10-30%) 480 m","c" ] = "Few clouds (10-30%) 480 m"
        
        guelmin.loc[guelmin["c"] == "Scattered clouds (40-50%) 000 000 18420 m/13", "c" ] = np.nan
        guelmin.loc[guelmin["c"] == "Altocumulus clouds T 690 m, cumulus congestus of great vertical extent , few clouds (10-30%) 780 m, cumulonimbus clouds , scattered clouds (40-50%) 3000 m", "c" ] = "Few clouds (10-30%) 780 m"
        
        
        guelmin.drop(["WW","W'W'"],axis=1,inplace = True)
        # append value 1 hour before for DD and ff
        guelmin.loc[(guelmin["DD"] == 'variable wind direction') & (guelmin["Ff"] <= 1) , 'DD'] = 'Calm, no wind'
        
        guelmin.loc[guelmin['VV'] == '10.0 and more', 'VV'] = 10
        
        guelmin['VV'] = guelmin['VV'].astype(np.float32) 
        
        col_shift_list = ['DD','Ff','T','P0','P','VV','Td','U','c','RainLevel']
        
        print(guelmin.isnull().sum(axis=0))
        for x_offset in xrange(3):
                    guelmin = guelmin.join(guelmin.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
                    
                    
                    for col in col_shift_list:
                        guelmin.loc[guelmin[col].isnull(), col] = guelmin['%s_minus_1' % col]
                    
                    guelmin.loc[(guelmin["DD"] == 'variable wind direction'), 'DD'] = guelmin['%s_minus_1' % 'DD']
                    guelmin.loc[(guelmin["c"] == 'Overcast (100%)'), 'c'] = guelmin['%s_minus_1' % 'c']
                    
                    for col in col_shift_list:
                        guelmin.drop(['%s_minus_1'%col], axis=1, inplace=True)
                

        WindParse2(guelmin,"DD")
        
        
        guelmin['cloud_array'] = guelmin.apply(lambda x: str(x['c']).split(",",1)[0].replace("Scattered ","Scattered_").replace("Few ","Few_").replace("Broken ","Broken_").replace("No Significant Clouds","No_Clouds_(0-0%) 10000 m").replace("No clouds","No_Clouds_(0-0%) 19999 m").replace("less than 30","15").replace(" (","_("), axis=1)
        guelmin['cloud_density'] = guelmin.apply(lambda x: x['cloud_array'].split(" ")[0], axis=1)
        
        CloudDensityParse(guelmin,'cloud_density')
        
        guelmin['cloud_distance'] = guelmin.apply(lambda x: x['cloud_array'].split(" ")[1], axis=1)
        
        guelmin['cloud_distance'] = guelmin['cloud_distance'].astype(np.float32)
        
        
        
        guelmin_wind = guelmin[np.logical_not(guelmin['Ff'].isnull())][['Ff','WindDirection1','WindDirection2','WindDirection3']].copy()
        
        for col_cat in ['WindDirection1','WindDirection2','WindDirection3']:
            newcols = toBinary(col_cat, guelmin_wind)
            guelmin_wind.drop([col_cat], axis=1, inplace = True)
            
        guelmin.drop(['DD','Ff','WindDirection1','WindDirection2','WindDirection3','c','cloud_density','cloud_array'], axis=1, inplace = True)        
        guelmin.dropna(inplace=True)
        guelmin_wind.dropna(inplace=True)
        
        
    if 1:
        print("Loading agadir data")
        
        agadir = pd.read_csv("fognet_agadir.csv",index_col = 0, parse_dates=[0])
        
        # drop garbage columns
        agadir.drop(["ff10"],axis=1,inplace = True)

        agadir.loc[agadir['WW'].isnull(), 'WW'] = agadir["W'W'"]        
        WeatherParse(agadir,"WW")

        agadir.drop(['WW',"W'W'"],axis=1,inplace = True)

        agadir.loc[agadir["c"] == "29970 m9, few clouds (10-30%) 480 m", "c"] = "Few clouds (10-30%) 480 m"
        agadir.loc[agadir["c"] == "29970 m9, few clouds (10-30%) 600 m", "c"] = "Few clouds (10-30%) 600 m"
        agadir.loc[agadir["c"] == "Scattered clouds (40-50%)30, cumulonimbus clouds , broken clouds (60-90%) 3000 m","c" ] = "Scattered clouds (40-50%) 30 m, cumulonimbus clouds , broken clouds (60-90%) 3000 m"
        
        agadir.loc[agadir["c"] == "Scattered clouds (40-50%) 000 000 18420 m/13", "c" ] = np.nan


        # fix the crazy "variable wind direction.."        
        agadir.loc[(agadir["DD"] == 'variable wind direction') & (agadir["Ff"] <= 1) , 'DD'] = 'Calm, no wind'
        

        
        col_shift_list = ["T","P0","P","U","DD","Ff","c","VV","Td"]
        
        for x_offset in xrange(17):
            agadir = agadir.join(agadir.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
            
            
            for col in col_shift_list:
                agadir.loc[agadir[col].isnull(), col] = agadir['%s_minus_1' % col]
    
            agadir.loc[(agadir["DD"] == 'variable wind direction'), 'DD'] = agadir['%s_minus_1' % 'DD']
            agadir.loc[(agadir["c"] == 'Overcast (100%)'), 'c'] = agadir['%s_minus_1' % 'c']
            
            for col in col_shift_list:
                agadir.drop(['%s_minus_1'%col], axis=1, inplace=True)


        agadir.loc[agadir['VV'] == '10.0 and more', 'VV'] = 10
        
        agadir['VV'] = agadir['VV'].astype(np.float32) 

        agadir['cloud_array'] = agadir.apply(lambda x: str(x['c']).split(",",1)[0].replace("Scattered ","Scattered_").replace("Few ","Few_").replace("Broken ","Broken_").replace("No Significant Clouds","No_Clouds_(0-0%) 10000 m").replace("No clouds","No_Clouds_(0-0%) 19999 m").replace("less than 30","15").replace(" (","_("), axis=1)
        agadir['cloud_density'] = agadir.apply(lambda x: x['cloud_array'].split(" ")[0], axis=1)
        
        CloudDensityParse(agadir,'cloud_density')
        
        agadir['cloud_distance'] = agadir.apply(lambda x: x['cloud_array'].split(" ")[1], axis=1)
        
        agadir['cloud_distance'] = agadir['cloud_distance'].astype(np.float32)
        

        agadir.drop(['c','cloud_array','cloud_density'],axis=1,inplace = True)

        WindParse2(agadir,"DD")
        
        for col_cat in ['WindDirection1','WindDirection2','WindDirection3']:
            newcols = toBinary(col_cat, agadir)
            agadir.drop([col_cat], axis=1, inplace = True)

        agadir.drop(['DD'], axis=1, inplace = True)

        agadir.dropna(inplace=True)
        
                

    train_micro_5mn = pd.read_csv("fognet_train_micro_5mn.csv", index_col = 0, parse_dates=[0])            
    train_micro_2h = pd.read_csv("fognet_train_micro_2h.csv", index_col = 0, parse_dates=[0])
    test_micro_5mn = pd.read_csv("fognet_test_micro_5mn.csv", index_col = 0, parse_dates=[0])            
    test_micro_2h = pd.read_csv("fognet_test_micro_2h.csv", index_col = 0, parse_dates=[0])
    
    train_target = pd.read_csv("fognet_target.csv", index_col = 0, parse_dates=[0])
    train_target['is_train'] = True
    
    test_submit = pd.read_csv("fognet_submit.csv", index_col = 0, parse_dates=[0])
    test_submit['is_train'] = False
    
    all_micro_5mn = pd.concat([train_micro_5mn, test_micro_5mn]).sort_index()

    
    all_micro_2h = pd.concat([train_micro_2h, test_micro_2h]).sort_index()
    print(all_micro_2h.isnull().sum(axis=0))
    
    
    all_micro_2h_460 = all_micro_2h[np.logical_not(all_micro_2h['leafwet460_min'].isnull())][['leafwet460_min']].copy()
    all_micro_2h.drop(['leafwet460_min'],axis=1,inplace=True)
    
    col_shift_list = ["percip_mm","humidity","temp","leafwet450_min","leafwet_lwscnt","gusts_ms","wind_dir","wind_ms"]
    
    for x_offset in xrange(21):
        all_micro_2h = all_micro_2h.join(all_micro_2h.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
                
        for col in col_shift_list:
            all_micro_2h.loc[all_micro_2h[col].isnull(), col] = all_micro_2h['%s_minus_1' % col]
        
        for col in col_shift_list:
            all_micro_2h.drop(['%s_minus_1'%col], axis=1, inplace=True)
    
    
    # Compte diffs
    
    full_period_1h=pd.date_range('2013-11-23 16:00:00',periods=18524, freq='1H')
    pd_full_period_1h = pd.DataFrame(index=full_period_1h)
    agadir_full_p_1h = pd_full_period_1h.join(agadir[['T','P','U']])
    
    col_shift_list = ["T","P","U"]
    
    for x_offset in xrange(2):
        agadir_full_p_1h = agadir_full_p_1h.join(agadir_full_p_1h.shift(periods=1, axis=0)[col_shift_list],rsuffix='_minus_1',how='left')
        
        for col in col_shift_list:
            agadir_full_p_1h.loc[agadir_full_p_1h[col].isnull(), col] = agadir_full_p_1h['%s_minus_1' % col]
        
        for col in col_shift_list:
            agadir_full_p_1h.drop(['%s_minus_1'%col], axis=1, inplace=True)
    
    full_period=pd.date_range('2013-11-23 16:00:00',periods=9262, freq='2H')
    pd_full_period = pd.DataFrame(index=full_period)
    agadir_full_p = pd_full_period.join(agadir_full_p_1h)
    agadir_offset = agadir_full_p[['T','P','U']].join(agadir_full_p[['T','P','U']].shift(periods=1, axis=0), rsuffix='_minus_1')
    agadir_offset['P_diff'] = agadir_offset['P'] - agadir_offset['P_minus_1']
    agadir_offset['T_diff'] = agadir_offset['T'] - agadir_offset['T_minus_1']
    agadir_offset['U_diff'] = agadir_offset['U'] - agadir_offset['U_minus_1']
    agadir_offset.drop(['T','P','U','T_minus_1','P_minus_1','U_minus_1'],axis=1,inplace = True)
    agadir_offset.dropna(inplace=True)



    all_micro_full_p = pd.concat([train_micro_2h, test_micro_2h]).sort_index()
    all_micro_full_p = pd_full_period.join(all_micro_full_p[['temp','leafwet_lwscnt']])
    
    all_micro_offset = all_micro_full_p[['temp','leafwet_lwscnt']].join(all_micro_full_p[['temp','leafwet_lwscnt']].shift(periods=1, axis=0), rsuffix='_minus_1')
    all_micro_offset['temp_diff'] = all_micro_offset['temp'] - all_micro_offset['temp_minus_1']
    all_micro_offset['leaf_diff'] = all_micro_offset['leafwet_lwscnt'] - all_micro_offset['leafwet_lwscnt_minus_1']
    all_micro_offset.drop(['temp','leafwet_lwscnt','temp_minus_1','leafwet_lwscnt_minus_1'], axis=1,inplace=True)
    all_micro_offset.dropna(inplace=True)



    date_model = pd.concat([train_target,test_submit]).sort_index()

    sun_model = fognet_utils.sun_at_fognets()
    
    date_model = date_model.join(sun_model)
    
    
    if 1:
        # Model based only on date
        print("Linear model on date")
        lin_model = date_model.copy()
                        
        
        lr = LinearRegression(fit_intercept=True,normalize=True,copy_X=True,n_jobs=6)
    
        print("fit")
        #lr.fit(lin_model.loc[date_model['is_train'] == True, ['m','woy']].as_matrix().astype(np.float32), lin_model.loc[date_model['is_train'] == True, 'yield'].as_matrix().astype(np.float32))
        lr.fit(lin_model.loc[date_model['is_train'] == True, ['sun_angle','hours_of_sun','hours_of_night','adjusted_hours_of_sun','is_day']].as_matrix().astype(np.float32), lin_model.loc[date_model['is_train'] == True, 'yield'].as_matrix().astype(np.float32))

        print("predict")
        #predicted_yield = lr.predict(lin_model.loc[date_model['is_train'] == False, ['m','woy']].as_matrix().astype(np.float32))
        predicted_yield = lr.predict(lin_model.loc[date_model['is_train'] == False, ['sun_angle','hours_of_sun','hours_of_night','adjusted_hours_of_sun','is_day']].as_matrix().astype(np.float32))
        
        lin_model.loc[date_model['is_train'] == False, 'yield_lin'] = predicted_yield
        lin_model.to_hdf('fognet_datemodel.hdf','datemodel',mode='w',complib='blosc')
        
        #print(lin_model)     

    if 1:
        # Model based only on agadir
        print("Agadir model")

        agadir_model = date_model.copy()

        if 1:
            group_list = GenerateGroupsBy(agadir, "agadir", hours_grouped = 4)
            for grouped_result in group_list:
                agadir_model = agadir_model.join(grouped_result)
                print(agadir_model.isnull().sum(axis=0))
                print(agadir_model.count(axis=0))                                        

        agadir_model.dropna(inplace=True)


        columns_selection = list(agadir_model.columns)
        columns_selection.remove('yield')
        columns_selection.remove('is_train')

        columns_selection = remove_correlated_features(agadir_model, columns_selection)


        cv_list = fognet_utils.compute_cv_ranges(lin_model)
        
        if 1:
            work_model = agadir_model.copy()

            sub_form = lin_model[lin_model['is_train'] == True].copy()
            sub_form = sub_form[ (sub_form.index.day <= 4) & (sub_form.index.day >= 1)]
            
            evaluations_list = []

            # now we use the add_eval_sets routine
            # in order to compute models  with different feature and different feature aggregations
            # for each set of featre it will compute "best" hyper parameters for xgboost

            if 1:
                    # best 759 results
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24") ], set_name = "sidi_wind_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24") ], set_name = "sidi_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 24, "guelmin_24") ], set_name = "guelmin_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(guelmin, 24, "guelmin_24") ], set_name = "guelminsidi_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(guelmin, 24, "guelmin_24"),(sidi_wind, 24, "sidi_wind_24") ], set_name = "guelminsidiwwind_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 24, "sidi_24"),(guelmin, 24, "guelmin_24"),(sidi_wind, 24, "sidi_wind_24"),(guelmin_wind, 24, "guelmin_wind_24") ], set_name = "guelminwwindsidiwwind_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 24, "guelmin_24"),(sidi, 24, "sidi_24") ,(sidi_wind, 24, "sidi_wind_24"),(guelmin_wind, 24, "guelmin_wind_24") , (sidi_pa, 24, "sidi_pa_24") ], set_name = "guelminwwindsidiwwindpa_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 24, "guelmin_24"),(sidi, 24, "sidi_24") ,(sidi_wind, 24, "sidi_wind_24"),(guelmin_wind, 24, "guelmin_wind_24") , (sidi_pa, 24, "sidi_pa_24"), (sidi_Tg, 24, "sidi_tg_24") ], set_name = "guelminwwindsidiwwindpatg_24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 12, "guelmin_12"),(sidi, 12, "sidi_12") ,(sidi_wind, 12, "sidi_wind_12"),(guelmin_wind, 12, "guelmin_wind_12") , (sidi_pa, 12, "sidi_pa_12"), (sidi_Tg, 24, "sidi_tg_24") ], set_name = "guelminwwindsidiwwindpa12_tg24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 12, "guelmin_12"),(sidi, 12, "sidi_12") ,(sidi_wind, 12, "sidi_wind_12"),(guelmin_wind, 12, "guelmin_wind_12")  ], set_name = "guelminwwindsidiwwind12", evaluations_list =  evaluations_list, cv_list = cv_list )

                    f = gzip.open("fognet_compare_test_24d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
                

                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 6, "sidi_wind_6") ], set_name = "sidi_wind_6", evaluations_list =  evaluations_list, cv_list = cv_list )                                
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(sidi_wind, 6, "sidi_wind_6") ], set_name = "sidi_wind_246", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(sidi, 6, "sidi_6") ], set_name = "sidi_6_wind24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(sidi, 6, "sidi_6") ,(sidi_pa, 6, "sidi_pa_6")], set_name = "sidi_wpa6_wind24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    f = gzip.open("fognet_compare_test_sidi62d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
    
                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6") ], set_name = "guelmin_6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(guelmin, 6, "guelmin_6") ], set_name = "guelmin_6_wind24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_pa, 6, "sidi_pa_6") ], set_name = "guelmin_6_sidiwpa6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 24, "sidi_wind_24"),(guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_pa, 6, "sidi_pa_6") ], set_name = "guelmin_6_wind24_sidiwpa6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_wind, 6, "sidi_wind_6") ], set_name = "guelmin_6_sidiwwind6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 6, "guelmin_6"),(sidi, 6, "sidi_6") ,(sidi_wind, 6, "sidi_wind_6"),(guelmin_wind, 6, "guelmin_wind_6") ], set_name = "guelminwwind_6_sidiwwind6", evaluations_list =  evaluations_list, cv_list = cv_list )
    
                    f = gzip.open("fognet_compare_test_guelmin62d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
    
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2") ], set_name = "sidi_2", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 2, "sidi_wind_2") ], set_name = "sidi_wind_2", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidi_wind_2") ], set_name = "sidi_wwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 2, "sidi_2"), (sidi_pa, 2, "sidi_pa_2") ], set_name = "sidi_wpa_2", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_sidi2d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         

    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 4, "sidi_4") ], set_name = "sidi_4", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    add_eval_sets(work_model = work_model, set_list = [ (sidi_wind, 4, "sidi_wind_4") ], set_name = "sidi_wind_4", evaluations_list =  evaluations_list, cv_list = cv_list )            
                    
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 4, "sidi_4"), (sidi_wind, 4, "sidi_wind_4") ], set_name = "sidi_wwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    add_eval_sets(work_model = work_model, set_list = [ (sidi, 4, "sidi_4"),(sidi_pa, 4, "sidi_pa_4"),(sidi_wind, 4, "sidi_wind_4"), (sidi_avg_rain, 4, "sidi_avg_4"), (sidi_Tg, 24, "sidi_tg_24")   ], set_name = "sidi_pawindavgtg_4", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_sidi4d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()         
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2") ], set_name = "guelmin_2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin_wind, 2, "guelminw_2") ], set_name = "guelminw_2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"), (guelmin_wind, 2, "guelminw_2") ], set_name = "guelminwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"),(sidi, 2, "sidi_2")  ], set_name = "guelmin_2_sidi2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"),(sidi, 2, "sidi_2") ,(sidi_wind, 2, "sidi_wind_2"),(guelmin_wind, 2, "guelmin_wind_2") ], set_name = "guelminwwind_2_sidiwwind2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 2, "guelmin_2"),(sidi, 2, "sidi_2") ,(sidi_wind, 2, "sidi_wind_2"),(guelmin_wind, 2, "guelmin_wind_2") ,(sidi_pa, 4, "sidi_pa_4") ], set_name = "guelminwwind_2_sidiwwind2_sidipa4", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_guelmin2d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()        
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4") ], set_name = "guelmin_4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin_wind, 4, "guelminw_4") ], set_name = "guelminw_4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4"), (guelmin_wind, 4, "guelminw_4") ], set_name = "guelminwwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4"),(sidi, 4, "sidi_4")  ], set_name = "guelmin_4_sidi4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (guelmin, 4, "guelmin_4"),(sidi, 4, "sidi_4") ,(sidi_wind, 4, "sidi_wind_4"),(guelmin_wind, 4, "guelmin_wind_4") ], set_name = "guelminwwind_4_sidiwwind2", evaluations_list =  evaluations_list, cv_list = cv_list )
    
        
                    f = gzip.open("fognet_compare_test_guelmin4d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()        
    
    
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2") ], set_name = "micro_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2") ], set_name = "micro_sidiguelmin_2", evaluations_list =  evaluations_list, cv_list = cv_list )                                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (guelmin_wind, 2, "guelminw_2") ], set_name = "micro2_guelminwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (sidi_wind, 2, "sidiwind_2") ], set_name = "micro2_sidiwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2") ], set_name = "micro2_sidiwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin, 2, "guelmin_2") ], set_name = "micro2_sidiwindguelmin_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2") ], set_name = "micro2_sidiwwindguelminwwind_2", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24') ], set_name = "micro2_sidiwwindguelminwwind_2tg24", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24'), (sidi_pa, 6, 'sidipa_6') ], set_name = "micro2_sidiwwindguelminwwind_2tg24pa6", evaluations_list =  evaluations_list, cv_list = cv_list )
                    
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 2, "micro_2"), (guelmin, 2, "guelmin_2"), (sidi, 2, "sidi_2"), (sidi_wind, 2, "sidiwind_2"), (guelmin_wind, 2, "guelminwind_2"), (sidi_Tg, 24, 'siditg_24'), (sidi_pa, 6, 'sidipa_6'), (all_micro_2h_460, 2, 'micro460_2') ], set_name = "micro2_sidiwwindguelminwwind460_2tg24pa6", evaluations_list =  evaluations_list, cv_list = cv_list )
        
        
                    f = gzip.open("fognet_compare_test_micro2d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()        
    
                if 1:
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4") ], set_name = "micro_4", evaluations_list =  evaluations_list, cv_list = cv_list )                                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4"), (guelmin, 4, "guelmin_4"), (guelmin_wind, 4, "guelminw_4") ], set_name = "micro4_guelminwwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )                
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4"), (sidi_wind, 4, "sidiwind_4") ], set_name = "micro2_sidiwind_4", evaluations_list =  evaluations_list, cv_list = cv_list )
                    add_eval_sets(work_model = work_model, set_list = [ (all_micro_2h, 4, "micro_4"), (sidi_wind, 4, "sidiwind_4"), (guelmin, 4, "guelmin_4") ], set_name = "micro4_sidiwindguelmin_4", evaluations_list =  evaluations_list, cv_list = cv_list )
        
                    f = gzip.open("fognet_compare_test_micro4d.gz","wb")
                    cPickle.dump( (evaluations_list, sub_form) , f,cPickle.HIGHEST_PROTOCOL)
                    f.close()                        

            else:
                f = gzip.open("fognet_compare_test_micro4d.gz","r")
                (evaluations_list, sub_form) = pickle.load(f)
                f.close()         


                            
            # Now we have a list of "models", we compare them with each other so that we can sort them
            comparison_list = fognet_utils.compare_evaluations2(evaluations_list, sub_form, cv_list)

            if 1:
                f = gzip.open("fognet_comparison_list10.gz","wb")
                cPickle.dump( comparison_list , f,cPickle.HIGHEST_PROTOCOL)
                f.close()
            else:
                f = gzip.open("fognet_comparison_list5.gz","r")
                comparison_list = pickle.load(f)
                f.close()
            
            # We use the sorted list of model in order to generate the best submission:
            fognet_utils.generate_valid_sub2(evaluations_list, lin_model.copy(), cv_list = cv_list, comparison_list = comparison_list)
                                            
            sys.exit(0)
                        
