# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:03:06 2025

@author: Y00277
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import json
from tensorflow.keras.models import model_from_json
import joblib
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


pretrained_model = xgb.Booster()
pretrained_model.load_model("pretrained_model_n.json")


model_feature_names = pretrained_model.feature_names
if model_feature_names is None:
       model_feature_names = ['workfunction', 'ea_pbe', 
            'interface_sub_Electronegativity_mean', 
            'interface_sub_Polarizability_mean',
            'interface_ads_Electronegativity_mean', 
            'interface_ads_FirstIonizationEnergy_mean', 
            'ads_MetalUnpair',
            'ads_NonMetalUnpair',
            'ads_MetalMendeleev',
            'ads_NonMetalMendeleev',
            'ads_NpValence_mean',
          ]

new_data = pd.read_csv("vdw-metal_features.csv")              
X_new = new_data[model_feature_names].values
y_new = new_data['SBH_n'].values
dnew = xgb.DMatrix(X_new, label=y_new, feature_names=model_feature_names)

params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'eta': 0.05,
    'max_depth': 6,                  
    'subsample': 0.9,               
    'learning_rate': 0.09,           
#    'n_estimators': 300,            
#    'reg_lambda': 1.0,
    'colsample_bytree': 0.9,  
    'seed': 23,
}

# training
updated_model = xgb.train(
    params,
    dtrain=dnew,
    num_boost_round=28,
    xgb_model=pretrained_model,
    verbose_eval=True
)

pred_data = pd.read_csv("vdw-metal_features.csv")          
X_pred = pred_data[model_feature_names].values
y_true = pred_data['SBH_n'].values
dpred = xgb.DMatrix(X_pred, feature_names=model_feature_names)
y_pred = updated_model.predict(dpred)

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"- MSE: {mse:.4f}")
print(f"- R²: {r2:.4f}")

# save model
updated_model.save_model("updated_model_n-vdw.json")



