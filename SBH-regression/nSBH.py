# -*- coding: utf-8 -*-
"""
Created on Mon Jun  11 09:40:29 2025

@author: Y00277
"""
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.impute import KNNImputer
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

#%%
import os

"""
Check file path

"""
print(os.getcwd())
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
os.chdir(BASE_DIR)

#%%

"""
Read Data

"""
data = pd.read_csv('Data_All_M2d.csv')
data = data[data['Note'] != 0]
data = data.dropna(subset = ['SBH_n'])


X = data[['workfunction', 'ea_pbe', 
     'interface_sub_Electronegativity_mean', 
     'interface_sub_Polarizability_mean',
     'interface_ads_Electronegativity_mean', 
     'interface_ads_FirstIonizationEnergy_mean', 
     'ads_MetalUnpair',
     'ads_NonMetalUnpair',
     'ads_MetalMendeleev',
     'ads_NonMetalMendeleev',
     'ads_NpValence_mean',     
]]
Y = data[['SBH_n']]

#%%
"""
Set Paramters

"""

kf = KFold(n_splits=10, shuffle=True, random_state=52)

params = {
    'objective':'reg:squarederror',
    'tree_method': 'hist',    
    'n_estimators': 200,             
    'max_depth': 6,                 
    'learning_rate': 0.06,           
    'subsample':0.8,                 
    'colsample_bytree':0.9,           
    'seed': 62,                      
    'verbosity': 0,                  
    'nthread': -1,                  
}

best_model = xgb.XGBRegressor(**params)
cv_scores = cross_val_score(best_model,X,Y,cv=kf,scoring='r2')
Y_pred = pd.DataFrame(cross_val_predict(best_model,X,Y,cv=kf),index=Y.index,columns=['Pred'])

#%%
"""
Data Cleaning
"""
Combine = pd.concat([Y,Y_pred],axis=1)
Y_post = Combine.drop(Combine[abs(Combine['SBH_n']-Combine['Pred'])>0.5].index).iloc[:,0]
X_post = X.drop(Combine[abs(Combine['SBH_n']-Combine['Pred'])>0.5].index)

cv_scores_post = cross_val_score(best_model,X_post,Y_post,cv=kf,scoring='r2')
Y_pred_post = pd.DataFrame(cross_val_predict(best_model,X_post,Y_post,cv=kf),index=Y_post.index,columns=['Pred_post'])

#%%
"""
RFECV

"""
selector = RFECV(best_model, step=1, cv=kf, scoring='r2', n_jobs = -1) 
selector.fit(X_post, Y_post.values.ravel())

## Save Cross Validation Info
cv_results = pd.DataFrame(selector.cv_results_)
fold_scores = [cv_results[f'split{i}_test_score'] for i in range(10)]
mean_scores = cv_results['mean_test_score'] 
iter_feature = cv_results['n_features'] 

## Get Selected Featrue Name
feature_name = X.columns.values.tolist()
sel_index = [i for i in range(len(selector.support_)) if selector.support_[i]]
sel_feature_name = []
for i in sel_index:
    sel_feature_name.append(feature_name[i])

#%%
"""
Plot Scores vs # of Features

"""

plt.figure(figsize=(12, 8), dpi=300)
plt.title('SBH_n (RFECV)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Number of Features', fontsize=14, labelpad=15)
plt.ylabel('CrossValidation Score ( R2 )', fontsize=14, labelpad=15)
plt.gca().set_facecolor('#f7f7f7')

## 10-fold scores
for i in range(10):
    plt.plot(iter_feature, fold_scores[i], marker='o', color='gray', linestyle='-', 
             linewidth=0.8, alpha=0.6)
    
## mean scores
plt.plot(iter_feature, mean_scores, marker='o', color='#696969', linestyle='-', 
         linewidth=3, label='Mean CV R2')

## Optimal
plt.axvline(x=selector.n_features_, color='#E76F51', linestyle='--', linewidth=2, label=f'Optimal = {selector.n_features_}')
plt.legend(fontsize=12, loc='best', frameon=True, shadow=True, facecolor='white', framealpha=0.9)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()

#%%
"""
After Feature Reduction

"""
X_trans = pd.DataFrame(selector.transform(X_post),index=X_post.index, columns=sel_feature_name)
Y_trans = Y_post.values.ravel()

cv_scores = cross_val_score(best_model,X_trans,Y_trans,cv=kf,scoring='r2')
print(cv_scores)
print(f'After Feature Reduction, Train CV R2: {cv_scores.mean()}')
Y_pred_trans=cross_val_predict(best_model, X_trans, Y_trans, cv=kf)
print('After Feature Reduction, Train CV MSE = %.3f' % mean_squared_error(y_true=Y_trans, y_pred=Y_pred_trans))

plt.figure(figsize=(12, 8))
plt.scatter(Y_trans,Y_pred_trans, alpha=0.4, label=f'{len(X_trans)} Sample')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.legend()
plt.title('Feature_SBH_n')
plt.show()

#%%
"""
Features Importance

"""

postrain = best_model.fit(X_trans,Y_trans)

feature_importance = pd.DataFrame({'Feature': sel_feature_name, 'Importance': postrain.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

palet = sns.color_palette("Greens_r",len(feature_importance))
sns.barplot(x='Importance', y='Feature', data=feature_importance[:15], palette=palet, hue='Feature')
plt.title('SBH_n Feature Importance')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

#%%
"""
Train Test

"""

X_train,X_test,Y_train,Y_test = train_test_split(X_trans,Y_trans,test_size=0.2,random_state=52)

cv_scores = cross_val_score(best_model,X_train,Y_train,cv=kf,scoring='r2')
print(cv_scores)
#print(f'Train CV R2: {cv_scores.mean()}')
Y_pred_train = cross_val_predict(best_model, X_train, Y_train, cv=kf)
#print('Train CV MSE = %.3f' % mean_squared_error(y_true=Y_train, y_pred=Y_pred_train))

best_model.fit(X_train,Y_train)

##save model for next Prediction
save_path = "pretrained_model_n.json"
best_model.save_model(save_path)  

print('Training Score = ' + str(round(best_model.score(X_train, Y_train),3)))
print('Test R2 = ' + str(round(best_model.score(X_test, Y_test), 3)))
print('Test MSE = %.3f' % mean_squared_error(y_true=Y_test, y_pred=best_model.predict(X_test)))

plt.rcParams['font.style'] = 'normal' 
plt.figure(figsize=(3, 4))
plt.scatter(Y_train, best_model.predict(X_train), alpha=0.9)
plt.scatter(Y_test, best_model.predict(X_test), alpha=0.3)
plt.plot([Y_trans.min(), Y_trans.max()], [Y_trans.min(), Y_trans.max()], 'k--', lw=2)
plt.xlabel('True Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)  
plt.xticks([ -1, 0, 1, 2, 3, 4])
plt.yticks([ -1, 0, 1, 2, 3, 4])
plt.title('SBH_n')
#plt.savefig('Regression-n.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

#%%
"""
SHAP Value

"""
import shap

X = [
     'workfunction', 'ea_pbe', 
          'interface_sub_Electronegativity_mean', 
          'interface_sub_Polarizability_mean',
          'interface_ads_Electronegativity_mean', 
          'interface_ads_FirstIonizationEnergy_mean', 
          'ads_MetalUnpair',
          'ads_NonMetalUnpair',
          'ads_MetalMendeleev',
          'ads_NonMetalMendeleev',
          'ads_NpValence_mean'
     ]

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(data[X])

##shap_values
shap.summary_plot(shap_values, data[X])

##shap_values importance
shap.summary_plot(shap_values, data[X], plot_type="bar")

##shap-interaction-values
shap_interaction_values = shap.TreeExplainer(best_model).shap_interaction_values(data[X])
shap.summary_plot(shap_interaction_values, data[X], max_display=5)
