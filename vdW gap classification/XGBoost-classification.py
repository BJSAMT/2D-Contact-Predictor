# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:25:31 2024

@author: Y00277
"""

import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from xgboost import plot_importance
from mlxtend.plotting import heatmap
import seaborn as sns
from xgboost import XGBClassifier as xgb
import matplotlib.pyplot as plt
import csv
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator
import warnings
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import joblib
plt.rcParams['xtick.direction'] = 'in'; plt.rcParams['ytick.direction'] ='in'
#%% Parameters

RANDOM_SEED = 62
ACTIVE_LEARNING_BATCH_SIZE = 10
ACTIVE_LEARNING_TEST_TARGET = 0.7
CLASS_THRESHOLD = 0.5
EVAL_METRIC = "auc"
EARLY_STOPPING_ROUNDS = 2
VAL_RATIO = 0.2
SCORE_FN_LIST = [accuracy_score]

BASE_PARAM_DICT = {
    "objective": "binary:logistic",
    "tree_method": "exact",
    "random_state": RANDOM_SEED,
    # "eval_metric": EVAL_METRIC,
    # "early_stopping_rounds": EARLY_STOPPING_ROUNDS
}
#%% Data

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap
warnings.filterwarnings('ignore')
data = pd.read_csv('total-data-vdW-gap.csv',index_col=0)

#%% Training

clf_params_dict = BASE_PARAM_DICT.copy()

# clf_params_dict.update(
#     {
#         "n_estimators": 47,                    #弱评估器的个数，快速调整模型
#         "learning_rate": 0.17785360874042755,
#         "max_depth": 7,                        #avoid overfiting parament
#         "gamma": 0.11160062104409085,
#         "reg_alpha": 0.033095551338589546,
#         "reg_lambda": 0.3367177222521667
#     }
# )

## features
X = data [['interface_Polarizability_mean','interface_ads_CovalentRadius_mean','interface_GSenergy_pa_mean',\
            'ads_FirstIonizationEnergy_mean','ads_GSvolume_pa_mean','ads_NUnfilled_mean','interface_sub_CovalentRadius_mean',\
                'interface_ads_Electronegativity_mean','interface_ElectronAffinity_mean', 'interface_MendeleevNumber_mean']]    
    
Y = data['vdwgap_class']

## split data into train and test sets by train_test_split method
seed = 65
test_size = 0.3
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

clf = xgb(max_depth=7, n_estimators=40)
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train), clf.score(x_test, y_test))
y_pred = clf.predict(x_test)
y_score = clf.predict_proba(x_test)[:,-1]
fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)

##auc
fig = plt.figure(figsize=(4, 4))
plt.plot(fpr, tpr, c='r', linewidth=1)
plt.xlim([0, 1]); plt.ylim([0, 1])
prec = precision_score(y_test, y_pred); accu = accuracy_score(y_test, y_pred) 
recall = recall_score(y_test, y_pred); auc = roc_auc_score(y_test, y_score)

plt.text(0.22, 0.83, 'AUC = %.3f\nAccuracy = %.3f\nPrecision = %.3f\nRecall = %.3f' % (auc, accu, prec, recall),\
          fontsize=10, verticalalignment='top')
y_train_pred = clf.predict(x_train)

plt.tick_params(labelsize=10)
print('test set: precision:', prec, 'accuracy:', accu, 'recall:', recall, 'auc:', auc)
print('train set: precision:', precision_score(y_train, y_train_pred), 'accuracy:', accuracy_score(y_train,y_train_pred), \
      'recall:', recall_score(y_train, y_train_pred))
ax=plt.gca()
minor_locator_y = AutoMinorLocator(2)
ax.get_xaxis().set_minor_locator(minor_locator_y)
ax.get_yaxis().set_minor_locator(minor_locator_y)
ax.spines['bottom'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR',fontsize=14)

ax1 = fig.add_axes([0.57, 0.24, 0.28, 0.28])
cm = confusion_matrix(y_test, y_pred)
temp = ax1.imshow(cm, cmap='Reds')
fig.colorbar(temp)
ax1.set_xlabel('Predicted label', labelpad=-1) 
ax1.yaxis.set_label_coords(-0.15,0.4)
ax1.set_ylabel('True label', labelpad=-1)
ax1.text(0, 0, cm[0,0], verticalalignment='center', horizontalalignment='center',color='white')
ax1.text(1, 0, cm[0,1], verticalalignment='center', horizontalalignment='center',color='k')
ax1.text(0, 1, cm[1,0], verticalalignment='center', horizontalalignment='center',color='k')
ax1.text(1, 1, cm[1,1], verticalalignment='center', horizontalalignment='center',color='white')
ax1.set_title('CM',fontsize=10)
ax1.tick_params('both',length=0)
#plt.savefig('vdW-gap-classifier.png', dpi=300, bbox_inches='tight')
plt.show()

#%% model scores

# model = xgb()
cv_train_scores = cross_val_score(clf, x_train, y_train, cv=10)
cv_test_scores = cross_val_score(clf, x_test, y_test, cv=10)
print(cv_train_scores)
print("Average training_cv_score:", cv_train_scores.mean())
print(cv_test_scores)
print("Average test_cv_score:", cv_test_scores.mean())
y_pred = clf.predict(x_test)

#%%features and feature importance data output and plot:Method2

feature_importance = clf.feature_importances_
feature_importance_df = pd.DataFrame(data=feature_importance, index=X.columns, columns=["importance"])
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)
feature_importance_df.to_csv('feature_importance.csv', index=True)

feature_matrix = pd.read_csv('feature_importance.csv', index_col=0)
plt.figure(figsize=(3.7, 3))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.tick_params(axis='both', 
               direction='in',   
               width=1,          
               length=4,        
               labelsize=10)    
palet = sns.color_palette("Greens_r",len(feature_matrix))
sns.barplot(x='importance', y=feature_matrix.index, data=feature_matrix.reset_index(), palette=palet, ax=ax)
plt.title('vdW-gap Feature Importance')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

#%%heatmap features-Correlation 
 
correlation_matrix = np.corrcoef(X, rowvar=False) 
correlation_matrix = pd.DataFrame(data = correlation_matrix, index=X.columns)

plt.figure(figsize=(3.7, 3))
ax = sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    xticklabels=correlation_matrix.index,  
    yticklabels=correlation_matrix.index   
)

ax.tick_params(
    axis='both',         
    which='major',       
    length=3,            
    width=1,              
    labelsize=10          
)

plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()

#%%
"""
SHAP Value

"""
import shap

X = ['interface_Polarizability_mean','interface_ads_CovalentRadius_mean',
     'interface_GSenergy_pa_mean','ads_FirstIonizationEnergy_mean',
     'ads_GSvolume_pa_mean','ads_NUnfilled_mean','interface_sub_CovalentRadius_mean',
      'interface_ads_Electronegativity_mean','interface_ElectronAffinity_mean', 
      'interface_MendeleevNumber_mean']

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(data[X])

#shap-value
shap.summary_plot(shap_values, data[X])

#shap-value importance
shap.summary_plot(shap_values, data[X], plot_type="bar")

#shap_interaction_values
shap_interaction_values = shap.TreeExplainer(clf).shap_interaction_values(data[X])
shap.summary_plot(shap_interaction_values, data[X], max_display=5)

