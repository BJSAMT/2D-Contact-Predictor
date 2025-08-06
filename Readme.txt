1, Data_All_M2d.csv contains all Schottky barrier height (SBH) related data for Type-I metal-semiconductor contacts, used to train the nSBH (pretrained_model_n.json) and 
    pSBH (pretrained_model_p.json)  regression models based on the nSBH.py and pSBH.py, respectively.

2, Semi-metal_features.csv and vdw-metal_features.csv contain all Schottky barrier height (SBH) related data  for Type-II and Type-III metal-semiconductor contacts, respectively. 
    Based on the nSBH/pSBH models, these datasets are primarily used for incremental training of the universal nSBH and pSBH models: updated_model_n-semi/vdw.json and updated_model_p-semi/vdw.json.

3, Total-data-vdW-gap.csv contains all van der Waals (vdW) gap data for Type-I contacts, used to train a vdW barrier height XGBoost classification model.