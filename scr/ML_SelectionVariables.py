import numpy as np
import pandas as pd
from statsmodels.iolib.summary2 import summary_col
from functools import reduce

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from patsy import dmatrices, dmatrix, NAAction # useful for converting formulas to matrices
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import GridSearchCV, cross_val_score # Grid search

from matplotlib import style
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') # eliminar warning messages 

#%%

# Loading District Census data 1993 to determine control group using ML approach

district_educ_1993 = pd.read_excel(r"data\raw\INEI\district_education.xlsx").copy()
district_elect_1993 = pd.read_excel(r"data\raw\INEI\district_electricity.xlsx").copy()
district_floor_1993 = pd.read_excel(r"data\raw\INEI\district_floor_type.xlsx").copy()
district_wall_1993 = pd.read_excel(r"data\raw\INEI\district_wall_type.xlsx").copy()
district_water_1993 = pd.read_excel(r"data\raw\INEI\district_water_provision.xlsx").copy()
district_work_1993 = pd.read_excel(r"data\raw\INEI\district_workforce_econsector.xlsx").copy()

dfs_to_merge = [
    district_elect_1993[['ubigeo','elect_share']],
    district_floor_1993[['ubigeo','floor_type']],
    district_wall_1993[['ubigeo','concrete_wall']],
    district_water_1993[['ubigeo','public_water']],
    district_work_1993[['ubigeo','skill_workforce','primary_sector_workfore']]
]

# Append dataset

data_1993 = reduce(
    lambda left, right: pd.merge(
        left, right,
        on="ubigeo",
        how="left",
        validate="1:1"
    ),
    dfs_to_merge,
    district_educ_1993
)

data_1993

# District's economic data from original paper 

data_paper = pd.read_stata(r"data\raw\INEI\data_person_all.dta")

# We keep only variables at district level and with no missing values for the year 2001 (pre-treatment period) to predict the probability of being treated.

data_paper = data_paper[["ubigeo","d2","year","canon_p_nuevossoles","pob93","denspob","perurb","linea","linpe","dist_caja_r","ejec_k","altura","superfic"]].copy()
data_paper = (
    data_paper[(~ data_paper.ejec_k.isna())].sort_values(['ubigeo','year'])
    .drop_duplicates(subset='ubigeo',
                     keep='first')
)

data_paper_district = (
    data_paper
    .assign(
        spend_k_percapita = data_paper.ejec_k/data_paper.pob93,
        dcloseMine = np.where( data_paper.d2 == "Close to city",1,0)
    )
    .drop(columns=['ejec_k','year'])

)

data_paper_district.info()

#Merge two sources of data at district level

data_district_merge = pd.merge( 
                               data_paper_district,
                               data_1993,
                               on='ubigeo',
                               how = 'inner',
                               validate = '1:1'
                               )
data_district_merge
data_district_merge.columns
# We use socioeconomic and electoral control variables

controls_vars = ['canon_p_nuevossoles','pob93','perurb', 'linea', 'linpe','altura',  'superfic', 'spend_k_percapita', 'high_educ', 'elect_share', 'floor_type', 'concrete_wall', 'public_water', 'skill_workforce', 'primary_sector_workfore']

formula = f'dcloseMine ~  ' + ' + '.join(controls_vars)

# Similar to model Matrix in R, dmatrices library helps us to get all variables from a formula regression model

y, X = dmatrices(formula, data= data_district_merge, return_type='dataframe')

X = X.drop(["Intercept"], axis = 1)
### Cross Validation for model performance and GreedSearch
# 1. Parameter Grids for GreeSearch

# RBF + Logistic
param_grid_rbf = {
            'rbf__gamma'        : [0.01, 0.1, 0.5, 1.0],
            'rbf__n_components' : [50, 100, 200],
            'logistic__C'       : [0.01, 0.1, 1.0, 10.0],
            'logistic__penalty' : ['l2'],
}

# Random Forest
param_grid_rf = {
    'rf__n_estimators'    : [200, 500], # number of trees in the forest 
    'rf__max_depth'       : [None, 5, 10], # maximum number of splits 
    'rf__min_samples_split': [20, 40, 80], # minimum number of observation required to split
    'rf__max_features'    : ['sqrt', 'log2'], # number of features at each split 
}

# Decision Tree
param_grid_tree = {
    'tree__max_depth'        : [3, 5, 6, 8, 10],  # maximum number of splits 
    'tree__min_samples_split': [20, 40, 80], # minimum number of observation required to split
    'tree__criterion'        : ['gini', 'entropy'], # impurity measure to evalaute split quality
}

# Gradient Boosting
param_grid_gb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2], # set the controbution of each tree
    'n_estimators' : [100, 300, 500], # number of tress added sequentially 
    'max_depth'    : [3, 5, 7], #  Depth of each tree
    'subsample'    : [0.7, 0.85, 1.0], # fraction of training observations
}

# 2.0 Set up models and Grids

model_configs = [
    {
        'name': 'RBF + Logistic',
        'model': Pipeline([
            ('rbf',      RBFSampler(random_state=1)),
            ('logistic', LogisticRegression(
                max_iter    = 1000,
                random_state= 1,
                solver      = 'lbfgs'   # efficient for medium datasets
            ))
        ]),
        'grid': param_grid_rbf
    },
    {
        'name': 'Random Forest',
        'model': Pipeline([
            ('scale', StandardScaler()),
            ('rf',    RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        'grid': param_grid_rf
    },
    {
        'name': 'Decision Tree',
        'model': Pipeline([
            ('scale', StandardScaler()),
            ('tree',  DecisionTreeClassifier(random_state=42))
        ]),
        'grid': param_grid_tree
    },
    {
        'name': 'Gradient Boosting',
        'model': GradientBoostingClassifier(random_state=42),
        'grid': param_grid_gb
    },
]
# 3. Conducting Grid search and evaluation

# I use outer and inner CV to obtain ubiased model evaluation as 
# selection of hyperparameters (Grid search) and error model as sample for selection and evaluation are completely separated 

CV_OUTER = 5   # cross-val folds for final evaluation
CV_INNER = 3   # cross-val folds inside GridSearch

results = []

for cfg in model_configs:

    print(f"  Tuning: {cfg['name']}")

    # Grid search (inner CV)
    
    gs = GridSearchCV(
        estimator  = cfg['model'],
        param_grid = cfg['grid'],
        scoring    = 'neg_log_loss', # negative of function loss 
        cv         = CV_INNER,
        n_jobs     = -1,
        verbose    = 1,
        refit      = True           # refit best model on full training data
    )
    gs.fit(X, y.to_numpy())

    # Evaluate best model (outer CV)
    
    best_score = cross_val_score(
        gs.best_estimator_,
        X, y.to_numpy(),
        scoring = 'neg_log_loss', # negative of function loss 
        cv      = CV_OUTER,
        n_jobs  = -1
    ).mean()

    print(f"Best params    : {gs.best_params_}")
    print(f"Inner CV score : {gs.best_score_:.4f}  (neg_log_loss)")
    print(f"Outer CV score : {best_score:.4f}  (neg_log_loss)")

    results.append({
        'Model'          : cfg['name'],
        'Best Params'    : gs.best_params_,
        'Inner CV NLL'   : round(gs.best_score_, 4),
        'Outer CV NLL'   : round(best_score, 4),
        'Best Estimator' : gs.best_estimator_
    })
### Best estimator for propensity score prediction
#  Comparing Models

df_results = pd.DataFrame(results)[['Model', 'Inner CV NLL', 'Outer CV NLL']]
df_results = df_results.sort_values('Outer CV NLL', ascending=False)  # less negative means better

print(df_results.to_string(index=False))

# Selecting best model

best_idx   = df_results['Outer CV NLL'].idxmax() # maximum Neg Log Loss on Outer CV 
best_model_name = df_results.loc[best_idx, 'Model']
best_estimator  = next(r['Best Estimator'] for r in results if r['Model'] == best_model_name)

print(f"Best model: {best_model_name}")
print(f"   Outer CV neg_log_loss: {df_results.loc[best_idx, 'Outer CV NLL']}")
# Propensity scores

ps_raw        = best_estimator.predict_proba(X)[:,1]

# Overlap and Support common

treated  = y.to_numpy().ravel()  == 1
control  = y.to_numpy().ravel()  == 0

# Propensity scores

ps_t = ps_raw[treated]
ps_c = ps_raw[control]

# Region of common support

cs_min = max(ps_t.min(), ps_c.min())
cs_max = min(ps_t.max(), ps_c.max())

# Filter for common support

in_support = (ps_raw >= cs_min) & (ps_raw <= cs_max)

# Covariables and outcome in the summon support 

X_trimmed = X[in_support]
y_trimmed = y[in_support]
ps_trimmed = ps_raw[in_support]

print(f"Number of districts original : {len(X)}")
print(f"Number of district on trimmed dataset  : {len(X_trimmed)}")
print(f"Propensity score trimmed range: [{ps_trimmed.min()}, {ps_trimmed.max()}]\n")

# Getting dataset in the common support

data_trim = data_district_merge.iloc[X_trimmed.index,:]
data_trim['samplepropensityML'] = 1
data_trim
data_trim.to_stata(r"data\processed\DataMLsample.dta",
                                write_index=False)
