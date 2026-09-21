# written by Liangying, 2025/06/10
#%%

import os
import pandas as pd
import pickle
import dask.dataframe as dd
from plotnine import *
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
#from missingpy import MissForest
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, RocCurveDisplay,confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import joblib
import shap
import pickle
from xgboost import XGBClassifier
from fancyimpute import KNN  
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.stats import pearsonr, spearmanr


imputers = {
    #"missforest":MissForest(random_state = 13),
    "None": "passthrough",
    "MICE": IterativeImputer(max_iter=10),
    "KNN": KNNImputer(n_neighbors=3),
    "Simple": SimpleImputer(strategy = 'most_frequent')
}

transformers = {
    "yeo": PowerTransformer(method = 'yeo-johnson'),
    "None": "passthrough"
}

scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "None": "passthrough"
}

models = {
    "LR": LogisticRegression(n_jobs=-1, random_state = 12), 
    "xgboost": XGBClassifier(objective = 'binary:logistic', eval_metric = 'logloss', n_jobs=-1,
                              use_label_encoder = False, random_state = 325)               
}


def find_param_grid(model_name):
    if model_name == "xgboost":
        param_grid = [
        {
            'clf': [models.get('xgboost')],
            'clf__learning_rate': [0.01, 0.1, 0.2],
            'clf__n_estimators': [100, 300, 500],
            'clf__max_depth': [3, 5, 7],
            'clf__subsample': [0.8, 1.0],
            'clf__min_child_weight': [1, 3, 5],
            'clf__colsample_bytree': [0.6, 0.8, 1],
            'clf__scale_pos_weight': [20.45]
        }]

    if model_name == "LR":
        param_grid = [
        {
            'clf': [models.get('LR')],
            #'clf__C': [0.1, 1, 10],
            #'clf__C': [0.001],
            #'clf__penalty': ['l2'],   
            'clf__penalty': [None],
            'clf__solver': ['saga']   # saga is fast
        }]
    
    return param_grid

def find_pipe(model_name):
    if model_name == "xgboost":

        pipe_con = Pipeline([
        ("imputer", imputers.get("None")),
        ("transformers", transformers.get('None')),
        ("scalers", scalers.get('None'))
        ])
        
        pipe_discrete = Pipeline([
            ("imputer", imputers.get("None"))
        ])

        pipe_norminal = Pipeline([
          ("imputer", imputers.get("Simple")),
          ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        pipe_comb = ColumnTransformer(
            transformers=[
            ("pipe_con", pipe_con, cols_con),
            ("pipe_discrete", pipe_discrete, cols_discrete),
            ("pipe_norminal", pipe_norminal, cols_norminal),
            
        ])

        pipe = Pipeline([
            #("imputer", imputers.get("None")),   # can atually ignore
            ('tranform', pipe_comb),
            ('clf', models.get(model_name))
        ]
        )

    if model_name == "LR":

        pipe_con = Pipeline([
        ("imputer", imputers.get("MICE")),       
        #("transformers", transformers.get('yeo')),
        ("scalers", scalers.get('standard'))
        ])

        pipe_discrete = Pipeline([
            ("imputer", imputers.get("None"))
        ])

        pipe_norminal = Pipeline([
            ("imputer", imputers.get("Simple")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        pipe_comb = ColumnTransformer(
            transformers=[
            ("pipe_con", pipe_con, cols_con),
            ("pipe_discrete", pipe_discrete, cols_discrete),
            ("pipe_norminal", pipe_norminal, cols_norminal)
        ])

        pipe = Pipeline([
            ('tranform', pipe_comb),
            ('clf', models.get(model_name))
        ]
        )
    return pipe


def explainer(model_n):
    grid= joblib.load(os.path.join(path_newFinal, "ML_models", model_n, f"{model_n}_gridsearch_model_new.pkl"))
    X_validation = pd.read_csv(os.path.join(path_newFinal, "ML_models", "X_validation.csv"))
    best_model = grid.best_estimator_.steps[-1][1]
    preproc_step = grid.best_estimator_.steps[0][1]
    X_validation_onehot = preproc_step.transform(X_validation)
    print("Best params: ", grid.best_params_)
    print("Best score: ", grid.best_score_)
    
    #X_derivation = pd.read_csv(os.path.join(path_newFinal, "ML_models", "X_derivation.csv"))
    explainer = shap.TreeExplainer(best_model, data = preproc_step.transform(X_derivation))
    np.random.seed(42)
    idx_test_random = np.random.choice(X_validation_onehot.shape[0], size = 5000, replace = False)
    data_shap = X_validation_onehot[idx_test_random, :]
    y_shap = y_validation[idx_test_random]
    shap_values_main = explainer.shap_values(data_shap)
    shap_values_expected = explainer.expected_value
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "data_shap.npy"), data_shap)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "shap_values_main.npy"), shap_values_main)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "shap_values_expected.npy"), shap_values_expected)
    
    onehot = preproc_step.transformers_[2][1]
    cols = preproc_step.transformers_[2][2]
    mean_abs_shap = np.abs(shap_values_main).mean(0)
    idx_ordered = np.argsort(mean_abs_shap)[::-1]
    feature_importance = pd.DataFrame({'feature':np.array(feature_names_final_onehot)[idx_ordered],
                                       'mean_abs_shap': mean_abs_shap[idx_ordered]})
    N = 20
    feature_dict = dict(zip(feature_importance['feature'], feature_importance['feature_new']))
    feature_names_final_onehot_new = [feature_dict.get(f) for f in feature_names_final_onehot]


def pearson_corr(group):
    n = len(group)
    dfree = n - 2 if n >= 2 else None
    if len(group) < 2:
        return pd.Series({"r": None, "p": None})
    r, p = pearsonr(group["feature_value"], group["shap_value"])
    return pd.Series({"r": r, "df":dfree, "p": p})
