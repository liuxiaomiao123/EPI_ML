# written by Liangying, 2025/09/08

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
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import joblib
import shap
import pickle
from xgboost import XGBClassifier

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm
from patsy import dmatrices

import seaborn as sns
import matplotlib.pyplot as plt

#%%
def age_dummy(X_imputed):
    # create dummy variables for age
    bins = [0, 18, 29, 39, 49, 59, 69, 79, 120]  
    labels = ['<18', '18-29','30-39','40-49','50-59','60-69','70-79','80+']

    X_imputed['age_diagnosis_cat'] = pd.cut(X_imputed['age_diagnosis'], bins=bins, labels=labels, right=True)
    X_imputed['sex'] = X_imputed['sex'].map({'F':0, 'M':1})

    X_imputed_dummies = pd.get_dummies(X_imputed, columns = ['age_diagnosis_cat'], drop_first=False)
    X_imputed_dummies = X_imputed_dummies.drop(columns = "age_diagnosis_cat_40-49")   
    X_imputed_dummies = X_imputed_dummies.astype(float)   
    return X_imputed_dummies

def X_preproc(X_train, X_test, feature_intermountain, cols_con, cols_cat):
    X_train_reduced = X_train[feature_intermountain]
    X_test_reduced = X_test[feature_intermountain]

    # impute missing data for continuous and categorical features
    X_train_reduced_con = X_train_reduced[cols_con]
    X_test_reduced_con = X_test_reduced[cols_con]

    imp = IterativeImputer(max_iter=10, random_state=0)
    X_train_reduced_con_imputed = imp.fit_transform(X_train_reduced_con)
    X_test_reduced_con_imputed = imp.transform(X_test_reduced_con)

    X_train_reduced_con_imputed_df = pd.DataFrame(
    X_train_reduced_con_imputed,
    columns = cols_con
    )
    
    X_test_reduced_con_imputed_df = pd.DataFrame(
    X_test_reduced_con_imputed,
    columns = cols_con
    )

    imp_cat = SimpleImputer(strategy = "most_frequent")
    X_train_reduced_cat_imputed = imp_cat.fit_transform(X_train_reduced[cols_cat])
    X_test_reduced_cat_imputed = imp_cat.transform(X_test_reduced[cols_cat])

    X_train_reduced_cat_imputed_df = pd.DataFrame(
    X_train_reduced_cat_imputed,
    columns = cols_cat
    )
    
    X_test_reduced_cat_imputed_df = pd.DataFrame(
    X_test_reduced_cat_imputed,
    columns = cols_cat
    )

    X_train_reduced_imputed = pd.concat([X_train_reduced_con_imputed_df, X_train_reduced_cat_imputed_df], axis = 1)
    X_test_reduced_imputed = pd.concat([X_test_reduced_con_imputed_df, X_test_reduced_cat_imputed_df], axis = 1)

    X_train_preproc = sm.add_constant(age_dummy(X_train_reduced_imputed))
    X_test_preproc = sm.add_constant(age_dummy(X_test_reduced_imputed))

    return X_train_preproc, X_test_preproc


def allMRS():
    risk_scores = {
        'IEIMRS' : ['age', 'erythrocyte distribution width', "sIgA deficiency",'albumin', "lymphocyte count", "count of diuretic use", "count of inpatient visits",
                    "count of anesthetics use", "anion gap", "heart rate", "chloride", "count of opioid use", "count of chemotherapy visits", "platelet count", "hemoglobin",
                    "(systolic) blood pressure", "MCV", "count of body weight measurement", "urea" ,"alkaline phosphatase"],
    
        'All-cause MRS': ["hematocrit", "white blood cell count", "platelet count", "mean corpuscular volume", "mean corpuscular hemoglobin concentration", "red cell distribution width", "mean platelet volume", "sodium", "potassium", "bicarbonate", "calcium", "glucose", "creatinine", "age", "sex"] ,
    
        'FI-LAB': ["albumin", "AST", "(systolic) blood pressure", "calcium", "creatinine", "folate", "glucose", "hemoglobin", "mean corpuscular volume", "alkaline phosphatase", "inorganic phosphorus", "potassium", "protein", "sodium", "TSH", "Thyroxine", "T4", "urea", "VDRL", "vitamin B12", "white blood cell count"],
    
        'APACHE II': ["sodium", "potassium", "creatinine", "white blood cell count", "age", "heart rate", "temperature", "mean blood pressure", "pH", "respiratory rate", "acute renal failure", "hematocrit", "Glasgow coma scale", "FiO2", "severe organ failure or immunocompromise"],
        
        'SAPS II': ["age", "heart rate", "(systolic) blood pressure", "temperature", "Glasgow coma scale", "FiO2", "PaO2", "on mechanical ventilation", "BUN", "urea", "sodium", "potassium", "bicarbonate", "bilirubin", "white blood cell count", "chronic disease", "mode of admission"], 
    
        'SOFA': ["FiO2", "PaO2", "on mechanical ventilation", "platelet count", "Glasgow coma scale", "bilirubin", "mean blood pressure", "creatinine"],
    
        'Hospital MRS 2020': ["hemoglobin", "white blood cell count", "glucose", "platelet count", "urea", "creatinine", "sodium", "potassium", "bicarbonate", "age"] ,
    
        'Hospital MRS 2006': ["age", "albumin", "alkaline phosphatase", "aspartate aminotransferase", "urea", "glucose", "lactate dehydrogenase", "neutrophil count", "white blood cell count"],
    
        'Hospital MRS 2005': ["sodium", "potassium", "urea", 'creatinine', "albumin", "hemoglobin", "white blood cell count", "age", "gender", "mode of admission"], 
        
        'Hospital MRS 2011': ["age", "sodium", "potassium", "chloride", "bicarbonate", "urea", "creatinine", "glucose", "white blood cell count", "neutrophil count", "lymphocyte count", "platelet count", "hemoglobin"]
    
    }
    
    all_factors = sorted(set(f for factors in risk_scores.values() for f in factors))
    
    matrix = pd.DataFrame(0, index=risk_scores.keys(), columns=all_factors)
    for score, factors in risk_scores.items():
        matrix.loc[score, factors] = 1
    
    factor_freq = matrix.sum(axis=0).sort_values(ascending=False)
    matrix = matrix[factor_freq.index]
    
    df_long = matrix.reset_index().melt(id_vars='index', var_name='RiskFactor', value_name='Presence')
    df_long.rename(columns={'index':'RiskScore'}, inplace=True)
    df_long['RiskFactor'] = pd.Categorical(df_long['RiskFactor'], categories=factor_freq.index, ordered=True)
    return df_long


def main():
    RFE(model_n)
    
    feature_intermountain = [
        "lab_result_num_val_4544-3_Hematocrit [Volume Fraction] of Blood by Automated count",
        "lab_result_num_val_6690-2_Leukocytes [#/volume] in Blood by Automated count",
        "lab_result_num_val_777-3_Platelets [#/volume] in Blood by Automated count",
        "lab_result_num_val_787-2_MCV [Entitic volume] by Automated count",
        "lab_result_num_val_786-4_MCHC [Mass/volume] by Automated count",
        "lab_result_num_val_788-0_Erythrocyte distribution width [Ratio] by Automated count",
        "lab_result_num_val_32623-1_Platelet mean volume [Entitic volume] in Blood by Automated count",
        "lab_result_num_val_2951-2_Sodium [Moles/volume] in Serum or Plasma",
        "lab_result_num_val_2823-3_Potassium [Moles/volume] in Serum or Plasma",
        "lab_result_num_val_2028-9_Carbon dioxide, total [Moles/volume] in Serum or Plasma",
        "lab_result_num_val_17861-6_Calcium [Mass/volume] in Serum or Plasma",
        "lab_result_num_val_2345-7_Glucose [Mass/volume] in Serum or Plasma",
        "lab_result_num_val_2160-0_Creatinine [Mass/volume] in Serum or Plasma",
        "age_diagnosis",
        "sex"
    ] 

    cols_cat = ['sex']  
    cols_con = [cl for cl in feature_intermountain if cl not in cols_cat]
    X_derivation_new_preproc, X_validation_new_preproc = X_preproc(X_derivation_new, X_validation_new, feature_intermountain, cols_con, cols_cat)
    y =  np.asarray(y_derivation_new, dtype=float)
    
    model = sm.Logit(y, X_derivation_new_preproc).fit()
    print(model.summary())
    
    model_n = "IMRS_new"
    os.makedirs(os.path.join(path_newFinal, "ML_models", model_n), exist_ok = True)
    joblib.dump(model, os.path.join(path_newFinal, "ML_models", model_n, f"{model_n}_model.pkl"))

    y_pred = model.predict(X_validation_new_preproc)
    pd.DataFrame({"y_pred": y_pred}).to_csv(
        os.path.join(path_newFinal, "ML_models", model_n, "y_pred.csv"), index=False
    )
    
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"), y_pred)
    auc_point, auc_boot, ci_l, ci_u, df_roc, df_roc_ci  = bootstrap_auc_ci(y_validation_new, y_pred)
    
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "auc_point.npy"), auc_point)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "auc_boot.npy"), auc_boot)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "ci_l.npy"), ci_l)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "ci_u.npy"), ci_u)
    
    df_roc.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "df_roc.csv"), index = False)
    df_roc_ci.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "df_roc_ci.csv"), index = False)

    feature_hospital = [
        "lab_result_num_val_718-7_Hemoglobin [Mass/volume] in Blood",
        "lab_result_num_val_6690-2_Leukocytes [#/volume] in Blood by Automated count",
        "lab_result_num_val_2345-7_Glucose [Mass/volume] in Serum or Plasma",
        "lab_result_num_val_777-3_Platelets [#/volume] in Blood by Automated count",
        "lab_result_num_val_3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma",
        "lab_result_num_val_2160-0_Creatinine [Mass/volume] in Serum or Plasma",
        "lab_result_num_val_2951-2_Sodium [Moles/volume] in Serum or Plasma",
        "lab_result_num_val_2823-3_Potassium [Moles/volume] in Serum or Plasma",
        "lab_result_num_val_2028-9_Carbon dioxide, total [Moles/volume] in Serum or Plasma",
        "age_diagnosis"
    ] 

    X_derivation_new_hospital = X_derivation_new[feature_hospital]
    X_validation_new_hospital = X_validation_new[feature_hospital]
    
    imp = IterativeImputer(max_iter=10, random_state=17)
    X_derivation_new_hospital_imputed = imp.fit_transform(X_derivation_new_hospital)
    X_validation_new_hospital_imputed = imp.transform(X_validation_new_hospital)
    
    y =  np.asarray(y_derivation_new, dtype=float)
    
    model = sm.Logit(y, X_derivation_new_hospital_imputed).fit()
    print(model.summary())

    model_n = "LR_Hospital_new"
    os.makedirs(os.path.join(path_newFinal, "ML_models", model_n), exist_ok = True)
    joblib.dump(model, os.path.join(path_newFinal, "ML_models", model_n, f"{model_n}_model.pkl"))
    
    y_pred = model.predict(X_validation_new_hospital_imputed)
    
    pd.DataFrame({"y_pred": y_pred}).to_csv(
        os.path.join(path_newFinal, "ML_models", model_n, "y_pred.csv"), index=False
    )
    
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"), y_pred)
    auc_point, auc_boot, ci_l, ci_u, df_roc, df_roc_ci  = bootstrap_auc_ci(y_validation_new, y_pred)
    
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "auc_boot.npy"), auc_point)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "auc_boot.npy"), auc_boot)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "ci_l.npy"), ci_l)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "ci_u.npy"), ci_u)
    
    df_roc.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "df_roc.csv"), index = False)
    df_roc_ci.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "df_roc_ci.csv"), index = False)

if __name__== "__main__":
  main()