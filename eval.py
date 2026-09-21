# written by Liangying, 2025/08/18

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
from sklearn.calibration import CalibratedClassifierCV

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

#%%

def classification_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan
    }
    return metrics


def metrics_eval_input(model_n):
    y_validation = np.load(os.path.join(path_newFinal, "ML_models", "y_validation.npy"))
    y_pred_validation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"))
    y_derivation = np.load(os.path.join(path_newFinal, "ML_models", "y_derivation.npy"))

    X_derivation = pd.read_csv(os.path.join(path_newFinal, "ML_models", "X_derivation_new.csv")) 

    if model_n in ["IMRS_new", "LR_Hospital_new"]:
        model = joblib.load(os.path.join(path_newFinal, "ML_models", model_n, f"{model_n}_model.pkl"))
        if model_n == "IMRS_new":
            X_derivation_new_preproc = pd.read_csv(os.path.join(path_newFinal, "ML_models", "IMRS_new", "X_derivation_new_preproc.csv"))
            y_pred_derivation = model.predict(X_derivation_new_preproc)
        else:
            X_derivation_new_preproc = pd.read_csv(os.path.join(path_newFinal, "ML_models", "LR_Hospital_new", "X_derivation_new_hospital_imputed.csv"))
            y_pred_derivation = model.predict(X_derivation_new_preproc)
    else:
        grid = joblib.load(os.path.join(path_newFinal, "ML_models", model_n, f"{model_n}_gridsearch_model_new.pkl"))
        y_pred_derivation = grid.best_estimator_.predict_proba(X_derivation)[:,1]
    
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "y_pred_derivation.npy"), y_pred_derivation)
    return y_derivation, y_pred_derivation, y_validation, y_pred_validation


def metrics_eval(y_derivation, y_pred_derivation, y_validation, y_pred_validation):
    fpr, tpr, thresholds = roc_curve(y_derivation, y_pred_derivation)
    youden = tpr - fpr

    best_idx = np.argmax(youden)
    best_cutoff = thresholds[best_idx]

    y_derivation_pred_label = (y_pred_derivation >= best_cutoff).astype(int)
    y_validation_pred_label = (y_pred_validation >= best_cutoff).astype(int)

    metrics_derivation = classification_metrics(
        y_derivation, y_derivation_pred_label
    )

    metrics_validation = classification_metrics(
        y_validation, y_validation_pred_label
    )
    return metrics_derivation, metrics_validation


def bootstrap_auc_ci(y_test, y_pred, n_boot = 1000, ci = 0.95, random_state = 45):
    y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
    rng = np.random.default_rng(random_state)

    idx_deceased, idx_censored = np.where(y_test == 1)[0], np.where(y_test == 0)[0]

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    tpr_boot = []

    auc_point = roc_auc_score(y_test, y_pred)
    auc_boot = []
    for _ in range(n_boot):

        idx_deceased_b = resample(idx_deceased, replace=True, random_state=rng.integers(1e9))
        idx_censored_b = resample(idx_censored, replace=True, random_state=rng.integers(1e9))
        idx_b = np.concatenate([idx_deceased_b, idx_censored_b])
        y_test_b = y_test[idx_b]
        y_pred_b = y_pred[idx_b]

        fpr_b, tpr_b, _ = roc_curve(y_test_b, y_pred_b)
        fpr_grid = np.linspace(0,1,100)
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tpr_boot.append(tpr_interp)

        auc_boot.append(roc_auc_score(y_test_b, y_pred_b))

    tpr_boot = np.array(tpr_boot)
    tpr_lower = np.percentile(tpr_boot, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_boot, 97.5, axis=0)
    fpr_grid = np.linspace(0,1,100)

    df_roc_ci = pd.DataFrame({
        'fpr': fpr_grid,
        'tpr_lower': tpr_lower,
        'tpr_upper': tpr_upper,
        'tpr_mean': np.mean(tpr_boot, axis=0)
    })
    auc_boot = np.array(auc_boot)
    
    alpha = (1-ci)/2
    ci_l, ci_u = np.quantile(auc_boot, [alpha, 1-alpha])

    return auc_point, auc_boot, ci_l, ci_u, df_roc, df_roc_ci


def cali(model_n):
    y_validation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_validation.npy"))
    y_pred_validation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"))
    y_derivation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_derivation.npy"))
    grid = joblib.load(os.path.join(path_newFinal, "ML_models", model_n, f"{model_n}_gridsearch_model_new.pkl"))
    best_model = grid.best_estimator_
    y_pred_derivation = grid.best_estimator_.predict_proba(X_derivation)[:,1]
    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_pred_derivation, y_derivation)  
    y_pred_validation_cal = iso.transform(y_pred_validation)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "y_pred_validation_cal.npy"), y_pred_validation_cal)
    
    df_eval = pd.DataFrame(
        {
            "y_pred":y_pred_validation,
            "y_cal": y_pred_validation_cal,
            "y_test":y_validation
        }
    )
    df_eval.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "df_eval.csv"), index = False)
    brier_validation = brier_score_loss(y_validation, y_pred_validation_cal)
    
def sub_eval(sub_name, X_test_original, X_test_original_long, path_final, model_n):
    if sub_name == "ICD10_code":
        df_result = X_test_original_long.dropna(subset = [sub_name]).groupby(sub_name).apply(lambda x: bootstrap_auc_ci(x['y_test'], x['y_pred'])).reset_index(name = 'func')
    else:
        df_result = X_test_original.dropna(subset = [sub_name]).groupby(sub_name).apply(lambda x: bootstrap_auc_ci(x['y_test'], x['y_pred'])).reset_index(name = 'func')

    result_func_list = df_result['func'].tolist()
    df_sub = pd.DataFrame(result_func_list, columns = ["auc_point", "auc_boot", "ci_l", "ci_u", "df_roc", "df_roc_ci"])
    df_sub[sub_name] = df_result[sub_name]

    df_sub.to_csv(os.path.join(path_final, "ML_models", model_n, f"df_sub_{sub_name}.csv"), index = False)


def main(model_n):
    y_validation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_validation.npy"))
    y_pred_derivation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred_derivation.npy"))
    y_pred_validation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"))
    y_derivation = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_derivation.npy"))
    cali(model_n)
    y_pred_derivation_cal = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred_derivation_cal.npy"))
    y_pred_validation_cal = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred_validation_cal.npy"))
    
    y_pred_all = np.concatenate([y_pred_derivation, y_pred_validation], axis=0)
    y_pred_all_cal = np.concatenate([y_pred_derivation_cal, y_pred_validation_cal], axis=0)
    y_all = np.concatenate([y_derivation, y_validation], axis=0)
    
    X_all = pd.concat([X_derivation, X_validation], axis=0, ignore_index=True)
    X_all['y_pred'] = y_pred_all
    X_all['y_cali'] = y_pred_all_cal
    X_all['is_dead'] = y_all
    
    icd_cols = [col for col in X_all.columns if col.startswith("D")]
    id_cols = [col for col in X_all.columns if col not in icd_cols]
    X_all_long = X_all.melt(id_vars = id_cols, value_vars = icd_cols,
                       var_name="ICD10_code", value_name="has_ICD")
    X_all_long = X_all_long[X_all_long["has_ICD"] == 1].drop("has_ICD", axis=1).reset_index(drop=True)
    df_y_pred = X_all_long[['ICD10_code', 'y_pred', 'y_cali', 'is_dead']]
    y_pred = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"))
    auc_point, auc_boot, ci_l, ci_u, df_roc, df_roc_ci  = bootstrap_auc_ci(y_test, y_pred)

    np.save(os.path.join(path_newFinal, "ML_models", model_n, "auc_point.npy"), auc_point)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "auc_boot.npy"), auc_boot)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "ci_l.npy"), ci_l)
    np.save(os.path.join(path_newFinal, "ML_models", model_n, "ci_u.npy"), ci_u)
    df_roc.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "df_roc.csv"), index = False)
    df_roc_ci.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "df_roc_ci.csv"), index = False)
    
    y_derivation_model, y_pred_derivation_model, y_validation_model, y_pred_validation_model = metrics_eval_input(model_n)
    metrics_derivation_model, metrics_validation_model = metrics_eval(y_derivation_model, y_pred_derivation_model, y_validation_model, y_pred_validation_model)

    X_test_original = pd.read_csv(os.path.join(path_newFinal, "ML_models", "X_test_original.csv")) 
    y_pred = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"))
    y_test = np.load(os.path.join(path_newFinal, "ML_models", "y_validation_new.npy"))
    X_test_original['y_test'] = y_test
    X_test_original['y_pred'] = y_pred
    
    sub_name = ['ICD10_code', 'ethnicity', 'sex', 'race'] 
    icd_cols = [col for col in X_test_original.columns if col.startswith("D")]
    id_cols = [col for col in X_test_original.columns if col not in icd_cols]
    
    X_test_original_long = X_test_original.melt(id_vars = id_cols, value_vars = icd_cols,
                       var_name="ICD10_code", value_name="has_ICD")
    X_test_original_long = X_test_original_long[X_test_original_long["has_ICD"] == 1].drop("has_ICD", axis=1).reset_index(drop=True)
    for sub in sub_name:
        sub_eval(sub, X_test_original, X_test_original_long, path_newFinal, model_n)

    X_test_age = pd.read_csv(os.path.join(path_newFinal, "ML_models", "X_validation_new.csv")) 
    y_test = np.load(os.path.join(path_newFinal, "ML_models", "y_validation.npy"))
    y_pred = np.load(os.path.join(path_newFinal, "ML_models", model_n, "y_pred.npy"))
    X_test_age['y_test'] = y_test
    X_test_age['y_pred'] = y_pred

    n_bin = 10
    X_test_age['age_bin'] = pd.qcut(X_test_age['age_diagnosis'], n_bin, duplicates="drop")
    X_test_age.to_csv(os.path.join(path_newFinal, "ML_models", model_n, "X_test_age.csv"),index = False)
    sub_name = ['age_bin']
    for sub in sub_name:
        sub_eval(sub, X_test_age, path_newFinal, model_n)


if __name__== "__main__":
  main(model_n)

