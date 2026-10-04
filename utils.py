# written by Liangying, Sep 12 2026
#%%
import os
import pandas as pd
import IDckle
import dask.dataframe as dd
from plotnine import *
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

#%%----------------------------------------------------------------------------------------------------------------------------------

def Examine(cov_name):
    df_IDSurv_Visits_tmp = dd.read_parquet(os.path.join(path2, "Model_old_bk", f"df_{cov_name}_IDSurv_Visits_filtered"))
    df_IDSurv_Visits = df_IDSurv_Visits_tmp[df_IDSurv_Visits_tmp['patient_id'].isin(id_DA_final)].compute()

    df_IDSurv_CovFreq= df_IDSurv_Visits.groupby(['code', 'is_dead'])['patient_id'].nunique().reset_index()
    df_IDSurv_CovFreq.rename(columns = {'patient_id':'Freq'}, inplace = True)

    df_Termi_unique = df_Termi.drop_duplicates(subset = ['code', 'code_description'])
    df_IDSurv_CovFreq_Termi = pd.merge(df_IDSurv_CovFreq, df_Termi_unique, on = "code", how = "left").drop(columns = ['unit', 'path'])

    n_IDSurv_deceased = df_IDSurv_Visits[df_IDSurv_Visits['is_dead'] == 1]['patient_id'].nunique()
    n_IDSurv_censored = df_IDSurv_Visits[df_IDSurv_Visits['is_dead'] == 0]['patient_id'].nunique()
    n_IDSurv = n_IDSurv_deceased + n_IDSurv_censored

    df_IDSurv_CovFreq_Termi['NA_Prob'] = np.where(df_IDSurv_CovFreq_Termi['is_dead'] == 1, 
                                                (n_IDSurv_deceased - df_IDSurv_CovFreq_Termi['Freq']) / n_IDSurv_deceased,
                                                (n_IDSurv_censored - df_IDSurv_CovFreq_Termi['Freq']) / n_IDSurv_censored)
    return df_IDSurv_CovFreq_Termi


def df_cov_final(cov_name):
    df_tmp = pd.read_csv(os.path.join(path2,"Model_final","CovFreq", f"df_{cov_name}_IDSurv_CovFreq_Termi_filtered.csv"))
    df_cov = df_tmp.drop_duplicates(['code', 'code_description'])[['code', 'code_description']]
    df_cov['code_new'] = df_cov['code_description']

def SaveDf_FilterCov(cov_name):
    df_IDSurv_Visits_tmp = dd.read_parquet(os.path.join(path_old, f"df_{cov_name}_IDSurv_Visits_filtered"))
    df_ID_visits = df_IDSurv_Visits_tmp[df_IDSurv_Visits_tmp['patient_id'].isin(id_DA_final)] 

    df_ID_cov = pd.read_csv(os.path.join(path_model, "CovFreq", f"df_{cov_name}_final.csv"))
    df_ID_cov['code'] = df_ID_cov['code'].astype(str)
    df_ID_visits_FilterCov = df_ID_visits[df_ID_visits['code'].isin(df_ID_cov['code'].unique())]   

    if cov_name == "med_ingre":
        date_name = "start_date"
    else:
        date_name = "date"
    df_ID_visits_FilterCov['diagnosis_date'] = dd.to_datetime(df_ID_visits_FilterCov['diagnosis_date'], format = "%Y-%m-%d")
    df_ID_visits_FilterCov['visit_survt'] = (df_ID_visits_FilterCov[date_name] - df_ID_visits_FilterCov['diagnosis_date']).dt.days


def Select_cov():
    df_med_ingre_list  = []
    for med_class, list in med_ingre_dict.items():
        for name in list:
            med_name = name
            code = med_ingre[med_ingre['code_description'] == name]['code'].unique()[0]
            code_new = med_class
            df_med_ingre_list.append([med_name, code, code_new])
    
    df_med_ingre = pd.DataFrame(df_med_ingre_list, columns = ["med_name", "code", "code_new"])

def id_cov(cov_name):
    df_DA_tmp = dd.read_parquet(os.path.join(path2, "Model_old_bk", f"df_{cov_name}_IDSurv_Visits_filtered"))
    df_DA = df_DA_tmp[df_DA_tmp['patient_id'].isin(id_DA)]
    return df_DA['patient_id'].unique()


def Create_df_FE(cov_name, value_name):
    code_name = "code"
    df_ID_visits_FilterCov = dd.read_parquet(os.path.join(path_input, f"df_{cov_name}_IDSurv_visits_FilterCov"))
    df_ID_visits_FilterCov_Max = df_ID_visits_FilterCov.sort_values('visit_survt', ascending = False).drop_duplicates(['patient_id', code_name, 'ICD10_code'])
    
    df_code_count = df_ID_visits_FilterCov.groupby(['patient_id', code_name,'ICD10_code']).size().reset_index()
    df_code_count = df_code_count.rename(columns = {0: 'code_count'})
    df_ID_visits_FilterCov_Max_count = dd.merge(df_ID_visits_FilterCov_Max, df_code_count, on = ['patient_id', code_name, 'ICD10_code'], how = 'left')

    df_patient = dd.read_csv(os.path.join(path, "patient_ID.csv"),dtype = {'death_date_source_id': 'object'})
    df_patient_demo_tmp = df_patient[['patient_id', 'year_of_birth','sex','race', 'ethnicity', 'marital_status']].drop_duplicates() 
    df_patient_demo = df_patient_demo_tmp[df_patient_demo_tmp['patient_id'].isin(id_DA_final)]   

    df_cov_stats = df_ID_visits_FilterCov.groupby(['patient_id', 'code', 'ICD10_code']).agg(
                    mean = (value_name, 'mean'),
                    median = (value_name, 'median'),
                    std = (value_name, 'std')
    ).reset_index().compute()   

    df_ID_visits_FilterCov_Max_count_CovStats = dd.merge(df_ID_visits_FilterCov_Max_count, df_cov_stats, on = ['patient_id', 'code', 'ICD10_code'], how = 'left')
    df_ID_visits_FilterCov_Max_count_CovStats_demo = dd.merge(df_ID_visits_FilterCov_Max_count_CovStats, df_patient_demo, on = 'patient_id', how = "left")
    df_ID_visits_FilterCov_Max_count_CovStats_demo['year_of_birth'] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_CovStats_demo['year_of_birth'], format = "%Y")
    df_ID_visits_FilterCov_Max_count_CovStats_demo['diagnosis_date'] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_CovStats_demo['diagnosis_date'], format = "%Y-%m-%d")
    df_ID_visits_FilterCov_Max_count_CovStats_demo['last_date'] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_CovStats_demo['last_date'], format = "%Y-%m-%d")
    df_ID_visits_FilterCov_Max_count_CovStats_demo['date'] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_CovStats_demo['date'], format = "%Y-%m-%d")
    df_ID_visits_FilterCov_Max_count_CovStats_demo['age_diagnosis'] = (df_ID_visits_FilterCov_Max_count_CovStats_demo['diagnosis_date'] - df_ID_visits_FilterCov_Max_count_CovStats_demo['year_of_birth']).dt.days / 365
    df_ID_visits_FilterCov_Max_count_CovStats_demo['age_last'] = (df_ID_visits_FilterCov_Max_count_CovStats_demo['last_date'] - df_ID_visits_FilterCov_Max_count_CovStats_demo['year_of_birth']).dt.days / 365
    df_ID_visits_FilterCov_Max_count_CovStats_demo['age_code'] = (df_ID_visits_FilterCov_Max_count_CovStats_demo['date'] - df_ID_visits_FilterCov_Max_count_CovStats_demo['year_of_birth']).dt.days / 365

    print(df_ID_visits_FilterCov_Max_count.shape[0].compute(), '\n')
    print(df_ID_visits_FilterCov_Max_count_CovStats_demo.shape[0].compute(), '\n')


def FindMax_Count_demo(cov_name):
    df_ID_cov = pd.read_csv(os.path.join(path_model, "CovFreq", f"df_{cov_name}_final.csv"))
    df_ID_cov['code_old'] = df_ID_cov['code_old'].astype(str)
    df_ID_cov['code'] = df_ID_cov['code'].astype(str)

    df_ID_visits_FilterCov_tmp = dd.read_parquet(os.path.join(path_input, f"df_{cov_name}_IDSurv_visits_FilterCov"))
    df_ID_visits_FilterCov_tmp = df_ID_visits_FilterCov_tmp.rename(columns = {'code':'code_old'})  
    df_ID_visits_FilterCov = dd.merge(df_ID_visits_FilterCov_tmp, df_ID_cov, on = "code_old", how = "left")

    code_name = "code"
    df_ID_visits_FilterCov_Max = df_ID_visits_FilterCov.sort_values('visit_survt', ascending = False).drop_duplicates(['patient_id', code_name, 'ICD10_code'])
    
    df_code_count = df_ID_visits_FilterCov.groupby(['patient_id', code_name,'ICD10_code']).size().reset_index()
    df_code_count = df_code_count.rename(columns = {0: 'code_count'})
    df_ID_visits_FilterCov_Max_count = dd.merge(df_ID_visits_FilterCov_Max, df_code_count, on = ['patient_id', code_name, 'ICD10_code'], how = 'left')

    df_patient = dd.read_csv(os.path.join(path, "patient_ID.csv"),dtype = {'death_date_source_id': 'object'})
    df_patient_demo_tmp = df_patient[['patient_id', 'year_of_birth','sex','race', 'ethnicity', 'marital_status']].drop_duplicates() 
    df_patient_demo = df_patient_demo_tmp[df_patient_demo_tmp['patient_id'].isin(id_DA_final)]  

    df_ID_visits_FilterCov_Max_count_demo = dd.merge(df_ID_visits_FilterCov_Max_count, df_patient_demo, on = 'patient_id', how = "left")

    if cov_name == "med_ingre":
        date_name = "start_date"
    else:
        date_name = "date"
    
    df_ID_visits_FilterCov_Max_count_demo['year_of_birth'] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_demo['year_of_birth'], format = "%Y")
    df_ID_visits_FilterCov_Max_count_demo['diagnosis_date'] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_demo['diagnosis_date'], format = "%Y-%m-%d")
    df_ID_visits_FilterCov_Max_count_demo['last_date'] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_demo['last_date'], format = "%Y-%m-%d")
    df_ID_visits_FilterCov_Max_count_demo[date_name] = dd.to_datetime(df_ID_visits_FilterCov_Max_count_demo[date_name], format = "%Y-%m-%d")
    df_ID_visits_FilterCov_Max_count_demo['age_diagnosis'] = (df_ID_visits_FilterCov_Max_count_demo['diagnosis_date'] - df_ID_visits_FilterCov_Max_count_demo['year_of_birth']).dt.days / 365
    df_ID_visits_FilterCov_Max_count_demo['age_last'] = (df_ID_visits_FilterCov_Max_count_demo['last_date'] - df_ID_visits_FilterCov_Max_count_demo['year_of_birth']).dt.days / 365
    df_ID_visits_FilterCov_Max_count_demo['age_code'] = (df_ID_visits_FilterCov_Max_count_demo[date_name] - df_ID_visits_FilterCov_Max_count_demo['year_of_birth']).dt.days / 365
    print(df_ID_visits_FilterCov_Max.shape[0].compute(), '\n')
    print(df_ID_visits_FilterCov_Max_count_demo.shape[0].compute(), '\n')
  
def FilterID(df, id, cov_name, path_input):
    df_final = df[df['patient_id'].isin(id['patient_id'].unique())]
    path_final = os.path.join(path_input, "input_lab_vitalSigns_procedure_medication")
    if cov_name in ("lab", "vitalSigns"):
        p = os.path.join(path_final, f"df_{cov_name}_DA_visits_FilterCov_Max_count_CovStats_demo_final")
    elif cov_name in ("med_ingre", "procedure"):
        p = os.path.join(path_final, f"df_{cov_name}_DA_visits_FilterCov_Max_count_demo_final")
   
def age_dummy(X_imputed):
    bins = [0, 18, 29, 39, 49, 59, 69, 79, 120]  
    labels = ['<18', '18-29','30-39','40-49','50-59','60-69','70-79','80+']

    X_imputed['age_diagnosis_cat'] = pd.cut(X_imputed['age_diagnosis'], bins=bins, labels=labels, right=True)
    X_imputed['sex'] = X_imputed['sex'].map({'F':0, 'M':1})
    X_imputed_dummies = pd.get_dummies(X_imputed, columns = ['age_diagnosis_cat'], drop_first=False)
    X_imputed_dummies = X_imputed_dummies.drop(columns = "age_diagnosis_cat_40-49")   
    X_imputed_dummies = X_imputed_dummies.astype(float)   
    return X_imputed_dummies

def X_preproc(X_train, X_test, feature_inter, cols_con, cols_cat):
    X_train_reduced = X_train[feature_inter]
    X_test_reduced = X_test[feature_inter]
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


def Onc_DF(df, path, cov_name):
    df_visit_count = df.groupby(['patient_id','ICD10_code']).size().reset_index()
    df_visit_count = df_visit_count.rename(columns = {0: 'visit_count'})

    id_Group = df_Ref_simple[['patient_id','ICD10_code', 'is_dead']].drop_duplicates()[df_Ref_simple['patient_id'].isin(id_DA_final)]
    visit_counts_Group = pd.merge(df_visit_count, id_Group, how = "left", on = ['patient_id','ICD10_code'])

    id_missing = list(set(id_DA_final) - set(df['patient_id'].unique()))
    df_visit_counts_missing =  id_Group[id_Group['patient_id'].isin(id_missing)]
    df_visit_counts_missing['visit_count'] = 0

    df_visit_count_add = pd.concat([visit_counts_Group, df_visit_counts_missing], axis = 0)
    df_visit_count_add.to_csv(os.path.join(path, f"df_{cov_name}_visit_count_add.csv"), index=False)
    visit_counts_Group.to_csv(os.path.join(path, f"df_{cov_name}_visit_count.csv"), index=False)
    return visit_counts_Group, df_visit_count_add
