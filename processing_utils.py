# written by Liangying, 2025/06/04
#%%

import os
import pandas as pd
import pickle
import dask.dataframe as dd
from plotnine import *
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

def FilterID(df, id, cov_name, path_input):
    df_final = df[df['patient_id'].isin(id['patient_id'].unique())]
    
    path_final = os.path.join(path_input, "input_lab_vitalSigns_procedure_medication")
    if cov_name in ("lab", "vitalSigns"):
        p = os.path.join(path_final, f"df_{cov_name}_PI_visits_FilterCov_Max_count_CovStats_demo_final")
    elif cov_name in ("med_ingre", "procedure"):
        p = os.path.join(path_final, f"df_{cov_name}_PI_visits_FilterCov_Max_count_demo_final")

    os.makedirs(p, exist_ok = True)
    df_final.to_parquet(p, engine = "pyarrow", compression = 'snappy', write_index = False)


def Oncology_DF(df, path, cov_name):
    df_visit_count = df.groupby(['patient_id','ICD10_code']).size().reset_index()
    df_visit_count = df_visit_count.rename(columns = {0: 'visit_count'})

    id_Group = df_Ref_simple[['patient_id','ICD10_code', 'is_dead']].drop_duplicates()[df_Ref_simple['patient_id'].isin(id_IEI_final)]
    visit_counts_Group = pd.merge(df_visit_count, id_Group, how = "left", on = ['patient_id','ICD10_code'])

    id_missing = list(set(id_IEI_final) - set(df['patient_id'].unique()))
    df_visit_counts_missing =  id_Group[id_Group['patient_id'].isin(id_missing)]
    df_visit_counts_missing['visit_count'] = 0

    df_visit_count_add = pd.concat([visit_counts_Group, df_visit_counts_missing], axis = 0)
    df_visit_count_add.to_csv(os.path.join(path, f"df_{cov_name}_visit_count_add.csv"), index=False)
    visit_counts_Group.to_csv(os.path.join(path, f"df_{cov_name}_visit_count.csv"), index=False)

    return visit_counts_Group, df_visit_count_add


def read_df(cov_name, path_final):
    if cov_name in ("lab", "vitalSigns"):
        df = dd.read_parquet(os.path.join(path_final, f"df_{cov_name}_PI_visits_FilterCov_Max_count_CovStats_demo_final")).compute()
    elif cov_name in ("med_ingre", "procedure"):
        df = dd.read_parquet(os.path.join(path_final, f"df_{cov_name}_PI_visits_FilterCov_Max_count_demo_final")).compute()
    else:
        df = pd.read_csv(os.path.join(path_final, f"df_{cov_name}_visit_count.csv"))
    return df


def long_to_wide(df_final_select, col, index_col, columns_col, values_col):
    
    df_final_select[col] = df_final_select[col].replace(9999999.0, np.nan)
    df_final_select[col] = df_final_select[col].clip(lower=0)

    df_long_list = []
    for cl in values_col:
        df_long = df_final_select.pivot_table(index = index_col, columns = columns_col, values = [cl], fill_value=None).reset_index()
        df_long_list.append(df_long)
    df_long_table = df_long_list[0]

    for df in df_long_list[1:]:
        df_long_table = df_long_table.merge(df, on = index_col, how = "outer")

    return df_long_table


def main():
    cov_name = "chemo"
    df_visit_count_chemo, df_visit_count_add_chemo = Oncology_DF(df_chemo_PI_Ref, path_final, cov_name)   
    cov_name = "lab"
    df_lab_final = read_df(cov_name, path_final)
    cov_name = "vitalSigns"
    df_vitalSigns_final = read_df(cov_name, path_final)
    cov_name = "med_ingre"
    df_med_ingre_final = read_df(cov_name, path_final)
    cov_name = "procedure"
    df_procedure_final = read_df(cov_name, path_final)
    cov_name = "chemo"
    df_chemo_final = read_df(cov_name, path_final)
    
    cov_select = ['patient_id', 'ICD10_code', 'is_dead',
                  'code', 'lab_result_num_val', 'code_count','median',
                  'age_diagnosis', 
                  'sex', 'race', 'ethnicity']

    df_final = df_lab_final.copy()
    df_final_select = df_final[cov_select]
    col = ['lab_result_num_val', 'median']    
    index_col = ['patient_id', 'ICD10_code','is_dead']
    columns_col = ['code']
    values_col = ['lab_result_num_val', 'median', 'code_count']
    df_long_lab_table = long_to_wide(df_final_select, col, index_col, columns_col, values_col)
    
    df_final = df_vitalSigns_final.copy()
    df_final_select = df_final[cov_select]
    col = ['value', 'median']  
    values_col = ['value', 'median', 'code_count']
    df_long_vitalSigns_table = long_to_wide(df_final_select, col, index_col, columns_col, values_col)
    
    df_final = df_med_ingre_final.copy()
    df_final_select = df_final[cov_select]
    col = []    
    columns_col = ['code']
    values_col = ['code_count']
    df_long_med_ingre_table = long_to_wide(df_final_select, col, index_col, columns_col, values_col)
    
    cov_global = ['patient_id', 'ICD10_code',  'code_count_total', 'ingre_counts']
    df_global = df_med_ingre_final[cov_global].drop_duplicates()
    df_global.columns = pd.MultiIndex.from_tuples([
        ('patient_id', ''),
        ('ICD10_code', ''),
        ('code_count_total', 'med_ingre'),
        ('ingre_counts', 'med_ingre'),
    ])
    
    df_long_med_ingre_table = df_long_med_ingre_table.merge(df_global, on = [('patient_id', ''), ('ICD10_code', '')], how = "left")

    df_final = df_procedure_final.copy()
    df_final_select = df_final[cov_select]
    col = []   
    columns_col = ['code']
    values_col = ['code_count']
    df_long_procedure_table = long_to_wide(df_final_select, col, index_col, columns_col, values_col)

    cov_global = ['patient_id', 'ICD10_code',  'code_count_total']
    df_global = df_procedure_final[cov_global].drop_duplicates()
    df_global.columns = pd.MultiIndex.from_tuples([
        ('patient_id', ''),
        ('ICD10_code', ''),
        ('code_count_total', 'procedure'),
    ])
    
    df_global = df_global.reset_index() 
    df_long_procedure_table = df_long_procedure_table.merge(df_global, on = [('patient_id', ''), ('ICD10_code', '')], how = "left").drop(columns=[('index', '')]) 

    cov_select = ['patient_id', 'ICD10_code', 'is_dead',
                  'visit_count']
    
    df_chemo_table = df_chemo_final.rename(columns = {'visit_count': 'chemo_visit_count'})
    df_chemo_table.columns = pd.MultiIndex.from_tuples([
        ('patient_id', ''),
        ('ICD10_code', ''),
        ('chemo_visit_count', ''),
        ('is_dead', ''),
    ])
    
    df_chemo_table = df_chemo_table.reset_index()
    col_static = ['sex','race', 'ethnicity', 'age_diagnosis']
    df_static = df_lab_final[['patient_id', 'ICD10_code'] + col_static].drop_duplicates()
    
    df_static.columns = pd.MultiIndex.from_tuples([
        ('patient_id', ''),
        ('ICD10_code', ''),
        ('sex', ''),
        ('race', ''),
        ('ethnicity', ''),
        ('age_diagnosis', '')
    ])
    df_static = df_static.reset_index()

if __name__== "__main__":
    main()


