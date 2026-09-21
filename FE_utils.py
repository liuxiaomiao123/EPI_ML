# written by Liangying, 2025/05/18
#%%
import os
import pandas as pd
import pickle
import dask.dataframe as dd
from plotnine import *
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

#%%----------------------------------------------------------------------------------------------------------------------------------

def Examine_NAProb(cov_name):
    df_PISurv_Visits_tmp = dd.read_parquet(os.path.join(path2, "Model_old_bk", f"df_{cov_name}_PISurv_Visits_filtered"))
    df_PISurv_Visits = df_PISurv_Visits_tmp[df_PISurv_Visits_tmp['patient_id'].isin(id_IEI_final)].compute()

    df_PISurv_CovFreq= df_PISurv_Visits.groupby(['code', 'is_dead'])['patient_id'].nunique().reset_index()
    df_PISurv_CovFreq.rename(columns = {'patient_id':'Freq'}, inplace = True)

    df_Termi_unique = df_Termi.drop_duplicates(subset = ['code', 'code_description'])
    df_PISurv_CovFreq_Termi = pd.merge(df_PISurv_CovFreq, df_Termi_unique, on = "code", how = "left").drop(columns = ['unit', 'path'])

    n_PISurv_deceased = df_PISurv_Visits[df_PISurv_Visits['is_dead'] == 1]['patient_id'].nunique()
    n_PISurv_censored = df_PISurv_Visits[df_PISurv_Visits['is_dead'] == 0]['patient_id'].nunique()
    n_PISurv = n_PISurv_deceased + n_PISurv_censored

    df_PISurv_CovFreq_Termi['NA_Prob'] = np.where(df_PISurv_CovFreq_Termi['is_dead'] == 1, 
                                                (n_PISurv_deceased - df_PISurv_CovFreq_Termi['Freq']) / n_PISurv_deceased,
                                                (n_PISurv_censored - df_PISurv_CovFreq_Termi['Freq']) / n_PISurv_censored)
    df_PISurv_CovFreq_Termi.to_csv(os.path.join(path2,"Model_final","CovFreq", f"df_{cov_name}_PISurv_CovFreq_Termi.csv"), index = False)
    return df_PISurv_CovFreq_Termi


def df_cov_final(cov_name):
    df_tmp = pd.read_csv(os.path.join(path2,"Model_final","CovFreq", f"df_{cov_name}_PISurv_CovFreq_Termi_filtered.csv"))
    df_cov = df_tmp.drop_duplicates(['code', 'code_description'])[['code', 'code_description']]
    df_cov['code_new'] = df_cov['code_description']
    df_cov.to_csv(os.path.join(path2,"Model_final","CovFreq", f"df_{cov_name}_final.csv"), index = False)


def SaveDf_FilterCov(cov_name):
    df_PISurv_Visits_tmp = dd.read_parquet(os.path.join(path_old, f"df_{cov_name}_PISurv_Visits_filtered"))
    df_PI_visits = df_PISurv_Visits_tmp[df_PISurv_Visits_tmp['patient_id'].isin(id_IEI_final)] 

    df_PI_cov = pd.read_csv(os.path.join(path_model, "CovFreq", f"df_{cov_name}_final.csv"))
    df_PI_cov['code'] = df_PI_cov['code'].astype(str)
    df_PI_visits_FilterCov = df_PI_visits[df_PI_visits['code'].isin(df_PI_cov['code'].unique())]   

    if cov_name == "med_ingre":
        date_name = "start_date"
    else:
        date_name = "date"
    df_PI_visits_FilterCov['diagnosis_date'] = dd.to_datetime(df_PI_visits_FilterCov['diagnosis_date'], format = "%Y-%m-%d")
    df_PI_visits_FilterCov['visit_survt'] = (df_PI_visits_FilterCov[date_name] - df_PI_visits_FilterCov['diagnosis_date']).dt.days

    p = os.path.join(path_input, f"df_{cov_name}_PISurv_visits_FilterCov")
    os.makedirs(p, exist_ok = True)
    df_PI_visits_FilterCov.to_parquet(p, engine = "pyarrow", compression = 'snappy', write_index = False )


def Select_cov():
    opioid_list = ['fentanyl','morphine','hydromorphone','oxycodone']
    sedative_list = ['midazolam','lorazepam']
    anesthetics_list = ['propofol'] 
    vasopressor_list = ['epinephrine']
    diuretic_list = ['furosemide']
    steriod_list = ['prednisone','methylprednisolone','dexamethasone','hydrocortisone']
    antibiotic_broad_list = ['vancomycin']
    
    med_ingre_dict = {
        "opioid" : opioid_list,
        "sedative" : sedative_list,
        "anesthetics" : anesthetics_list,
        "vasopressor" : vasopressor_list,
        "diuretic" : diuretic_list,
        "steriod" : steriod_list,
        "antibiotic" : antibiotic_broad_list
    }
    
    df_med_ingre_list  = []
    for med_class, list in med_ingre_dict.items():
        for name in list:
            med_name = name
            code = med_ingre[med_ingre['code_description'] == name]['code'].unique()[0]
            code_new = med_class
            df_med_ingre_list.append([med_name, code, code_new])
    
    df_med_ingre = pd.DataFrame(df_med_ingre_list, columns = ["med_name", "code", "code_new"])
    df_med_ingre.to_csv(os.path.join(path2,"Model_final","CovFreq", f"df_med_ingre_final.csv"), index = False)


def id_cov(cov_name):
    df_IEI_tmp = dd.read_parquet(os.path.join(path2, "Model_old_bk", f"df_{cov_name}_PISurv_Visits_filtered"))
    df_IEI = df_IEI_tmp[df_IEI_tmp['patient_id'].isin(id_IEI)]
    return df_IEI['patient_id'].unique()


def Create_df_FE(cov_name, value_name):
    code_name = "code"
    df_PI_visits_FilterCov = dd.read_parquet(os.path.join(path_input, f"df_{cov_name}_PISurv_visits_FilterCov"))
    df_PI_visits_FilterCov_Max = df_PI_visits_FilterCov.sort_values('visit_survt', ascending = False).drop_duplicates(['patient_id', code_name, 'ICD10_code'])
    
    df_code_count = df_PI_visits_FilterCov.groupby(['patient_id', code_name,'ICD10_code']).size().reset_index()
    df_code_count = df_code_count.rename(columns = {0: 'code_count'})
    df_PI_visits_FilterCov_Max_count = dd.merge(df_PI_visits_FilterCov_Max, df_code_count, on = ['patient_id', code_name, 'ICD10_code'], how = 'left')

    df_patient = dd.read_csv(os.path.join(path, "patient_PI.csv"),dtype = {'death_date_source_id': 'object'})
    df_patient_demo_tmp = df_patient[['patient_id', 'year_of_birth','sex','race', 'ethnicity', 'marital_status']].drop_duplicates() 
    df_patient_demo = df_patient_demo_tmp[df_patient_demo_tmp['patient_id'].isin(id_IEI_final)]   

    df_cov_stats = df_PI_visits_FilterCov.groupby(['patient_id', 'code', 'ICD10_code']).agg(
                    mean = (value_name, 'mean'),
                    median = (value_name, 'median'),
                    std = (value_name, 'std')
    ).reset_index().compute()   

    df_PI_visits_FilterCov_Max_count_CovStats = dd.merge(df_PI_visits_FilterCov_Max_count, df_cov_stats, on = ['patient_id', 'code', 'ICD10_code'], how = 'left')
    df_PI_visits_FilterCov_Max_count_CovStats_demo = dd.merge(df_PI_visits_FilterCov_Max_count_CovStats, df_patient_demo, on = 'patient_id', how = "left")

    df_PI_visits_FilterCov_Max_count_CovStats_demo['year_of_birth'] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_CovStats_demo['year_of_birth'], format = "%Y")
    df_PI_visits_FilterCov_Max_count_CovStats_demo['diagnosis_date'] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_CovStats_demo['diagnosis_date'], format = "%Y-%m-%d")
    df_PI_visits_FilterCov_Max_count_CovStats_demo['last_date'] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_CovStats_demo['last_date'], format = "%Y-%m-%d")
    df_PI_visits_FilterCov_Max_count_CovStats_demo['date'] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_CovStats_demo['date'], format = "%Y-%m-%d")

    df_PI_visits_FilterCov_Max_count_CovStats_demo['age_diagnosis'] = (df_PI_visits_FilterCov_Max_count_CovStats_demo['diagnosis_date'] - df_PI_visits_FilterCov_Max_count_CovStats_demo['year_of_birth']).dt.days / 365
    df_PI_visits_FilterCov_Max_count_CovStats_demo['age_last'] = (df_PI_visits_FilterCov_Max_count_CovStats_demo['last_date'] - df_PI_visits_FilterCov_Max_count_CovStats_demo['year_of_birth']).dt.days / 365
    df_PI_visits_FilterCov_Max_count_CovStats_demo['age_code'] = (df_PI_visits_FilterCov_Max_count_CovStats_demo['date'] - df_PI_visits_FilterCov_Max_count_CovStats_demo['year_of_birth']).dt.days / 365

    print(df_PI_visits_FilterCov_Max_count.shape[0].compute(), '\n')
    print(df_PI_visits_FilterCov_Max_count_CovStats_demo.shape[0].compute(), '\n')

    p = os.path.join(path_input,"df_Max_stats", f"df_{cov_name}_PI_visits_FilterCov_Max_count_CovStats_demo")
    os.makedirs(p, exist_ok = True)     
    df_PI_visits_FilterCov_Max_count_CovStats_demo.to_parquet(p, engine = "pyarrow", compression = 'snappy', write_index = False)


def FindMax_Count_demo(cov_name):
    df_PI_cov = pd.read_csv(os.path.join(path_model, "CovFreq", f"df_{cov_name}_final.csv"))
    df_PI_cov['code_old'] = df_PI_cov['code_old'].astype(str)
    df_PI_cov['code'] = df_PI_cov['code'].astype(str)

    df_PI_visits_FilterCov_tmp = dd.read_parquet(os.path.join(path_input, f"df_{cov_name}_PISurv_visits_FilterCov"))
    df_PI_visits_FilterCov_tmp = df_PI_visits_FilterCov_tmp.rename(columns = {'code':'code_old'})  
    df_PI_visits_FilterCov = dd.merge(df_PI_visits_FilterCov_tmp, df_PI_cov, on = "code_old", how = "left")

    code_name = "code"
    df_PI_visits_FilterCov_Max = df_PI_visits_FilterCov.sort_values('visit_survt', ascending = False).drop_duplicates(['patient_id', code_name, 'ICD10_code'])
    
    df_code_count = df_PI_visits_FilterCov.groupby(['patient_id', code_name,'ICD10_code']).size().reset_index()
    df_code_count = df_code_count.rename(columns = {0: 'code_count'})
    df_PI_visits_FilterCov_Max_count = dd.merge(df_PI_visits_FilterCov_Max, df_code_count, on = ['patient_id', code_name, 'ICD10_code'], how = 'left')

    df_patient = dd.read_csv(os.path.join(path, "patient_PI.csv"),dtype = {'death_date_source_id': 'object'})
    df_patient_demo_tmp = df_patient[['patient_id', 'year_of_birth','sex','race', 'ethnicity', 'marital_status']].drop_duplicates() 
    df_patient_demo = df_patient_demo_tmp[df_patient_demo_tmp['patient_id'].isin(id_IEI_final)]  

    df_PI_visits_FilterCov_Max_count_demo = dd.merge(df_PI_visits_FilterCov_Max_count, df_patient_demo, on = 'patient_id', how = "left")

    if cov_name == "med_ingre":
        date_name = "start_date"
    else:
        date_name = "date"
    
    df_PI_visits_FilterCov_Max_count_demo['year_of_birth'] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_demo['year_of_birth'], format = "%Y")
    df_PI_visits_FilterCov_Max_count_demo['diagnosis_date'] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_demo['diagnosis_date'], format = "%Y-%m-%d")
    df_PI_visits_FilterCov_Max_count_demo['last_date'] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_demo['last_date'], format = "%Y-%m-%d")
    df_PI_visits_FilterCov_Max_count_demo[date_name] = dd.to_datetime(df_PI_visits_FilterCov_Max_count_demo[date_name], format = "%Y-%m-%d")

    df_PI_visits_FilterCov_Max_count_demo['age_diagnosis'] = (df_PI_visits_FilterCov_Max_count_demo['diagnosis_date'] - df_PI_visits_FilterCov_Max_count_demo['year_of_birth']).dt.days / 365
    df_PI_visits_FilterCov_Max_count_demo['age_last'] = (df_PI_visits_FilterCov_Max_count_demo['last_date'] - df_PI_visits_FilterCov_Max_count_demo['year_of_birth']).dt.days / 365
    df_PI_visits_FilterCov_Max_count_demo['age_code'] = (df_PI_visits_FilterCov_Max_count_demo[date_name] - df_PI_visits_FilterCov_Max_count_demo['year_of_birth']).dt.days / 365

    print(df_PI_visits_FilterCov_Max.shape[0].compute(), '\n')
    print(df_PI_visits_FilterCov_Max_count_demo.shape[0].compute(), '\n')

    p = os.path.join(path_input, "df_Max_stats", f"df_{cov_name}_PI_visits_FilterCov_Max_count_demo")
    os.makedirs(p, exist_ok = True)     
    df_PI_visits_FilterCov_Max_count_demo.to_parquet(p, engine = "pyarrow", compression = 'snappy', write_index = False)


def main():
    cov_list = ['lab', 'procedure', 'vitalSigns']  
    for cov_name in cov_list:
        Examine_NAProb(cov_name)
        
    cov_name = "vitalSigns"
    df_PISurv_CovFreq_Termi = pd.read_csv(os.path.join(path2, "Model_final", "CovFreq", f"df_{cov_name}_PISurv_CovFreq_Termi.csv"))
    df_PISurv_CovFreq_Termi_filtered = df_PISurv_CovFreq_Termi[df_PISurv_CovFreq_Termi['NA_Prob'] <= thre]
    print(df_PISurv_CovFreq_Termi_filtered['code_description'].unique())
    n_cov = df_PISurv_CovFreq_Termi_filtered['code'].nunique()
    print(f"\n the total amount of variables is: {n_cov}")
    df_PISurv_CovFreq_Termi_filtered.to_csv(os.path.join(path2,"Model_final","CovFreq", f"df_{cov_name}_PISurv_CovFreq_Termi_filtered.csv"), index = False)
        
    cov_list = ['lab', 'procedure', 'vitalSigns', 'med_ingre']
    for cov in cov_list:
        SaveDf_FilterCov(cov)
        
    id_lab = id_cov("lab")
    id_vitalSigns = id_cov("vitalSigns")
    id_procedure = id_cov("procedure")
    id_med_ingre = id_cov("med_ingre")
    
    id_lab_vitalSigns = list(set(id_lab.compute()) & set(id_vitalSigns.compute()))
    n_lab_vitalSigns = len(id_lab_vitalSigns)
    
    id_lab_vitalSigns_procedure_med = list(set(id_lab.compute()) & set(id_vitalSigns.compute()) & set(id_procedure.compute()) & set(id_med_ingre.compute()))
    n_lab_vitalSigns_procedure_med = len(id_lab_vitalSigns_procedure_med)
    
    cov_name = "lab"
    value_name = "lab_result_num_val"
    Create_df_FE(cov_name, value_name)
    
    cov_name = "vitalSigns"
    value_name = "value"
    Create_df_FE(cov_name, value_name)
    
    cov_name = "procedure"
    FindMax_Count_demo(cov_name)

    cov_name = "med_ingre"
    FindMax_Count_demo(cov_name)
    

if __name__== "__main__":
    main()




