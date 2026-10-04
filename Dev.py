# -*- coding: utf-8 -*-
"""
Created on Oct 1st 2026

@author: liangyingliu
"""

import pandas as pd

def encode_percentiles(pro):
    return pd.qcut(pro, q=[0, 0.25, 0.75, 1], labels=['low', 'medium', 'high'])


def create_GH2(tx, row):
    q_patient = "MERGE (p:Patient {id: $patient_id, age: $patient_age, gender: $patient_gender})"
    tx.run(q_patient, patient_id = row['Patient_ID'], patient_age = row['Age'], patient_gender = row['Gender'])
    
    q_pro1 = "MERGE (pro1: pro1 {name: $pro1_name, level: $pro1_level})"
    q_pro2 = "MERGE (pro2: pro2 {name: $pro2_name, level: $pro2_level})"
    q_pro3 = "MERGE (pro3: pro3 {name: $pro3_name, level: $pro3_level})"
    tx.run(q_pro1, pro1_name = 'pro1', pro1_level = row['pro1_category'])
    tx.run(q_pro2, pro2_name = 'pro2', pro2_level = row['pro2_category'])
    tx.run(q_pro3, pro3_name = 'pro3', pro3_level = row['pro3_category'])
    
    q_tumor = "MERGE (t:Tumor {stage: $stage})"
    tx.run(q_tumor, {"stage": row['Tumour_Stage']})
    
    q_histology = "MERGE (h:Histology {type: $type})"
    tx.run(q_histology, {"type": row['Histology']})
    
    q_ER = "MERGE (er:ER {status: $status_ER})"
    tx.run(q_ER, {"status_ER": row['ER status']})
    
    q_PR = "MERGE (pr:PR {status: $status_PR})"
    tx.run(q_PR, {"status_PR": row['PR status']})
    
    q_Caq2 = "MERGE (Caq2:Caq2 {status: $status_Caq2})"
    tx.run(q_Caq2, {"status_Caq2": row['Caq2 status']})
    
    q_patient_pro = """ MERGE (p) -[:KEA_pro]-> (pro1)
                            MERGE (p) -[:KEA_pro]-> (pro2)
                            MERGE (p) -[:KEA_pro]-> (pro3)"""
    tx.run(q_patient_pro)
                            
    q_patient_tumor = "MERGE (p) -[:KEA_TUMOR_STAGE]-> (t)"
    tx.run(q_patient_tumor)
    
    q_patient_histology = "MERGE (p) -[:KEA_HISTOLOGY]-> (h)"
    tx.run(q_patient_histology)
    
    q_patient_ER = "MERGE (p) -[:KEA_ER_STATUS]-> (er)"
    q_patient_PR = "MERGE (p) -[:KEA_PR_STATUS]-> (pr)"
    q_patient_Caq2 = "MERGE (p) -[:KEA_Caq2_STATUS]-> (Caq2)"
    tx.run(q_patient_ER)
    tx.run(q_patient_PR)
    tx.run(q_patient_Caq2)
    
    
    q_tumor_pro = """ MERGE (t) -[:KEA_pro]-> (pro1)
                          MERGE (t) -[:KEA_pro]-> (pro2)
                          MERGE (t) -[:KEA_pro]-> (pro3)"""
    tx.run(q_tumor_pro)        
                          
    q_tumor_histology = "MERGE (t) -[:KEA_HISTOLOGY]-> (h)"
    tx.run(q_tumor_histology) 
    
    q_tumor_ER = "MERGE (t) -[:KEA_ER_STATUS]-> (er)"
    q_tumor_PR = "MERGE (t) -[:KEA_PR_STATUS]-> (pr)"
    q_tumor_Caq2 = "MERGE (t) -[:KEA_Caq2_STATUS]-> (Caq2)"
    tx.run(q_tumor_ER)
    tx.run(q_tumor_PR)
    tx.run(q_tumor_Caq2)
    

    q_histology_ER = "MERGE (h) -[:KEA_ER_STATUS]-> (er)"
    q_histology_PR = "MERGE (h) -[:KEA_PR_STATUS]-> (pr)"
    q_histology_Caq2 = "MERGE (h) -[:KEA_Caq2_STATUS]-> (Caq2)"
    tx.run(q_histology_ER)
    tx.run(q_histology_PR)
    tx.run(q_histology_Caq2)
    

#%%  

def create_GH(tx, row):
    q = """ MERGE (p:Patient {id: $patient_id, age: $patient_age, gender: $patient_gender})
            MERGE (pro1: pro1 {name: $pro1_name, level: $pro1_level})
            MERGE (pro2: pro2 {name: $pro2_name, level: $pro2_level})
            MERGE (pro3: pro3 {name: $pro3_name, level: $pro3_level})
            
            MERGE (t:Tumor {stage: $stage})
            
            MERGE (h:Histology {type: $type})
            
            MERGE (er:ER {status: $status_ER})
            MERGE (pr:PR {status: $status_PR})
            MERGE (Caq2:Caq2 {status: $status_Caq2})
            
            MERGE (p) -[:KEA_pro]-> (pro1)
            MERGE (p) -[:KEA_pro]-> (pro2)
            MERGE (p) -[:KEA_pro]-> (pro3)
            
            MERGE (p) -[:KEA_TUMOR_STAGE]-> (t)
            
            MERGE (p) -[:KEA_HISTOLOGY]-> (h)
            
            MERGE (p) -[:KEA_ER_STATUS]-> (er)
            MERGE (p) -[:KEA_PR_STATUS]-> (pr)
            MERGE (p) -[:KEA_Caq2_STATUS]-> (Caq2) 
            
            MERGE (t) -[:KEA_pro]-> (pro1)
            MERGE (t) -[:KEA_pro]-> (pro2)
            MERGE (t) -[:KEA_pro]-> (pro3)
            
            MERGE (t) -[:KEA_HISTOLOGY]-> (h)
            
            MERGE (t) -[:KEA_ER_STATUS]-> (er)
            MERGE (t) -[:KEA_PR_STATUS]-> (pr)
            MERGE (t) -[:KEA_Caq2_STATUS]-> (Caq2)
            
            MERGE (h) -[:KEA_ER_STATUS]-> (er)
            MERGE (h) -[:KEA_PR_STATUS]-> (pr)
            MERGE (h) -[:KEA_Caq2_STATUS]-> (Caq2)
    """
    
    
    tx.run(q, patient_id = row['Patient_ID_new'], patient_age = row['Age'], patient_gender = row['Gender'], 
              pro1_name = 'pro1', pro1_level = row['pro1_category'],
              pro2_name = 'pro2', pro2_level = row['pro2_category'],
              pro3_name = 'pro3', pro3_level = row['pro3_category'],
              stage = row['Tumour_Stage'],
              type = row['Histology'],
              status_ER = row['ER status'],
              status_PR = row['PR status'],
              status_Caq2 = row['Caq2 status'])
    
#%%    
def create_GH_patient_pro(tx, row):
    q = """ MERGE (p:Patient {id: $patient_id, age: $patient_age, gender: $patient_gender})
            MERGE (pro1: pro1 {name: $pro1_name, level: $pro1_level})
            MERGE (pro2: pro2 {name: $pro2_name, level: $pro2_level})
            MERGE (pro3: pro3 {name: $pro3_name, level: $pro3_level})
            
            MERGE (p) -[:KEA_pro]-> (pro1)
            MERGE (p) -[:KEA_pro]-> (pro2)
            MERGE (p) -[:KEA_pro]-> (pro3)           
    """

    tx.run(q, patient_id = row['Patient_ID_new'], patient_age = row['Age'], patient_gender = row['Gender'], 
              pro1_name = 'pro1', pro1_level = row['pro1_category'],
              pro2_name = 'pro2', pro2_level = row['pro2_category'],
              pro3_name = 'pro3', pro3_level = row['pro3_category'],
              stage = row['Tumour_Stage'],
              type = row['Histology'],
              status_ER = row['ER status'],
              status_PR = row['PR status'],
              status_Caq2 = row['Caq2 status'])
    
#%%
def create_GH_patient_pro_tumor(tx, row):
    
    q = """ MERGE (p:Patient {id: $patient_id, age: $patient_age, gender: $patient_gender})
            MERGE (pro1: pro1 {name: $pro1_name, level: $pro1_level})
            MERGE (pro2: pro2 {name: $pro2_name, level: $pro2_level})
            MERGE (pro3: pro3 {name: $pro3_name, level: $pro3_level})
            
            MERGE (t:Tumor {stage: $stage})
            
            MERGE (p) -[:KEA_pro]-> (pro1)
            MERGE (p) -[:KEA_pro]-> (pro2)
            MERGE (p) -[:KEA_pro]-> (pro3)
            
            MERGE (p) -[:KEA_TUMOR_STAGE]-> (t)
            
            MERGE (t) -[:KEA_pro]-> (pro1)
            MERGE (t) -[:KEA_pro]-> (pro2)
            MERGE (t) -[:KEA_pro]-> (pro3)
    """
    
    
    tx.run(q, patient_id = row['Patient_ID_new'], patient_age = row['Age'], patient_gender = row['Gender'], 
              pro1_name = 'pro1', pro1_level = row['pro1_category'],
              pro2_name = 'pro2', pro2_level = row['pro2_category'],
              pro3_name = 'pro3', pro3_level = row['pro3_category'],
              stage = row['Tumour_Stage'],
              type = row['Histology'],
              status_ER = row['ER status'],
              status_PR = row['PR status'],
              status_Caq2 = row['Caq2 status'])
    
    
#%%

def create_GH_patient_pro_tumor(tx, row):
    
    q = """ MERGE (p:Patient {id: $patient_id, age: $patient_age, gender: $patient_gender})
            MERGE (pro1: pro1 {name: $pro1_name, level: $pro1_level})
            MERGE (pro2: pro2 {name: $pro2_name, level: $pro2_level})
            MERGE (pro3: pro3 {name: $pro3_name, level: $pro3_level})
            
            MERGE (t:Tumor {stage: $stage})
            
            MERGE (p) -[:KEA_pro]-> (pro1)
            MERGE (p) -[:KEA_pro]-> (pro2)
            MERGE (p) -[:KEA_pro]-> (pro3)
            
            MERGE (p) -[:KEA_TUMOR_STAGE]-> (t)
            
            
            MERGE (t) -[:KEA_pro]-> (pro1)
            MERGE (t) -[:KEA_pro]-> (pro2)
            MERGE (t) -[:KEA_pro]-> (pro3)
            
    """
    
    tx.run(q, patient_id = row['Patient_ID_new'], patient_age = row['Age'], patient_gender = row['Gender'], 
              pro1_name = 'pro1', pro1_level = row['pro1_category'],
              pro2_name = 'pro2', pro2_level = row['pro2_category'],
              pro3_name = 'pro3', pro3_level = row['pro3_category'],
              stage = row['Tumour_Stage'],
              type = row['Histology'],
              status_ER = row['ER status'],
              status_PR = row['PR status'],
              status_Caq2 = row['Caq2 status'])
    
   