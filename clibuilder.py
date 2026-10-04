# -*- coding: utf-8 -*-
"""
Created on Sep 25 2026

@author: liangyingliu
"""

import pandas as pd
import sys
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict

#%%

class CliBuilder():
    def __init__(self, df_re, df_ent):
        self.df_re = df_re
        self.df_ent = df_ent
        
    def search_ent_mention(self):
        ent1 = pd.merge(self.df_re, self.df_ent, 
                           left_on = ['ent1_Type', 'ent1_mshID'],
                           right_on = ['ent_Type', 'ent_mshID'],
                           how = 'inner').drop(['ent_Type', 'ent_mshID', 'Resource'], axis = 1)
        
        ent1.rename(columns = {'Mention':'ent1_Mention'}, inplace = True)
        ent1_2 = pd.merge(ent1, self.df_ent, 
                           left_on = ['ent2_Type', 'ent2_mshID'],
                           right_on = ['ent_Type', 'ent_mshID'],
                           how = 'inner').drop(['ent_Type', 'ent_mshID', 'Resource'], axis = 1)
        
        ent1_2.rename(columns = {'Mention':'ent2_Mention'}, inplace = True)
        return ent1_2
        
    def REclib_to_df(self, df):
        RE_df = []
        for idx, row in df.iterrows():
             ent1_Type= row['ent1_mshID'].split('|')[0]
             ent1_mshID= row['ent1_mshID'].split('|')[1][5:]
             
             ent2_Type= row['ent2_mshID'].split('|')[0]
             ent2_mshID= row['ent2_mshID'].split('|')[1][5:]
             
             RE_df.append([row['pw'], row['re'], ent1_Type, ent1_mshID, ent2_Type, ent2_mshID])
             
        return pd.DataFrame(RE_df, columns = ['pw', 're', 'ent1_Type', 'ent1_mshID', 'ent2_Type', 'ent2_mshID'])
        

    def entPutator_to_df(self, df):
        df['ent_mshID'] = df['ent_mshID'].str.split(":").str[1]
        return df
        
    def calculate(self): 
        ent_types = sorted(list(set(df_re['ent1_Type'].unique()) | 
                                 set(df_re['ent2_Type'].unique())))

        label_to_idx = {}
        idx_to_label = {}
        node_colors = []

        left_types = sorted(df_re['ent1_Type'].unique())
        for i, label in enumerate(left_types):
            label_to_idx[f"{label}_left"] = i
            idx_to_label[i] = label
            node_colors.append(color_map[label])

        right_types = sorted(df_re['ent2_Type'].unique())
        offset = len(left_types)
        for i, label in enumerate(right_types):
            label_to_idx[f"{label}_right"] = i + offset
            idx_to_label[i + offset] = label
            node_colors.append(color_map[label])

        source = []
        target = []
        value = []
        link_colors = []

        for _, row in df_re.iterrows():
            source_idx = label_to_idx[f"{row['ent1_Type']}_left"]
            target_idx = label_to_idx[f"{row['ent2_Type']}_right"]
            source.append(source_idx)
            target.append(target_idx)
            value.append(1)
           
            base_color = color_map[row['ent1_Type']]
            rgba_color = f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.4,)}"
            link_colors.append(rgba_color)

#%%

CUDA_VISIBLE_DEVICES=$1 python src/run_exp.py \
    --task_name "bio" \
    --test_file "$OUTPUT_TSV" \
	--max_seq_length 512
    --model_name_or_path "$PRETRAINED_MODEL_PATH" \
    --output_dir "bio_model" \
    --logging_steps 10 \
	--save_steps 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
	--num_train_epochs 10 \
#%%
