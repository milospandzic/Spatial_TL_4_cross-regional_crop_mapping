#!/usr/bin/env python
# coding: utf-8

import numpy as np
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import time
from joblib import dump

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import LabelEncoder

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the data
slo_data = pd.read_pickle('data/SLO_S1_data.pkl')
srb_data = pd.read_pickle('data/SRB_S1_data.pkl')
dates = slo_data.drop(columns=['ID', 'class']).columns.str[:4].unique()

X_slo = slo_data.iloc[:,1:-1]
y_slo = slo_data.iloc[:,-1]
X_srb = srb_data.iloc[:,1:-1]
y_srb = srb_data.iloc[:,-1]

# Split the SLO data into training (70%), validation (15%), and testing (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_slo, y_slo, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

# Split the SRB data into training (15%), validation (15%), and testing (70%) sets
X_temp_srb, X_test_srb, y_temp_srb, y_test_srb = train_test_split(X_srb, y_srb, test_size=0.7, random_state=seed)
X_train_srb, X_val_srb, y_train_srb, y_val_srb = train_test_split(X_temp_srb, y_temp_srb, test_size=0.5, random_state=seed)



def train_model(X_train, Y_train, X_val, Y_val, params):

    res = {}

    model = RandomForestClassifier(n_estimators=params[0], max_depth=params[1], min_samples_leaf=params[2],
                                   warm_start=True)

    start_time = time.time()
    model.fit(X_train, Y_train)
    time_comp = time.time() - start_time

    dump(model,
         f'models/RF-n_estimators-{params[0]}-max_depths-{params[1]}-min_samples_leaf-{params[2]}.joblib')

    predictions = model.predict(X_val)

    res['accuracy'] = accuracy_score(Y_val, predictions)
    res['F1 score'] = f1_score(Y_val, predictions, average='micro')
    res['precision'] = precision_score(Y_val, predictions, average='micro')
    res['recall'] = recall_score(Y_val, predictions, average='micro')

    res['Time complexity'] = time_comp

    return res

n_estimators = [100, 500, 1000]
max_depths = [1, 5, 10, 50]
min_samples_leaf = [1, 5, 10, 50, 100]
params = list(product(n_estimators, max_depths, min_samples_leaf))

df_res = pd.DataFrame()

for p in params:
    res = train_model(X_train_srb, y_train_srb, X_val_srb, y_val_srb, p)

    res['Params'] = str(p)

    # df_res = df_res.append(pd.DataFrame(res, index=[0]))
    df_res = pd.concat([df_res, pd.DataFrame(res, index=[0])], ignore_index=True)
    df_res.to_csv('results/hypp_opt_RF_SRB_S1.csv')

'''
import joblib
best_model_path = 'models/RF-n_estimators-100-max_depths-50-min_samples_leaf-1.joblib'  # Replace with your actual file path
best_loaded_model = joblib.load(best_model_path)
'''