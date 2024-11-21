#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import time
from joblib import dump

import torch
from sklearn.preprocessing import LabelEncoder

from helpers import *

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data('SLO_S2_data.pkl', 0.7, 0.15)

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

