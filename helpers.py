from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

import itertools

import torch
import joblib
import re

data_path = Path('data')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def load_data(filename, train_size, test_size):

    data = pd.read_pickle(data_path / filename)

    X = data.iloc[:, 1:-1]
    Y = data.iloc[:, -1]

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1-train_size, random_state=seed)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size/(1-train_size), random_state=seed)

    print(X_test[:5])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def train_model(model, data):

    res = {}

    X_train, Y_train, X_val, Y_val, _, _ = data

    model.fit(X_train, Y_train)

    predictions = model.predict(X_val)

    res = evaluation_metrics(res, Y_val, predictions)

    return res, model

def hyperparameter_opt(model_def, params, data, df, filename):

    model = model_def(**params)

    for iteration in np.arange(0, 5):
        res, model = train_model(model, data)
        res['Params'] = str(params)
        res['Iteration'] = iteration + 1
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        param_str = [f"{k}-{v}" for k, v in params.items()]

        joblib.dump(model,f'models/{filename[:filename.rfind("_")]}/{re.sub("[^A-Z]", "", model_def.__name__)}-{"_".join(param_str)}_iteration-{iteration+1}.joblib')

    return df

def evaluation_metrics(res, Y_val, predictions):

    res['accuracy'] = accuracy_score(Y_val, predictions)
    res['F1 score'] = f1_score(Y_val, predictions, average='macro')
    res['precision'] = precision_score(Y_val, predictions, average='macro')
    res['recall'] = recall_score(Y_val, predictions, average='macro')

    return res