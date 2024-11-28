from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import Dataset

import itertools

import torch
import joblib
import re

data_path = Path('data')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

crops = ['corn', 'wheat', 'soybean', 'sunflower', 'sugar beet', 'oilseed rape', 'barley', 'clover', 'orchard']

def load_data(filename, train_size, test_size):
    """
    :param filename: .pkl file name
    :param train_size: percentage of full dataset (0.0 - 1.0)
    :param test_size: percentage of full dataset (0.0 - 1.0)
    :return:
    """
    data = pd.read_pickle(data_path / filename)

    X = data.iloc[:, 1:-1]
    Y = data.iloc[:, -1]

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=1-train_size, random_state=seed)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size/(1-train_size), random_state=seed)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def train_model(model, data):

    res = {}

    # X_train, Y_train, X_val, Y_val, _, _ = data
    #
    # model.fit(X_train, Y_train)

    train_loader, val_loader,  = data

    model.fit(train_loader, val_loader, num_epochs = 3)

    # predictions = model.predict(X_val)
    #
    # res = evaluation_metrics(res, Y_val, predictions)

    return res, model

def hyperparameter_opt(model_def, params, data, df, filename):

    model = model_def(**params)

    for iteration in np.arange(0, 1):
        res, model = train_model(model, data)
        res['Params'] = str(params)
        res['Iteration'] = iteration + 1
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        param_str = [f"{k}-{v}" for k, v in params.items()]

        joblib.dump(model,f'models/{filename[:filename.rfind("_")]}/{re.sub("[^A-Z]", "", model_def.__name__)}-{"_".join(param_str)}_iteration-{iteration+1}.joblib')

    return df

def evaluation_metrics(res, labels, predictions):

    res['accuracy'] = accuracy_score(labels, predictions)
    res['F1 score'] = f1_score(labels, predictions, average='macro')
    res['precision'] = precision_score(labels, predictions, average='macro')
    res['recall'] = recall_score(labels, predictions, average='macro')

    res = evaluation_metrics_per_class(res, labels, predictions)

    return res

def evaluation_metrics_per_class(res, labels, predictions):

    f1_score_per_class = f1_score(labels, predictions, average=None)
    precision_per_class = precision_score(labels, predictions, average=None)
    recall_per_class = recall_score(labels, predictions, average=None)

    for label in np.unique(labels):
        res[f'F1 score {crops[label]}'] = f1_score_per_class[label]
        res[f'Precision {crops[label]}'] = precision_per_class[label]
        res[f'Recall {crops[label]}'] = recall_per_class[label]

    return res

############################################ Transformers specific #####################################################

def reshape_input(X):
    X = X.values.reshape(-1, len(X.columns.str[:4].unique()),  len(X.columns) // len(X.columns.str[:4].unique()))  # Reshape to (samples, time_steps, features)
    return X


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]