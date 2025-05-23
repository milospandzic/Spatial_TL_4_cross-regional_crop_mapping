from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from torch.utils.data import DataLoader, Dataset

import itertools

import joblib
import re

from transformers_modules import *

data_path = Path('data')
results_path = Path('results')

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

def train_model(model, data, methodology, dates):

    res = {}

    X_train, Y_train, X_val, Y_val, _, _ = data

    if methodology=='TR':
        train_loader = prepare_dataloader(X_train,Y_train, True)
        val_loader = prepare_dataloader(X_val, Y_val, False)

        model, optimizer = model.fit(model, train_loader, val_loader, dates, num_epochs=3)

        predictions = model.predict(model, val_loader, dates)
        predictions = torch.tensor(predictions, device='cpu')

    else:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_val)
        optimizer = None

    res = evaluation_metrics(res, Y_val, predictions)

    return res, model, optimizer

def hyperparameter_opt(model_def, params, data, df, filename, methodology, dates):

    model = model_def(**params)

    for iteration in np.arange(0, 1):
        res, model, optimizer = train_model(model, data, methodology, dates)
        res['Params'] = str(params)
        res['Iteration'] = iteration + 1
        df = pd.concat([df, pd.DataFrame(res, index=[0])], ignore_index=True)

        param_str = [f"{k}-{v}" for k, v in params.items()]

        model_filename = f'models/{filename[:filename.rfind("_")]}/{methodology}-{"_".join(param_str)}_iteration-{iteration + 1}.joblib'

        if methodology == 'TR':
            save_dl_model(model, optimizer, model_filename)
        else:
            joblib.dump(model, model_filename)

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


def extract_optimal_parameters(filename_results):
    results = pd.read_csv(results_path / filename_results, index_col=0)

    params_agg = results.groupby('Params').mean()
    optimal_params = params_agg.iloc[np.argmax(params_agg['F1 score']), :].name
    optimal_params_res = results.loc[results['Params'] == optimal_params, :]
    optimal_iteration = optimal_params_res.loc[optimal_params_res['F1 score'] == max(optimal_params_res['F1 score']), 'Iteration'].values[0]

    return optimal_params, optimal_iteration


def load_best_model(model_def, params, iteration, filename, methodology):
    param_str = [f"{k}-{v}" for k, v in eval(params).items()]

    model_path = f'models/{filename[:filename.rfind("_")]}/{methodology}-{"_".join(param_str)}_iteration-{iteration}.joblib'

    if methodology == 'RF':
        model = joblib.load(model_path)
    else:
        model = model_def(**eval(params))
        model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])

    return model

############################################ Transformers specific #####################################################

def reshape_input(X):
    X = X.values.reshape(-1, len(X.columns.str[:4].unique()),  len(X.columns) // len(X.columns.str[:4].unique()))  # Reshape to (samples, time_steps, features)
    return X

def prepare_dataloader(X, Y, shuffle):

    X = reshape_input(X)

    dataset = TimeSeriesDataset(X, Y.values)

    loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

    return loader


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def save_dl_model(model, optimizer, path="models/transformer_model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

