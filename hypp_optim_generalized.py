from helpers import *
from transformers_modules import *

methodology = 'TR'  # TR or RF

filename = 'SRB_S2-S1_data.pkl'

train_val_test_ratio = [0.7, 0.15, 0.15]

data = load_data(filename, train_val_test_ratio[0], train_val_test_ratio[2])
dates = data[0].columns.str[:4].unique()

model = {'RF': RandomForestClassifier, 'TR': TransformerModel}

params = {'RF': {'n_estimators': [20, 100], 'max_depth': [5], 'min_samples_leaf': [5]},
          'TR': {'nhead': [2, 4], 'output_dim': [128], 'num_encoder_layers': [2], 'dropout': [0.1], 'input_dim': [len(data[0].columns) // len(dates)], 'num_classes': [len(np.unique(data[1]))]}}

combination_params = [dict(zip(params[methodology].keys(), values)) for values in itertools.product(*params[methodology].values())]

df_res = pd.DataFrame()

meta_params = {'filename': filename, 'methodology': methodology, 'dates': dates}

for p in combination_params:
    df_res = hyperparameter_opt(model[methodology], p, data, df_res, **meta_params)
    df_res.to_csv(f'results/hypp_opt_{filename[:filename.rfind("_")]}_{re.sub("[^A-Z]", "", model[methodology].__name__)}.csv')

