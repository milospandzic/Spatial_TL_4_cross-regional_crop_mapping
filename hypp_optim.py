from helpers import *

deep_learning = False

filename = 'SRB_S2-S1_data.pkl'

train_val_test_ratio = [0.7, 0.15, 0.15]

params = {'n_estimators': [20, 100],
          'max_depth': [5],
          'min_samples_leaf': [5]
          }

model = RandomForestClassifier

data = load_data(filename, train_val_test_ratio[0], train_val_test_ratio[2])
dates = data[0].columns.str[:4].unique()

if deep_learning:

    params['input_dim'] = [len(data[0].columns) // len(dates)]
    params['num_classes'] = [len(np.unique(data[1]))]

combination_params = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]

df_res = pd.DataFrame()

meta_params = {'filename': filename, 'deep_learning': deep_learning, 'dates': dates}

for p in combination_params:
    df_res = hyperparameter_opt(model, p, data, df_res, **meta_params)
    df_res.to_csv(f'results/hypp_opt_{filename[:filename.rfind("_")]}_{re.sub("[^A-Z]", "", model.__name__)}.csv')

