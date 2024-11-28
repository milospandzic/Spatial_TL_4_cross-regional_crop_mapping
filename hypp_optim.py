from helpers import *

filename = 'SRB_S2-S1_data.pkl'

params = {'n_estimators': [100],
          'max_depth': [5],
          'min_samples_leaf': [5]
          }

model = RandomForestClassifier

train_val_test_ratio = [0.7, 0.15, 0.15]

data = load_data(filename, train_val_test_ratio[0], train_val_test_ratio[2])
combination_params = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]

df_res = pd.DataFrame()

for p in combination_params:
    df_res = hyperparameter_opt(model, p, data, df_res, filename)
    df_res.to_csv(f'results/hypp_opt_{filename[:filename.rfind("_")]}_{re.sub("[^A-Z]", "", model.__name__)}.csv')

