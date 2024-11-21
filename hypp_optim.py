from helpers import *

filename = 'SRB_S2-S1_data.pkl'

params = {'n_estimators': [10, 100, 500],
          'max_depth': [3, 2],
          'min_samples_leaf': [5,15]}

model = RandomForestClassifier

data = load_data(filename, 0.15, 0.7)
combination_params = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]

df_res = pd.DataFrame()

for p in combination_params:
    df_res = hyperparameter_opt(model, p, data, df_res)
    df_res.to_csv(f'results/hypp_opt_{filename[:filename.rfind("_")]}.csv')
