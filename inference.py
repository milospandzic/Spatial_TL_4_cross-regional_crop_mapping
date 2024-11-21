from helpers import *


results_path = Path('results')
filename_results = 'hypp_opt_SRB_S2-S1_RFC.csv'

results = pd.read_csv(results_path / filename_results, index_col=0)

params_agg = results.groupby('Params').mean()
optimal_params = params_agg.iloc[np.argmax(params_agg['F1 score']), :].name

optimal_params_res = results.loc[results['Params'] == optimal_params, :]

optimal_iteration = optimal_params_res.loc[optimal_params_res['F1 score']==max(optimal_params_res['F1 score']), 'Iteration'].values[0]

filename_train = 'SRB_S2-S1_data.pkl'
filename_test = 'SRB_S2-S1_data.pkl'

model = RandomForestClassifier

_, _, _, _, X_test, Y_test = load_data(filename_test, 0.15, 0.7)


param_str = [f"{k}-{v}" for k, v in eval(optimal_params).items()]

best_model_path = f'models/{filename_train[:filename_train.rfind("_")]}/{re.sub("[^A-Z]", "", model.__name__)}-{"_".join(param_str)}_iteration-{optimal_iteration}.joblib'
best_loaded_model = joblib.load(best_model_path)

predictions = best_loaded_model.predict(X_test)

res = evaluation_metrics({}, Y_test, predictions)

print(res)

