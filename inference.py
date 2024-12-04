from helpers import *

methodology = 'RF'

filename_train = 'SLO_S2-S1_data.pkl'
filename_test = 'SRB_S2-S1_data.pkl'
filename_results = f'hypp_opt_{filename_train[:filename_train.rfind("_")]}_{methodology}.csv'

train_val_test_ratio = [0.15, 0.15, 0.7]

model = {'RF': RandomForestClassifier, 'TR': TransformerModel}

#-----------------------------------------------------------------------------------------------------------------------#

optimal_params, optimal_iteration = extract_optimal_parameters(filename_results)

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(filename_test, train_val_test_ratio[0], train_val_test_ratio[2])
dates = X_train.columns.str[:4].unique()

best_model = load_best_model(model[methodology], optimal_params, optimal_iteration, filename_train, methodology)

#-----------------------------------------------------------------------------------------------------------------------#

if methodology == 'RF':
    predictions = best_model.predict(X_test)
else:
    test_loader = prepare_dataloader(X_test, Y_test, False)
    best_model.to(device)
    predictions = best_model.predict(best_model, test_loader, dates)

res = pd.DataFrame(evaluation_metrics({}, Y_test, predictions), index=[0])

approach = 'FS' if filename_train.split('_')[0] == filename_test.split('_')[0] else 'naive'
res.to_csv(results_path/f'inference-{approach}_{filename_train[:filename_train.rfind("_")]}_{filename_test[:filename_test.rfind("_")]}_{methodology}.csv')

# ---------------------------------------------- TL --------------------------------------------------------------------#

if approach=='naive':
    if methodology=='RF':
        best_model.warm_start = True
        best_model.n_estimators += best_model.n_estimators
        best_model.fit(X_train, Y_train)
        predictions = best_model.predict(X_test)

    else:
        train_loader = prepare_dataloader(X_train, Y_train, True)
        val_loader = prepare_dataloader(X_val, Y_val, False)

        best_model.freeze_feature_extractor()

        best_model, _ = best_model.fit(best_model, train_loader, val_loader, dates, num_epochs=3)

        predictions = best_model.predict(best_model, test_loader, dates)


    res = pd.DataFrame(evaluation_metrics({}, Y_test, predictions), index=[0])
    res.to_csv(results_path/f'inference-TL_{filename_train[:filename_train.rfind("_")]}_{filename_test[:filename_test.rfind("_")]}_{methodology}.csv')


