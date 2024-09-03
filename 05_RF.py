# RF
# Train RF on Slovenian data and appply to Serbian.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
slo_data = pd.read_pickle('data/SLO_S1_data.pkl')
srb_data = pd.read_pickle('data/SRB_S2-S1_data.pkl')

print("These are dataset results:")

# Split features and labels
X = slo_data.drop(columns=['ID', 'class'])
y = slo_data['class']
X_srb = srb_data.drop(columns=['ID', 'class'])
y_srb = srb_data['class']

# Split the data into training and validation (80%), and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split for the Serbian data into training and validation (15%), and testing (85%) sets
X_srb_train, X_srb_test, y_srb_train, y_srb_test = train_test_split(X_srb, y_srb, test_size=0.85, random_state=42, stratify=y_srb)

a, b, c, d = X_srb_train, X_srb_test, y_srb_train, y_srb_test

X_srb_train, X_srb_test, y_srb_train, y_srb_test = b, a, d, c

# Define the parameter grid for hyperparameter optimization
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [1,5, 10, 50],
    # 'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10, 50, 100],
    # 'bootstrap': [True, False]
}

# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [5],
#     # 'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [5],
#     # 'bootstrap': [True, False]
# }

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=None, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)
grid_search.fit(X_srb_train, y_srb_train)

# Retrieve the best model
best_rf = grid_search.best_estimator_

# Train the model on the training set
# best_rf.fit(X_train, y_train)
best_rf.fit(X_srb_train, y_srb_train)

# Evaluate the model on the testing set

# y_pred_test = best_rf.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_pred_test)
print("Best parameter combination: ", best_rf)
print("Best parameter combination: ", grid_search.best_params_)
# print("Test Accuracy:", test_accuracy)
# print("Classification Report for Test Set:")
# print(classification_report(y_test, y_pred_test))

# Evaluate the model on the additional SRB data

y_pred_srb = best_rf.predict(X_srb_test)
srb_accuracy = accuracy_score(y_srb_test, y_pred_srb)
print("SRB Test Accuracy:", srb_accuracy)
print("Classification Report for SRB Set:")
print(classification_report(y_srb_test, y_pred_srb))