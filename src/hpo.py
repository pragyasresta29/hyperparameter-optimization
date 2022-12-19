import copy
import json
import os
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, tree, metrics, model_selection


from classifiers import *

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# decision tree paramaters
dt_params = params = {
    'max_depth': [2, 6, 8, 10, 12, 16],
    'min_samples_leaf': [5, 10, 20, 50],
    'criterion': ["gini", "entropy"]
}

# adaboost  paramaters
ada_params = {
    'n_estimators': [2, 6, 8, 10, 12, 16],
    'learning_rate': [(0.97 + x / 100) for x in range(0, 5)],
    'algorithm': ['SAMME', 'SAMME.R']
}

# xgboost paramaters
xgb_params = {
    "max_depth": [6, 8, 10, 12, 16],
    "eta": [0.3, 0.4, 0.5],
    "subsample": [0, 0.6, 0.8],
    "colsample_bytree": [0.6, 0.7, 0.8],
}

# models to be used
models = [{'name': 'DecisionTree', 'model': tree.DecisionTreeClassifier(), 'params': dt_params},
          {'name': 'AdaBoost', 'model': AdaBoostClassifier(), 'params': ada_params},
          {'name': 'XGBoost', 'model': xgb.XGBClassifier(verbosity = 0, silent=True), 'params': xgb_params}]


# pass the ranges and dataset to this function to generate best_parameters
def gather_hpo_data(sizes, data, data_info):
    print("-----Gather HPO Data-----")
    for r in sizes:
        new_data = copy.deepcopy(data)
        # print(len(new_data.data))
        new_data.data = data.data[:r]
        new_data.target = data.target[:r]
        # print(len(new_data.data))
        for model in models:
            print("-----Data: {} Size: {}, Model: {}-----".format(data_info, r, model['name']))
            best_params = find_optimal_paramaters(model['model'], new_data, model['params'])
            info = {'data': data_info, 'range': r, 'model': model['name'],'best_params': best_params}
            update_data(info)


def find_optimal_paramaters(model, dataset, params):
    print("-----Finding optimal params-----")
    print("size: ", len(dataset.data))
    tuned_model = model_selection.GridSearchCV(model, params, scoring="accuracy", cv=10)
    tuned_model.fit(dataset.data, dataset.target)
    return tuned_model.best_params_


def update_data(info):
    print("-----updating data---[info:", info)
    if os.path.exists('info.json'):
        data = []
        with open('info.json') as file:
            data = json.load(file)
        data.append(info)
        with open('info.json', 'w') as file:
            json_object = json.dumps(data, indent=4)
            file.write(json_object)

    else:
        data = [info]
        with open('info.json', 'w') as file:
            json_object = json.dumps(data, indent=4)
            file.write(json_object)
    print("-----update complete-----")


#datasets
steel_data = datasets.fetch_openml(data_id=1504)
eeg_data = datasets.fetch_openml(data_id=1471)
electricity_data = datasets.fetch_openml(data_id=151)
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [1])], remainder="passthrough")
new_data = ct.fit_transform(electricity_data.data)
electricity_data.data = pd.DataFrame(new_data, columns=ct.get_feature_names(), index=electricity_data.data.index)


gather_hpo_data([100], steel_data, 'Steel_data')
gather_hpo_data([100], eeg_data, 'EEG_data')
gather_hpo_data([100], electricity_data, 'Electricity_data')












