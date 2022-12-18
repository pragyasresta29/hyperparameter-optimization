import json
import os
import random

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from matplotlib import pyplot

from classifiers import *

# maximum size of data we want to use for training
max_size = 10000

steel_data = datasets.fetch_openml(data_id=1504)
eeg_data = datasets.fetch_openml(data_id=1471)
electricity_data = datasets.fetch_openml(data_id=151)
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [1])], remainder="passthrough")
new_data = ct.fit_transform(electricity_data.data)
electricity_data.data = pd.DataFrame(new_data, columns=ct.get_feature_names_out(), index=electricity_data.data.index)


def compute_accuracy(row):
    dataset = get_data(row['data'])
    model = get_model(row['model'])
    params = eval(row['best_params'])
    return model(dataset, params)


def get_data(name):
    data = []
    if name == 'Steel_data':
        data = steel_data
    elif name == 'EEG_data':
        data = eeg_data
    elif name == 'Electricity_data':
        data = electricity_data
    size = min(max_size, len(data.data))
    data.data = data.data[:size]
    data.target = data.target[:size]
    return data


def get_model(name):
    if name == "DecisionTree":
        return dt_classifier
    elif name == "AdaBoost":
        return ada_classifier
    elif name == "XGBoost":
        return xgboost_classifier

def compute_all_accuracies():
    df = fetch_data()
    print("-----Computing accuracies-----")
    for index, row in df.iterrows():
        if pd.isna(row['Accuracy']):
            score = compute_accuracy(row)
            df.loc[index, ['Accuracy']] = score
            print(df.loc[index])
            df.to_csv('hpo_data.csv')
    print("-----Computing accuracies complete!-----")


def fetch_data():
    if os.path.exists('hpo_data.csv'):
        return pd.read_csv('hpo_data.csv', index_col=0)
    else:
        print("-----Create csv file for hpo data-----")
        with open('info.json') as file:
            data = json.load(file)
        df = pd.DataFrame().from_dict(data)
        df['Accuracy'] = np.nan
        df.to_csv('hpo_data.csv')
        return df


def plot_graph():
    df = fetch_data()
    datasets = df['data'].unique().tolist()
    for data_name in datasets:
        data = df[df['data'] == data_name]
        models = data['model'].unique().tolist()
        color = ['orange', 'blue', 'green', 'yellow', 'magenta']
        for index, model in enumerate(models):
            df_M = data[data['model'] == model]
            pyplot.plot(df_M['range'], df_M['Accuracy'], color=color[index], label=model)
        pyplot.xlabel("Training Size for HPO")
        pyplot.ylabel("Accuracy")
        pyplot.ylim([0,1])
        pyplot.legend(loc="lower right")
        pyplot.savefig(data_name + '.png')
        pyplot.show()


# compute_all_accuracies()
plot_graph()










