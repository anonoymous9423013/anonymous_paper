from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from rq3_dataset import balance
import logging as log
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import utils 
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import os

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def trainDecisionTree(df, cv=5, **kwargs):
    max_depth = kwargs.get('max_depth', 5)
    model = DecisionTreeClassifier(max_depth=max_depth)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = 'Decision Tree'
    return scores, desc

def trainRandomForest(df, cv=5, **kwargs):
    max_depth = kwargs.get('max_depth', 20)
    n_estimators = kwargs.get('n_estimators', 20)
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = 'Random Forest'
    return scores, desc


def trainSVM(df, cv=5, **kwargs):
    kernel = kwargs.get('kernel', 'rbf')
    model = SVC(kernel=kernel)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = f'SVM with {kernel} kernel'
    return scores, desc

def trainMLP(df, cv=5, **kwargs):
    hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (50, 100))
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate='adaptive')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = f'MLP with {hidden_layer_sizes} hidden layer sizes'
    return scores, desc


def train(df_main, cv=5, approach='dt', output_file='rq3.txt', platform='PID', **kwargs):
    results_dict = {'Method': [], 'Input': [], 'Precision': [], 'Recall': [], 'F1': []}
    flen = kwargs.get('fitnesses_len', 0)
    df_train = df_main.sample(frac=0.8, random_state=200)
    df_test = df_main.drop(df_train.index).reset_index(drop=True)
    df_train = balance(df_train, output_file=None)
    train = df_train.copy()
    test = df_test.copy()
    with open(output_file, 'a') as f:
        ##  With all the features...
        if approach == 'dt':
            scores, desc = trainDecisionTree(train, cv=cv, **kwargs)
        elif approach == 'rf':
            scores, desc = trainRandomForest(train, cv=cv, **kwargs)
        elif approach == 'svm':
            scores, desc = trainSVM(train, cv=cv, **kwargs)
        elif approach == 'mlp':
            scores, desc = trainMLP(train, cv=cv, **kwargs)
        models = scores['estimator']
        predictions = [model.predict(test.iloc[:, :-1].values) for model in models]
        precisions = [precision_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        recalls = [recall_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        f1s = [f1_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        results_dict['Method'].append(approach)
        results_dict['Input'].append('All scenario features' + kwargs['input'])
        results_dict['Precision'].append(np.max(precisions))
        results_dict['Recall'].append(np.max(recalls))
        results_dict['F1'].append(np.max(f1s))
        f.write('*' * 25 + '  With all the features  ' + '*' * 25 + '\n')
        f.write(f'Approach: {desc}\n')
        f.write(f'Inputs: {list(df_main.columns[:-1])}\n')
        f.write(f'Output: {list(df_main.columns[-1])}\n')
        f.write(f'Number of fitnesses included in input: {flen}\n')
        f.write(f'Precision scores: {scores["test_precision"]}\n')
        f.write(f'Mean precision score: {scores["test_precision"].mean()}\n')
        f.write(f'Recall scores: {scores["test_recall"]}\n')
        f.write(f'Mean recall score: {scores["test_recall"].mean()}\n')
        f.write(f'F1 scores: {scores["test_f1"]}\n')
        f.write(f'Mean F1 score: {scores["test_f1"].mean()}\n')
        f.write(f'Test set precision: {precisions}\n')
        f.write(f'Test set mean precision: {np.mean(precisions)}\n')
        f.write(f'Test set best precision: {np.max(precisions)}\n')
        f.write(f'Test set recall: {recalls}\n')
        f.write(f'Test set mean recall: {np.mean(recalls)}\n')
        f.write(f'Test set best recall: {np.max(recalls)}\n')
        f.write(f'Test set F1: {f1s}\n')
        f.write(f'Test set mean F1: {np.mean(f1s)}\n')
        f.write(f'Test set best F1: {np.max(f1s)}\n')
        ##  Without weather features...
        if platform == 'PID':
            df = df_main.drop(['Weather1', 'Weather2', 'Weather3'], axis=1)
            train = df_train.drop(['Weather1', 'Weather2', 'Weather3'], axis=1)
            test = df_test.drop(['Weather1', 'Weather2', 'Weather3'], axis=1)
        elif platform == 'Pylot':
            df = df_main.drop(['Weather'], axis=1)
            train = df_train.drop(['Weather'], axis=1)
            test = df_test.drop(['Weather'], axis=1)
        if approach == 'dt':
            scores, desc = trainDecisionTree(train, cv=cv, **kwargs)
        elif approach == 'rf':
            scores, desc = trainRandomForest(train, cv=cv, **kwargs)
        elif approach == 'svm':
            scores, desc = trainSVM(train, cv=cv, **kwargs)
        elif approach == 'mlp':
            scores, desc = trainMLP(train, cv=cv, **kwargs)
        models = scores['estimator']
        predictions = [model.predict(test.iloc[:, :-1].values) for model in models]
        precisions = [precision_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        recalls = [recall_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        f1s = [f1_score(test.iloc[:, -1].values, prediction) for prediction in predictions]

        results_dict['Method'].append(approach)
        results_dict['Input'].append('Without weather features' + kwargs['input'])
        results_dict['Precision'].append(np.max(precisions))
        results_dict['Recall'].append(np.max(recalls))
        results_dict['F1'].append(np.max(f1s))
        f.write('*' * 25 + '  Without weather features  ' + '*' * 25 + '\n')
        f.write(f'Approach: {desc}\n')
        f.write(f'Inputs: {list(df.columns[:-1])}\n')
        f.write(f'Output: {list(df.columns[-1])}\n')
        f.write(f'Number of fitnesses included in input: {flen}\n')
        f.write(f'Precision scores: {scores["test_precision"]}\n')
        f.write(f'Mean precision score: {scores["test_precision"].mean()}\n')
        f.write(f'Recall scores: {scores["test_recall"]}\n')
        f.write(f'Mean recall score: {scores["test_recall"].mean()}\n')
        f.write(f'F1 scores: {scores["test_f1"]}\n')
        f.write(f'Mean F1 score: {scores["test_f1"].mean()}\n')
        f.write(f'Test set precision: {precisions}\n')
        f.write(f'Test set mean precision: {np.mean(precisions)}\n')
        f.write(f'Test set best precision: {np.max(precisions)}\n')
        f.write(f'Test set recall: {recalls}\n')
        f.write(f'Test set mean recall: {np.mean(recalls)}\n')
        f.write(f'Test set best recall: {np.max(recalls)}\n')
        f.write(f'Test set F1: {f1s}\n')
        f.write(f'Test set mean F1: {np.mean(f1s)}\n')
        f.write(f'Test set best F1: {np.max(f1s)}\n')
        ##  Without blueprints...
        ['RoadType', 'RoadID', 'ScenarioLength', 'VehicleFront', 'VehicleAdjacent', 'VehicleOpposite', 'VehicleFrontTwoWheeled', 'VehicleAdjacentTwoWheeled', 'VehicleOppositeTwoWheeled', 'TimeOfDay', 'Weather', 'NumberOfPeople', 'TargetSpeed', 'Trees', 'Buildings', 'Task']
        if platform == 'PID':
            df = df_main.drop(['Ego', 'NonEgo'], axis=1)
            train = df_train.drop(['Ego', 'NonEgo'], axis=1)
            test = df_test.drop(['Ego', 'NonEgo'], axis=1)
        elif platform == 'Pylot':
            df = df_main.drop(['RoadType', 'RoadID'], axis=1)
            train = df_train.drop(['RoadType', 'RoadID'], axis=1)
            test = df_test.drop(['RoadType', 'RoadID'], axis=1)
        if approach == 'dt':
            scores, desc = trainDecisionTree(df, cv=cv, **kwargs)
        elif approach == 'rf':
            scores, desc = trainRandomForest(train, cv=cv, **kwargs)
        elif approach == 'svm':
            scores, desc = trainSVM(df, cv=cv, **kwargs)
        elif approach == 'mlp':
            scores, desc = trainMLP(df, cv=cv, **kwargs)
        models = scores['estimator']
        predictions = [model.predict(test.iloc[:, :-1].values) for model in models]
        precisions = [precision_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        recalls = [recall_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        f1s = [f1_score(test.iloc[:, -1].values, prediction) for prediction in predictions]

        results_dict['Method'].append(approach)
        results_dict['Input'].append('Without road type and id' + kwargs['input'])
        results_dict['Precision'].append(np.max(precisions))
        results_dict['Recall'].append(np.max(recalls))
        results_dict['F1'].append(np.max(f1s))
        f.write('*' * 25 + '  Without road type and id  ' + '*' * 25 + '\n')
        f.write(f'Approach: {desc}\n')
        f.write(f'Inputs: {list(df.columns[:-1])}\n')
        f.write(f'Output: {list(df.columns[-1])}\n') 
        f.write(f'Number of fitnesses included in input: {flen}\n')
        f.write(f'Precision scores: {scores["test_precision"]}\n')
        f.write(f'Mean precision score: {scores["test_precision"].mean()}\n')
        f.write(f'Recall scores: {scores["test_recall"]}\n')
        f.write(f'Mean recall score: {scores["test_recall"].mean()}\n')
        f.write(f'F1 scores: {scores["test_f1"]}\n')
        f.write(f'Mean F1 score: {scores["test_f1"].mean()}\n')
        f.write(f'Test set precision: {precisions}\n')
        f.write(f'Test set mean precision: {np.mean(precisions)}\n')
        f.write(f'Test set best precision: {np.max(precisions)}\n')
        f.write(f'Test set recall: {recalls}\n')
        f.write(f'Test set mean recall: {np.mean(recalls)}\n')
        f.write(f'Test set best recall: {np.max(recalls)}\n')
        f.write(f'Test set F1: {f1s}\n')
        f.write(f'Test set mean F1: {np.mean(f1s)}\n')
        f.write(f'Test set best F1: {np.max(f1s)}\n')
        ##  Without weather and blueprints...
        if platform == 'PID':
            df = df_main.drop(['Ego', 'NonEgo', 'Weather1', 'Weather2', 'Weather3'], axis=1)
            train = df_train.drop(['Ego', 'NonEgo', 'Weather1', 'Weather2', 'Weather3'], axis=1)
            test = df_test.drop(['Ego', 'NonEgo', 'Weather1', 'Weather2', 'Weather3'], axis=1)
        elif platform == 'Pylot':
            df = df_main.drop(['Weather', 'RoadType', 'RoadID'], axis=1)
            train = df_train.drop(['Weather', 'RoadType', 'RoadID'], axis=1)
            test = df_test.drop(['Weather', 'RoadType', 'RoadID'], axis=1)
        if approach == 'dt':
            scores, desc = trainDecisionTree(train, cv=cv, **kwargs)
        elif approach == 'rf':
            scores, desc = trainRandomForest(train, cv=cv, **kwargs)
        elif approach == 'svm':
            scores, desc = trainSVM(train, cv=cv, **kwargs)
        elif approach == 'mlp':
            scores, desc = trainMLP(train, cv=cv, **kwargs)
        models = scores['estimator']
        predictions = [model.predict(test.iloc[:, :-1].values) for model in models]
        precisions = [precision_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        recalls = [recall_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
        f1s = [f1_score(test.iloc[:, -1].values, prediction) for prediction in predictions]

        results_dict['Method'].append(approach)
        results_dict['Input'].append('Without weather and road type and id' + kwargs['input'])
        results_dict['Precision'].append(np.max(precisions))
        results_dict['Recall'].append(np.max(recalls))
        results_dict['F1'].append(np.max(f1s))
        f.write('*' * 25 + '  Without weather and and road type and id' + '*' * 25 + '\n')
        f.write(f'Approach: {desc}\n')
        f.write(f'Inputs: {list(df.columns[:-1])}\n')
        f.write(f'Output: {list(df.columns[-1])}\n')
        f.write(f'Number of fitnesses included in input: {flen}\n')
        f.write(f'Precision scores: {scores["test_precision"]}\n')
        f.write(f'Mean precision score: {scores["test_precision"].mean()}\n')
        f.write(f'Recall scores: {scores["test_recall"]}\n')
        f.write(f'Mean recall score: {scores["test_recall"].mean()}\n')
        f.write(f'F1 scores: {scores["test_f1"]}\n')
        f.write(f'Mean F1 score: {scores["test_f1"].mean()}\n')
        f.write(f'Test set precision: {precisions}\n')
        f.write(f'Test set mean precision: {np.mean(precisions)}\n')
        f.write(f'Test set best precision: {np.max(precisions)}\n')
        f.write(f'Test set recall: {recalls}\n')
        f.write(f'Test set mean recall: {np.mean(recalls)}\n')
        f.write(f'Test set best recall: {np.max(recalls)}\n')
        f.write(f'Test set F1: {f1s}\n')
        f.write(f'Test set mean F1: {np.mean(f1s)}\n')
        f.write(f'Test set best F1: {np.max(f1s)}\n')

    results_df = pd.DataFrame(results_dict)
    return results_df

def visualizeNumberOfClasses(df):
    sns.set_theme()
    plt.subplots(1, 5, figsize=(20, 5), layout='tight')
    plt.subplot(1, 5, 1)
    sns.countplot(x='Y0', data=df)
    plt.xlabel('Fitness 1')
    plt.subplot(1, 5, 2)
    sns.countplot(x='Y1', data=df)
    plt.xlabel('Fitness 2')
    plt.subplot(1, 5, 3)
    sns.countplot(x='Y2', data=df)
    plt.xlabel('Fitness 3')
    plt.subplot(1, 5, 4)
    sns.countplot(x='Y3', data=df)
    plt.xlabel('Fitness 4')
    plt.subplot(1, 5, 5)
    sns.countplot(x='Y', data=df)
    plt.xlabel('Overall')
    plt.show()
    

if __name__  == '__main__':
    df_main = pd.read_csv('dataset.csv')
    df_main = df_main.sample(frac=1, random_state=42).reset_index(drop=True)
    df_delta = pd.read_csv('dataset_delta.csv')
    df_delta = df_delta.sample(frac=1, random_state=42).reset_index(drop=True)

    
    ##  Visualize number of classes...
    visualizeNumberOfClasses(df_main)
    visualizeNumberOfClasses(df_delta)
    df_main = df_main.drop(['Y0', 'Y1', 'Y2', 'Y3'], axis=1)
    df_delta = df_delta.drop(['Y0', 'Y1', 'Y2', 'Y3'], axis=1)

    os.remove('rq3.txt') if os.path.exists('rq3.txt') else None
    approaches = ['dt', 'rf', 'svm', 'mlp']
    results_df = pd.DataFrame(columns=['Method', 'Input', 'Precision', 'Recall', 'F1'])
    for approach in approaches:
        ##  With 1 set of fitnesses...
        df = df_main.copy()
        res = train(df, cv=5, approach=approach, output_file='rq3.txt', max_depth=5, fitnesses_len=1, input=', with 1 set of fitnesses')
        results_df = results_df.append(res, ignore_index=True)
        ##  With delta fitnesses...
        df = df_delta.copy()
        res = train(df, cv=5, approach=approach, output_file='rq3.txt', max_depth=5, fitnesses_len=2, input=', with delta fitnesses')
        results_df = results_df.append(res, ignore_index=True)

    results_df['Method'] = results_df['Method'].map({'dt': 'Decision Tree', 'svm': 'SVM', 'mlp': 'MLP', 'rf':' Random Forest'})
    results_df.sort_values(by=['F1'], ascending=False, inplace=True)
    results_df.to_csv('rq3_models.csv', index=False)
    results_df.to_latex('tab-rq3-models.tex', index=False, escape=False, float_format=lambda x: '%.2f' % x)
