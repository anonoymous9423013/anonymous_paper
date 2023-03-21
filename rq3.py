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
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import rq1

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# min_legend, mean_legend = 'RSwREP', 'RS'
min_legend, mean_legend = 'RS(n=10)', 'RS(n=1)'
fitness_labels = ['F1', 'F2', 'F3', 'F4']
no_of_fitnesses = len(fitness_labels)
df_main = pd.read_csv('dataset.csv')
probs = [len(df_main[df_main['Y'] == 0]) / len(df_main), len(df_main[df_main['Y'] == 1]) / len(df_main)]
fitness_threshes = np.array([-1.5, -90, -1000, -10])
log.info(f'Proportions of classes: {probs}')


def trainDecisionTree(df, cv=5, **kwargs):
    max_depth = kwargs.get('max_depth', 5)
    model = DecisionTreeClassifier(max_depth=max_depth)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(model, X, y, cv=cv, scoring=['precision', 'recall', 'f1'], return_estimator=True)
    desc = 'Decision Tree'
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

def trainModels(df_main, df_delta, cv=5, **kwargs):
    models = []
    df_main = df_main.drop(['Ego', 'NonEgo'], axis=1)  ##  Drop the ego and non-ego columns
    df_delta = df_delta.drop(['Ego', 'NonEgo'], axis=1)  ##  Drop the ego and non-ego columns
    if 'Timestep' in df_main.columns:
        df_main = df_main.drop(['Timestep'], axis=1)
    ##  Decision Tree with 1 set of fitnesses...
    df_train = df_main.sample(frac=1., random_state=200)
    df_train = balance(df_train, output_file=None)
    scores, desc = trainDecisionTree(df_train, cv=cv, **kwargs)
    f1s = scores["test_f1"]
    models.append(scores['estimator'][np.argmax(f1s)])

    ##  Decision Tree with delta fitnesses...
    df_train = df_delta.sample(frac=1., random_state=200)
    df_train = balance(df_train, output_file=None)
    scores, desc = trainMLP(df_train, cv=cv, **kwargs)
    f1s = scores["test_f1"]
    models.append(scores['estimator'][np.argmax(f1s)])

    return models

def plotRS(f_mins, f_mean, labels, show=True):
    """Plots the random search results in iterations...

    Args:
        f_mins (list): 
        f_mean (_type_): _description_
        labels (list): The first elements are labels for f_mins, the last element is for f_mean.
        show (bool, optional): _description_. Defaults to True.
    """
    output_file = 'rss_pid_iters.pdf'
    t0 = time.time()
    df_mins_dict = [{'Iteration': [], **{fitness_labels[j]: [] for j in range(len(fitness_labels))}} for i in range(len(f_mins))]
    df_mean_dict = {'Iteration': [], **{fitness_labels[j]: [] for j in range(len(fitness_labels))}}
    ##  f_mean...
    for i in range(len(f_mean)):
        for iter in range(len(f_mean[i])):
            df_mean_dict['Iteration'].append(iter)
            for j in range(f_mean.shape[-1]):
                df_mean_dict[fitness_labels[j]].append(f_mean[i, iter, j])
    ##  f_mins...
    for k in range(len(f_mins)):
        for i in range(len(f_mins[k])):
            for iter in range(len(f_mins[k][i])):
                df_mins_dict[k]['Iteration'].append(iter)
                for j in range(f_mins[k].shape[-1]):
                    df_mins_dict[k][fitness_labels[j]].append(f_mins[k][i, iter, j])


    df_mins = [pd.DataFrame(df_min_dict) for df_min_dict in df_mins_dict]
    df_mean = pd.DataFrame(df_mean_dict)
    
    loc, fontsize = 'upper right', 12
    sns.set(font_scale=2.2)
    fig, ax = plt.subplots(nrows=1,ncols=no_of_fitnesses,subplot_kw=dict(box_aspect=1), figsize=(24,5), layout='constrained')
    for i in range(no_of_fitnesses):
        plt.subplot(1, no_of_fitnesses, i+1)
        sns.lineplot(data=df_mean, x='Iteration', y=fitness_labels[i], label=labels[-1])
        for j in range(len(df_mins)):
            sns.lineplot(data=df_mins[j], x='Iteration', y=fitness_labels[i], label=labels[j])
        plt.xlabel('Iteration')
        plt.ylabel('')
        plt.title(fitness_labels[i])
        plt.legend(loc=loc, fontsize=fontsize)
        if i == 1:
            # plt.setp(ax[i].get_legend().get_texts(), fontsize='15') # for legend text
            # plt.setp(ax[i].get_legend().get_title(), fontsize='15') # for legend title
            pass
        else:
            ax[i].get_legend().remove()
            pass

    
    plt.savefig(output_file, bbox_inches='tight')
    t1 = time.time()
    log.info(f'Plotting random search fitnesses took {t1-t0} seconds. Output file: {output_file}')
    plt.show() if show else None

def plotBox(f_mins, f_mean, labels, show=True):
    output_file = 'rss_pid_box.pdf'
    t0 = time.time()
    df_box_dict_mins = [{'Method': [], **{fitness_labels[j]: [] for j in range(len(fitness_labels))}} for i in range(len(f_mins))]
    df_box_dict_mean = {'Method': [], **{fitness_labels[j]: [] for j in range(len(fitness_labels))}}
    for i in range(len(f_mean)):
        df_box_dict_mean['Method'].append('RS')
        for j in range(f_mean.shape[-1]):
            df_box_dict_mean[fitness_labels[j]].append(f_mean[i, -1, j])
    df_box_mean = pd.DataFrame(df_box_dict_mean)
    for k in range(len(f_mins)):
        for i in range(len(f_mins[k])):
            df_box_dict_mins[k]['Method'].append(labels[k])
            for j in range(f_mins[k].shape[-1]):
                df_box_dict_mins[k][fitness_labels[j]].append(f_mins[k][i, -1, j])
    df_box_mins = [pd.DataFrame(df_box_dict_min) for df_box_dict_min in df_box_dict_mins]
    df_box = pd.concat(df_box_mins+[df_box_mean], ignore_index=True)

    shining_green = '#00FF00'  ##  For displaying mean
    sns.set(font_scale=2.2)
    fig, ax = plt.subplots(nrows=1, ncols=no_of_fitnesses, subplot_kw=dict(box_aspect=1), figsize=(8,12), layout='constrained')
    for i in range(no_of_fitnesses):
        plt.subplot(no_of_fitnesses, 1, i+1)
        boxplot = sns.boxplot(data=df_box, x='Method', y=fitness_labels[i], hue='Method', showmeans=True, meanprops={'marker':'o', 'markerfacecolor':shining_green, 'markeredgecolor':shining_green}, dodge=False)
        plt.title(fitness_labels[i])
        plt.ylabel('')
        plt.xlabel('')
        plt.legend([], [], frameon=False)
        plt.xticks(rotation=90, fontsize=8)
        ax[i].set_xticklabels(ax[i].get_xticklabels(), fontsize=8)
    plt.savefig(output_file, bbox_inches='tight')
    t1 = time.time()
    log.info(f'Plotting boxplots took {t1-t0} seconds. Output file: {output_file}')
    plt.show() if show else None

def smartRandomSearchOneIteration(scenario, f_arr_detailed_iter, models=None, method='or', platform='PID'):
    if platform == 'PID':
        scenario = list(map(int, scenario[8:].split(',')))
    elif platform == 'Pylot':
        pass
    dropped = [3, 4]  ##  RoadType and RoadID...
    scenario = [scenario[i] for i in range(len(scenario)) if i not in dropped]
    cnt = 1
    X = np.array([scenario])
    Xfitness = np.concatenate((X[0], f_arr_detailed_iter[0]))[np.newaxis, :]  ##  1 set of fitnesses...
    ys = []
    ps = []
    if isinstance(models, list):  ##  Machine learning methods...
        ps.append(models[0].predict(Xfitness)[0] > 0.5)
    elif isinstance(models, str) and models == 'thresh':
        pass
    else:
        ps.append(np.random.rand() > probs[0])
    for i in range(2, 10):  ##  max number of time steps we want to proceed...
        deltas = np.max(f_arr_detailed_iter[:i], axis=0) - np.min(f_arr_detailed_iter[:i], axis=0)
        Xdelta = np.concatenate((X[0], deltas))[np.newaxis, :]
        ys.append(f_arr_detailed_iter[i])
        if isinstance(models, list):  ##  Machine learning methods...
            ps.append(models[1].predict(Xdelta)[0] > 0.5)
        elif isinstance(models, str) and models == 'thresh':
            # flag = (np.max(f_arr_detailed_iter[:i], axis=0) > fitness_threshes) & (np.min(f_arr_detailed_iter[:i], axis=0) < fitness_threshes)
            flag = deltas >= delta_max * 0.05
            if np.any(flag):
                cnt -=- 1
                break
        else:
            ps.append(np.random.rand() > probs[0])
        cnt -=- 1
        if method == 'and' and not all(ps):
            break
        elif method == 'or' and not any(ps):
            break
    f_min = np.min(ys, axis=0)
    f_mean = np.mean(ys, axis=0)
    return f_min, f_mean, cnt
    

def smartRandomSearchOneRun(scenarios, f_arr_detailed, models=None, method='or', platform='PID'):
    f_min, f_mean, cnt = smartRandomSearchOneIteration(scenarios[0], f_arr_detailed[0], models, method, platform)
    f_mins = [f_min]
    f_means = [f_mean]
    cnts = [cnt]
    for i in range(1, len(scenarios)):
        f_min, f_mean, cnt = smartRandomSearchOneIteration(scenarios[i], f_arr_detailed[i], models, method, platform)
        f_mins.append([min(f_min[j], f_mins[-1][j]) for j in range(len(f_min))]) 
        f_means.append([min(f_mean[j], f_means[-1][j]) for j in range(len(f_mean))])
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)


def smartRandomSearch(scenarios, f_arr_detailed, runs=20, models=None, method='or', platform='PID'):
    n_iterations = len(f_arr_detailed) // runs
    f_mins = []
    f_means = []
    cnts = []
    for i in range(runs):
        f_mins_prime, f_means_prime, cnt = smartRandomSearchOneRun(scenarios[i*n_iterations:(i+1)*n_iterations], f_arr_detailed[i*n_iterations:(i+1)*n_iterations], models, method, platform)
        f_mins.append(f_mins_prime)
        f_means.append(f_means_prime)
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)

def statisticalTests(f_mins, baseline, f_mins_labels, baseline_label=None, cnts=None, baseline_cnt=None, csv_output_file='statistical_tests_rq3.txt', tex_output_file='tab-rq3-stattest.tex', pdf_output_file='tab-rq3-stattest.pdf', show=False):
    t0 = time.time()
    def a12(lst1,lst2,rev=True):
        more = same = 0.0
        for x in lst1:
            for y in lst2:
                if   x==y : same += 1
                elif rev     and x > y : more += 1
                elif not rev and x < y : more += 1
        return (more + 0.5*same)  / (len(lst1)*len(lst2))
    f_mins_mean = [np.mean(f_min, axis=0) for f_min in f_mins]  ##  list of shapes (no_of_iterations, no_of_fitnesses)
    baseline_mean = np.mean(baseline, axis=0)  ##  shape (no_of_iterations, no_of_fitnesses)
    ##  Wilcoxon signed-rank test...
    res = [[wilcoxon(baseline_mean[:, i], f_min[:, i]) for i in range(no_of_fitnesses)] for f_min in f_mins_mean]
    p_values = [[r.pvalue for r in q] for q in res]
    ##  Vargha-Delaney A^12 test...
    a12_values = [[a12(f_min[:, i], baseline_mean[:, i]) for i in range(no_of_fitnesses)] for f_min in f_mins_mean]
    ##  Writing results to file...
    df_dict = {'Method': [], 'Simulations': []}
    df_dict = {**df_dict, **{f'F{i+1} p-value': [] for i in range(no_of_fitnesses)}}
    df_dict = {**df_dict, **{f'F{i+1} A^12': [] for i in range(no_of_fitnesses)}}
    for i in range(len(f_mins)):
        df_dict['Method'].append(f_mins_labels[i])
        df_dict['Simulations'].append(cnts[i])
        for j in range(no_of_fitnesses):
            df_dict[f'F{j+1} p-value'].append(p_values[i][j])
            df_dict[f'F{j+1} A^12'].append(a12_values[i][j])
    df = pd.DataFrame(df_dict)
    df.to_csv(csv_output_file, index=False)
    df.to_latex(tex_output_file, index=False, escape=False, column_format='l' + 'c' * (2 * no_of_fitnesses))

    ##  Plotting results...
    sns.set(font_scale=2.2)
    plt.subplots(2, 4, figsize=(20, 10), layout='constrained')
    for i in range(no_of_fitnesses):
        plt.subplot(2, 4, i+1)    
        plt.title(f'Wilcoxon signed-rank test for {fitness_labels[i]}')
        plt.xlabel('p-value')
        plt.ylabel('No of simulations')
        for j in range(len(f_mins)):
            plt.scatter(p_values[j][i], cnts[j], marker='o')
            plt.text(p_values[j][i], cnts[j], f_mins_labels[j], fontsize=8)
        plt.subplot(2, 4, i+5)
        plt.title(f'Vargha-Delaney A^12 test for {fitness_labels[i]}')
        plt.xlabel('A^12')
        plt.ylabel('No of simulations')
        for j in range(len(f_mins)):
            plt.scatter(a12_values[j][i], cnts[j], marker='o')
            plt.text(a12_values[j][i], cnts[j], f_mins_labels[j], fontsize=8)
    plt.savefig('rq3_statistical_tests.pdf', bbox_inches='tight')
    t1 = time.time()
    if show:
        plt.show()
    log.info(f'Statistical tests took {t1-t0} seconds. Results are in {csv_output_file}, {tex_output_file}, and {pdf_output_file}...')

def accuracyTable(f_mins, baseline, f_mins_labels, baseline_label=None, cnts=None, baseline_cnt=None, csv_output_file='rq3-acc.txt', tex_output_file='tab-rq3-acc.tex', pdf_output_file='tab-rq3-acc.pdf', show=False):
    t0 = time.time()
    f_mins_mean = [np.mean(f_min[:, -1, :], axis=0) for f_min in f_mins]  ##  list of shapes (no_of_fitnesses,)
    baseline_mean = np.mean(baseline[:, -1, :], axis=0)  ##  shape (no_of_fitnesses,)
    df_dict = {'Method': [], 'Simulations': []}
    df_dict = {**df_dict, **{f'F{i+1}': [] for i in range(no_of_fitnesses)}}
    ##  Baseline...
    df_dict['Method'].append(baseline_label)
    df_dict['Simulations'].append(baseline_cnt)
    for i in range(no_of_fitnesses):
        df_dict[f'F{i+1}'].append(baseline_mean[i])
    ##  Other methods...
    for i in range(len(f_mins)):
        df_dict['Method'].append(f_mins_labels[i])
        df_dict['Simulations'].append(cnts[i])
        for j in range(no_of_fitnesses):
            df_dict[f'F{j+1}'].append(f_mins_mean[i][j])
    df = pd.DataFrame(df_dict)
    for col in df.columns:
        if col != 'Method' and col != 'Simulations':
            df[col] = df[col].apply(lambda x: f'{x:.3f} ({x / df[col][0] * 100:.1f}\%)')
    df.to_csv(csv_output_file, index=False)
    df.to_latex(tex_output_file, index=False, escape=False, column_format='l' + 'c' * (no_of_fitnesses + 2))
    t1 = time.time()
    log.info(f'Accuracy table took {t1-t0} seconds. Results are in {csv_output_file}, {tex_output_file}, and {pdf_output_file}...')

def evalDeltaSteps(sceanrios, f_arr_detailed, models, platform='PID'):
    result_dict = {'Timestep': [], 'Baseline Presicion': [], 'Baseline Recall': [], 'Baseline F1': [], 'Model Presicion': [], 'Model Recall': [], 'Model F1': []}
    dropped = [3, 4]  ##  RoadType and RoadID...
    for t in range(2, 10):
        targets = []
        preds_model = []
        preds_thresh = []
        for i in range(len(scenarios)):
            if platform == 'PID':
                scenario = list(map(int, scenarios[i][8:].split(',')))
            elif platform == 'Pylot':
                scenario = scenarios[i]
            scenario = [scenario[i] for i in range(len(scenario)) if i not in dropped]
            X = np.array([scenario])
            f_arr = f_arr_detailed[i, :t, :]
            deltas = np.max(f_arr, axis=0) - np.min(f_arr, axis=0)
            Xdelta = np.concatenate((X[0], deltas))[np.newaxis, :]
            y_pred_model = (models[1].predict(Xdelta)[0] > 0.5).astype(int)
            y_pred_thresh = (deltas >= delta_max * 0.05).any().astype(int)
            deltas_target = np.max(f_arr_detailed[i], axis=0) - np.min(f_arr_detailed[i], axis=0)
            y_target = np.any(deltas_target >= delta_max * 0.05)
            targets.append(y_target)
            preds_model.append(y_pred_model)
            preds_thresh.append(y_pred_thresh)

        targets = np.array(targets)
        preds_model = np.array(preds_model)
        preds_thresh = np.array(preds_thresh)
        precision_model = precision_score(targets, preds_model)
        precision_thresh = precision_score(targets, preds_thresh)
        recall_model = recall_score(targets, preds_model)
        recall_thresh = recall_score(targets, preds_thresh)
        f1_model = f1_score(targets, preds_model)
        f1_thresh = f1_score(targets, preds_thresh)
        result_dict['Timestep'].append(t)
        result_dict['Baseline Presicion'].append(precision_thresh)
        result_dict['Baseline Recall'].append(recall_thresh)
        result_dict['Baseline F1'].append(f1_thresh)
        result_dict['Model Presicion'].append(precision_model)
        result_dict['Model Recall'].append(recall_model)
        result_dict['Model F1'].append(f1_model)
    df = pd.DataFrame(result_dict)
    df.to_latex('tab-rq3-delta.tex', index=False, escape=False, column_format='l' + 'c' * 6)
    ##  Plotting in 3 subplots...
    sns.set()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(df['Timestep'], df['Baseline Presicion'], 'x--', label='Baseline')
    axs[0].plot(df['Timestep'], df['Model Presicion'], 'x--', label='Model')
    axs[0].set_xlabel('Timestep')
    axs[0].set_ylabel('Precision')
    axs[0].legend()
    axs[1].plot(df['Timestep'], df['Baseline Recall'], 'x--', label='Baseline')
    axs[1].plot(df['Timestep'], df['Model Recall'], 'x--', label='Model')
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('Recall')
    axs[1].legend()
    axs[2].plot(df['Timestep'], df['Baseline F1'], 'x--', label='Baseline')
    axs[2].plot(df['Timestep'], df['Model F1'], 'x--', label='Model')
    axs[2].set_xlabel('Timestep')
    axs[2].set_ylabel('F1')
    axs[2].legend()
    plt.savefig('fig-rq3-delta.pdf')
    plt.show()
    log.info('Plot saved to fig-rq3-delta.pdf...')
            
            

if __name__  == '__main__':
    ##  Training models...
    df_main = pd.read_csv('dataset.csv')
    df_main = df_main.sample(frac=1, random_state=42).reset_index(drop=True)
    df_delta = pd.read_csv('dataset_delta.csv')
    df_delta = df_delta.sample(frac=1, random_state=42).reset_index(drop=True)
    df_main = df_main.drop(['Y0', 'Y1', 'Y2', 'Y3'], axis=1)
    df_delta = df_delta.drop(['Y0', 'Y1', 'Y2', 'Y3'], axis=1)
    models = trainModels(df_main, df_delta)
    log.info('Models trained...')
    assert len(models) == 2

    ##  Loading fitnesses...
    # files = glob('log/log_*.txt')
    # scenarios = list(set([f[:f.rfind('_')] for f in files]))
    # scenarios.sort()
    # scenarios = utils.preprocess(scenarios)
    files = glob('log/log_*.txt')
    scenarios = list(set([f[:f.rfind('_')] for f in files]))
    scenarios.sort()
    scenarios = utils.preprocess(scenarios)
    result_detailed, f_arr_detailed = utils.fitnessForAllScenarios(scenarios, normalize=False)
    deltas = np.max(f_arr_detailed, axis=1) - np.min(f_arr_detailed, axis=1)  ##  (no_of_scenarios, no_of_fitnesses)
    delta_max = np.max(deltas, axis=0)  ##  (no_of_fitnesses,)
    scenarios, __ = utils.minAndMeanFitnesses(result_detailed, normalize=True)
    # # result_detailed, f_arr_detailed = utils.fitnessForAllScenarios(scenarios, normalize=False)  ##  f_arr_detailed.shape = (n_scenarios, n_reps, n_fitnesses)
    # ##  Smart random search random mode...
    # f_min_baseline_manual, f_mean_baseline_manual, cnt_baseline_manual = smartRandomSearch(scenarios, f_arr_detailed, models='thresh', method='baseline',platform='PID')
    
    # ##  Smart random search random mode with or ...
    # f_min_smart_random_or, f_mean_smart_random_or, cnt_random_or = smartRandomSearch(scenarios, f_arr_detailed, models=None, method='or', platform='PID')

    # ##  Smart random search with models with or...
    # f_min_smart_or, f_mean_smart_or, cnt_or = smartRandomSearch(scenarios, f_arr_detailed, models=models, method='or', platform='PID')

    # ##  Smart random search random mode with and ...
    # f_min_smart_random_and, f_mean_smart_random_and, cnt_random_and = smartRandomSearch(scenarios, f_arr_detailed, models=None, method='and', platform='PID')

    # ##  Smart random search with models with and...
    # f_min_smart_and, f_mean_smart_and, cnt_and = smartRandomSearch(scenarios, f_arr_detailed, models=models, method='and', platform='PID')

    # ##  Random search for 10 repetitions...
    # scenarios, f_arr10 = utils.minAndMeanFitnesses(result_detailed, normalize=False, n_reps=10)
    # f_min10, f_mean10 = rq1.randomSearch(f_arr10)
    # min_ = np.min(f_arr10, axis=(0, 1))
    # max_ = np.max(f_arr10, axis=(0, 1))
    # f_min10 = (f_min10 - min_) / (max_ - min_)
    # f_mean10 = (f_mean10 - min_) / (max_ - min_)

    # ##  Normalizing fitnesses for smart random search...
    # f_min_baseline_manual = (f_min_baseline_manual - min_) / (max_ - min_)
    # f_min_smart_or = (f_min_smart_or - min_) / (max_ - min_)
    # f_min_smart_random_or = (f_min_smart_random_or - min_) / (max_ - min_)
    # f_min_smart_and = (f_min_smart_and - min_) / (max_ - min_)
    # f_min_smart_random_and = (f_min_smart_random_and - min_) / (max_ - min_)

    # plotRS([f_min_baseline_manual, f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or, f_min10], f_mean10, ['Baseline-Manual', 'Baseline-AND', 'RSO-AND', 'Baseline-OR', 'RSO-OR', 'RS(n=10)', 'RS'], show=True)
    # plotBox([f_min_baseline_manual, f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or, f_min10], f_mean10, ['Baseline-Manual', 'Baseline-AND', 'RSO-AND', 'Baseline-OR', 'RSO-OR', 'RS(n=10)', 'RS'], show=True)

    # log.info(f'Number of iterations for baseline manual: {cnt_baseline_manual}')
    # log.info(f'Number of iterations for smart RS in random mode OR: {cnt_random_or}')
    # log.info(f'Number of iterations for smart RS with models OR: {cnt_or}')
    # log.info(f'Number of iterations for smart RS in random mode AND: {cnt_random_and}')
    # log.info(f'Number of iterations for smart RS with models AND: {cnt_and}')

    # cnts = [cnt_baseline_manual, cnt_random_and, cnt_and, cnt_random_or, cnt_or, 10000]
    # cnt_baseline = 10000
    # statisticalTests([f_min_baseline_manual, f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or], f_mean10, ['Baseline-Manual', 'Baseline-AND', 'RSO-AND', 'Baseline-OR', 'RSO-OR'], 'RS(n=10)', cnts, cnt_baseline)
    
    # accuracyTable([f_min_baseline_manual, f_min_smart_random_and, f_min_smart_and, f_min_smart_random_or, f_min_smart_or], f_min10, ['Baseline-Manual', 'Baseline-AND', 'RSO-AND', 'Baseline-OR', 'RSO-OR'], 'RS(n=10)', cnts, cnt_baseline, csv_output_file='rq3-acc-table.csv', tex_output_file='tab-rq3-acc.tex')
    evalDeltaSteps(scenarios, f_arr_detailed, models)