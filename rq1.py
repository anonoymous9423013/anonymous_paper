from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
import logging as log
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import utils 
import time

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

min_legend, mean_legend = 'RS(n=10)', 'RS(n=1)'
fitness_labels = ['F1', 'F2', 'F3', 'F4']
no_of_fitnesses = len(fitness_labels)
    
def randomSearchOneRun(f_arr):
    fit_min = [f_arr[0, 0].tolist()]
    fit_mean = [f_arr[0, 1].tolist()]
    for i in range(1, len(f_arr)):
        fit_min_prime = f_arr[i, 0].tolist()
        fit_mean_prime = f_arr[i, 1].tolist()
        fit_min.append([min(fit_min[-1][j], fit_min_prime[j]) for j in range(f_arr.shape[-1])])
        fit_mean.append([min(fit_mean[-1][j], fit_mean_prime[j]) for j in range(f_arr.shape[-1])])
    fit_min = np.array(fit_min)
    fit_mean = np.array(fit_mean)
    return fit_min, fit_mean 
        
def randomSearch(f_arr, runs=20):
    n_iterations = len(f_arr) // runs
    fit_min = []
    fit_mean = []
    t0 = time.time()
    for i in range(runs):
        fit_min_prime, fit_mean_prime = randomSearchOneRun(f_arr[i*n_iterations:(i+1)*n_iterations])
        fit_min.append(fit_min_prime)
        fit_mean.append(fit_mean_prime)
    fit_min = np.array(fit_min)
    fit_mean = np.array(fit_mean)
    t1 = time.time()
    log.info(f'Random search with {runs} runs took {t1-t0} seconds...')
    return fit_min, fit_mean

def plotRS(f_min, f_mean, show=True, platform='PID'):
    output_file = f'rs_iters_{platform}.pdf'
    t0 = time.time()
    df_min_dict = {'Iteration': [], **{f: [] for f in fitness_labels}}
    df_mean_dict = {'Iteration': [], **{f: [] for f in fitness_labels}}
    for i in range(len(f_min)):
        for iter in range(len(f_min[i])):
            df_min_dict['Iteration'].append(iter)
            df_mean_dict['Iteration'].append(iter)
            for j in range(f_min.shape[-1]):
                df_min_dict[fitness_labels[j]].append(f_min[i, iter, j])
                df_mean_dict[fitness_labels[j]].append(f_mean[i, iter, j])

    df_min = pd.DataFrame(df_min_dict)
    df_mean = pd.DataFrame(df_mean_dict)
    
    loc, fontsize = 'lower left', 8
    sns.set()
    plt.subplots(nrows=1,ncols=no_of_fitnesses,subplot_kw=dict(box_aspect=1), figsize=(24,5), layout='constrained')
    for i in range(no_of_fitnesses):
        plt.subplot(1, no_of_fitnesses, i+1)
        sns.lineplot(data=df_mean, x='Iteration', y=fitness_labels[i], label=mean_legend)
        sns.lineplot(data=df_min, x='Iteration', y=fitness_labels[i], label=min_legend)
        plt.xlabel('Iteration')
        plt.ylabel(fitness_labels[i])
        plt.legend(loc=loc, fontsize=fontsize)
    
    plt.savefig(output_file, bbox_inches='tight')
    t1 = time.time()
    log.info(f'Plotting random search fitnesses took {t1-t0} seconds. Output file: {output_file}')
    plt.show() if show else None

def plotBox(f_min, f_mean, show=True, platform='PID'):
    output_file = f'rs_boxplot_{platform}.pdf'
    t0 = time.time()
    df_box_dict = {'Method': [], **{f: [] for f in fitness_labels}}
    for i in range(len(f_min)):
        df_box_dict['Method'].append('RSwREP')
        df_box_dict['Method'].append('RS')
        for j in range(f_min.shape[-1]):
            df_box_dict[fitness_labels[j]].append(f_min[i, -1, j])
            df_box_dict[fitness_labels[j]].append(f_mean[i, -1, j])
    df_box = pd.DataFrame(df_box_dict)

    shining_green = '#00FF00'  ##  For displaying mean
    sns.set()
    plt.subplots(nrows=1, ncols=no_of_fitnesses, subplot_kw=dict(box_aspect=1), figsize=(24,5), layout='constrained')
    for i in range(no_of_fitnesses):
        plt.subplot(1, no_of_fitnesses, i+1)
        sns.boxplot(data=df_box, x='Method', y=fitness_labels[i], hue='Method', showmeans=True, meanprops={'marker':'o', 'markerfacecolor':shining_green, 'markeredgecolor':shining_green})
        plt.ylabel(fitness_labels[i])
        plt.legend([], [], frameon=False)
    plt.savefig(output_file, bbox_inches='tight')
    t1 = time.time()
    log.info(f'Plotting boxplots took {t1-t0} seconds. Output file: {output_file}')
    plt.show() if show else None
    
def variationsTable(f_arr_detailed:np.ndarray, thresholds=(-1, 0, 0.5), platform='PID'):
    output_file = f'table_variations_{platform}.txt'
    t0 = time.time()
    deltas = np.max(f_arr_detailed, axis=1) - np.min(f_arr_detailed, axis=1)  ##  (no_of_scenarios, no_of_fitnesses)
    delta_max = np.max(deltas, axis=0)  ##  (no_of_fitnesses,)
    no = [[0 for _ in range(no_of_fitnesses)] for _ in range(len(thresholds) - 1)]
    for i in range(1, len(thresholds)):
        for j in range(no_of_fitnesses):
            no[i-1][j] = np.sum((deltas[:, j] / delta_max[j] > thresholds[i-1]) & (deltas[:, j] / delta_max[j] <= thresholds[i]))
            
    no = np.array(no)  ##  (no_of_thresholds-1, no_of_fitnesses)
    log.debug(f'Sum of numbers of each column for variation table: {np.sum(no, axis=0)}')
    ##  Generate variation_table.csv...
    with open(output_file, 'w') as f:
        ##  Readable mode...
        for i in range(no_of_fitnesses):
            for j in range(len(thresholds) - 1):
                f.write(f'Threshold: ({thresholds[j]:.2f}, {thresholds[j+1]:.2f}] for {fitness_labels[i]}:\t{no[j, i] / len(f_arr_detailed) * 100:3.1f}% ({no[j, i]})\n')
        ##  LateX mode...
        f.write('LateX mode...\n\n\n')
        s = '\\multirow{2}{*}{' + platform + '} '
        for i in range(no_of_fitnesses):
            s += f'& \emph{{{fitness_labels[i]}}}  '
            for j in range(len(thresholds) - 1):
                s += f'& {no[j, i] / len(f_arr_detailed) * 100:3.1f}\\% ({no[j, i]})  '
            if i < no_of_fitnesses - 1:
                s += '\\\\ \\cline{2-11} \n'
            else:
                s += '\\\\ \\hline \n'
        f.write(s)
        
    t1 = time.time()
    log.info(f'Generating variation table took {t1-t0} seconds. Output file: {output_file}')

def failureTable(f_arr_detailed, thresholds, platform='PID'):
    output_file = f'table_failure_{platform}.txt'
    t0 = time.time()
    no = [[0 for _ in range(no_of_fitnesses)] for _ in range(len(thresholds))]
    
    for i in range(len(thresholds)):
        for j in range(no_of_fitnesses):
            no[i][j] = np.sum(np.any(f_arr_detailed[:, :, j] < thresholds[i][j], axis=1))
            
    no = np.array(no)  ##  (no_of_thresholds, no_of_fitnesses)

    ##  Generate failure_table.csv...
    with open(output_file, 'w') as f:
        ##  Readable mode...
        for i in range(no_of_fitnesses):
            for j in range(len(thresholds)):
                f.write(f'Threshold {thresholds[j][i]:4.3f} for {fitness_labels[i]}:\t{no[j, i]}\n')
        ##  LateX mode...
        f.write('LateX mode...\n\n\n')
        s = '\\multirow{2}{*}{CARLA} '
        for i in range(no_of_fitnesses):
            s += f'& \emph{{{fitness_labels[i]}}}  '
            for j in range(len(thresholds)):
                s += f'& {no[j, i]}'
            if i < no_of_fitnesses - 1:
                s += '\\\\ \\cline{2-5} \n'
            else:
                s += '\\\\ \\hline \n'
        f.write(s)
    t1 = time.time()
    log.info(f'Generating failure table took {t1-t0} seconds. Output file: {output_file}...')

def statisticalTests(f_min, f_mean, platform='PID'):
    output_file = f'statistical_tests_{platform}.txt'
    t0 = time.time()
    def a12(lst1,lst2,rev=True):
        more = same = 0.0
        for x in lst1:
            for y in lst2:
                if   x==y : same += 1
                elif rev     and x > y : more += 1
                elif not rev and x < y : more += 1
        return (more + 0.5*same)  / (len(lst1)*len(lst2))
    f_min_mean = np.mean(f_min, axis=0)  ##  (no_of_iterations, no_of_fitnesses)
    f_mean_mean = np.mean(f_mean, axis=0)  ##  (no_of_iterations, no_of_fitnesses)
    ##  Wilcoxon signed-rank test...
    res = [wilcoxon(f_min_mean[:, i], f_mean_mean[:, i]) for i in range(no_of_fitnesses)]
    p_values = [r.pvalue for r in res]
    ##  Vargha-Delaney A^12 test...
    a12_values = [a12(f_mean_mean[:, i], f_min_mean[:, i]) for i in range(no_of_fitnesses)]
    with open(output_file, 'w') as f:
        f.write(f'p-values: {p_values}\n')
        f.write(f'A^12 values: {a12_values}\n')
    t1 = time.time()
    log.info(f'Statistical tests took {t1-t0} seconds. Results are in {output_file}...')

    
if __name__  == '__main__':
    platform = 'Pylot'
    result_detailed_beam, f_arr_detailed_beam = utils.fitnessForAllScenarios(platform=platform, csv_file='ds_pylot.csv', normalize=False)  ##  f_arr_detailed.shape = (no_of_scenarios, no_of_repeats, no_of_fitnesses)
    scenarios_beam, f_arr_beam = utils.minAndMeanFitnesses(result_detailed_beam, normalize=True)
    f_min_beam, f_mean_beam = randomSearch(f_arr_beam)  ##  (no_of_runs, no_of_iterations, no_of_fitnesses) for both

    df_beam = pd.DataFrame(f_arr_detailed_beam.reshape(-1, f_arr_detailed_beam.shape[-1]), columns=['F1', 'F2', 'F3', 'F4'])
    ##  Visualize df statistics as a table using seaborn...
    plt.figure()
    sns.set_theme()
    sns.heatmap(df_beam.describe()[1:].transpose(), annot=True, fmt='.2f')
    plt.title('Statistics of fitnesses')
    plt.savefig(f'fitness_statistics_{platform}.pdf')
    # plt.show()
    
    # ##  Plot random search fitnesses mean over runs...
    plotRS(f_min_beam, f_mean_beam, show=True, platform=platform)

    # ##  Plot boxplots...
    plotBox(f_min_beam, f_mean_beam, show=True, platform=platform)

    # ##  Variations table...
    variation_thresholds = (-1, 0, 0.01, 0.05, 0.1, 0.4, 0.7, 1.)
    variationsTable(f_arr_detailed_beam, variation_thresholds, platform=platform)
    
    # ##  Failure table...
    fitness_thresholds = np.array([-1.5, -90, -1000, -10])
    fitness_thresholds_beam = np.array([-1.5, -90, -1000, -10])
    tolerance = 0.1
    failure_thresholds_beam = np.array([[t + tolerance * abs(t) * i for i in range(-1, 2)] for t in fitness_thresholds]).T
    # min_ = np.min(f_arr_detailed, axis=(0, 1))
    # max_ = np.max(f_arr_detailed, axis=(0, 1))
    # fitness_thresholds_normalized = (failure_thresholds - min_) / (max_ - min_) * 2 - 1
    log.info('Failure thresholds:')
    print(failure_thresholds_beam)
    failureTable(f_arr_detailed_beam, failure_thresholds_beam, platform=platform)

    ##  Wilcoxon signed-rank test and Vargha-Delaney A measure...
    statisticalTests(f_min_beam, f_mean_beam, platform=platform)
        