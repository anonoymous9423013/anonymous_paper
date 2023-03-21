import utils 
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
import logging as log
import pandas as pd
from random import shuffle
import os
from scipy.stats import wilcoxon
from multiprocessing import Pool
import time


if __name__  == '__main__':
    files = glob('log/log_*.txt')
    scenarios = list(set([f[:f.rfind('_')] for f in files]))
    result_detailed, f_arr_detailed = utils.fitnessForAllScenarios(scenarios, normalize=True)
    scenarios, f_arr = utils.minAndMeanFitnesses(result_detailed)
    flaky_scenarios, flaky_arr, flaky_norm, indices = utils.findFlakyScenarios(scenarios, f_arr)
    
    ##  Plotting histogram for flaky_norm...
    plt.figure()
    sns.set_theme()
    sns.histplot(flaky_norm, bins=20)
    plt.title('Histogram of flaky scenarios')
    plt.xlabel('Norm of fitness difference between min and mean')
    plt.show()

    ##  Plotting norm of fitnesses for each scenario...
    flaky_detailed = f_arr_detailed[indices]
    flaky_fitness_norms = np.linalg.norm(flaky_detailed, axis=2)
    n = 100

    plt.figure(figsize=(20, 1.5*n))
    sns.set_theme()
    for i in range(n):
        plt.subplot(int(n/4), 4, i+1)
        plt.plot(range(10), flaky_fitness_norms[i], '-x')
        plt.title(f'Rank {i+1}')
        plt.xlabel('Run')
        plt.ylabel('Norm of fitness')
        print(f'Rank {i+1}: {scenarios[indices[i]]}')
    plt.tight_layout()
    plt.savefig('flaky_fitness_norms.pdf')


        