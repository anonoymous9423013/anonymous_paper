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
from functools import partial

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fitnessForAllScenarios(platform='PID', csv_file=None, normalize=True, repeat=10):
    """  Calculates fitness for each scenario  """
    t0 = time.time()
    if platform == 'PID':
        df = pd.read_csv(csv_file)
        result = []
        input_cols = list(df.columns)[:8]
        output_cols = list(df.columns)[-4:]
        test_inputs = df[input_cols].values
        unique_arr, counts = np.unique(test_inputs, axis=0, return_counts=True)
        unique = unique_arr.tolist()
        ##  Throw away scenarios with less than 10 repetitions...
        unique = [u for u, c in zip(unique, counts) if c >= repeat]
        for scenario in unique:
            result.append([scenario])
            fitnesses = df[(df.iloc[:, :8].values == np.array(scenario)).all(axis=1)][output_cols].values
            fitnesses = fitnesses[:repeat]
            for f_row in fitnesses:
                result[-1].append(f_row.tolist())
        t1 = time.time()
        arr = np.array([r[1:1+repeat] for r in result])
        if normalize:
            # arr = normalizeStd(arr)
            arr = normalizeMinMax(arr)
    elif platform == 'Pylot':
        df = pd.read_csv(csv_file)
        result = []
        input_cols = list(df.columns)[1:-5]
        output_cols = list(df.columns)[-5:]
        output_cols.remove('f3')
        test_inputs = df[input_cols].values
        unique_arr, counts = np.unique(test_inputs, axis=0, return_counts=True)
        unique = unique_arr.tolist()
        ##  Throw away scenarios with less than 10 repetitions...
        unique = [u for u, c in zip(unique, counts) if c >= repeat]
        for scenario in unique:
            result.append([scenario])
            fitnesses = df[(df.iloc[:, 1:-5].values == np.array(scenario)).all(axis=1)][output_cols].values
            fitnesses = fitnesses[:repeat]
            for f_row in fitnesses:
                result[-1].append(f_row.tolist())
        t1 = time.time()
        arr = np.array([r[1:1+repeat] for r in result])
        if normalize:
            # arr = normalizeStd(arr)
            arr = normalizeMinMax(arr)
    elif platform == 'BeamNG':
        df = pd.read_csv(csv_file)
        result = []
        input_cols = list(df.columns)[:6]
        output_cols = list(df.columns)[-4:]
        test_inputs = df[input_cols].values
        unique_arr, counts = np.unique(test_inputs, axis=0, return_counts=True)
        unique = unique_arr.tolist()
        ##  Throw away scenarios with less than 10 repetitions...
        unique = [u for u, c in zip(unique, counts) if c >= repeat]
        for scenario in unique:
            result.append([scenario])
            fitnesses = df[(df.iloc[:, :6].values == np.array(scenario)).all(axis=1)][output_cols].values
            fitnesses = fitnesses[:repeat]
            for f_row in fitnesses:
                result[-1].append(f_row.tolist())
        t1 = time.time()
        arr = np.array([r[1:1+repeat] for r in result])
        if normalize:
            # arr = normalizeStd(arr)
            arr = normalizeMinMax(arr)
    ##  Keeping only up to a factor of 50...
    print('ARR: ', len(arr))
    n = len(arr) - len(arr) % 50
    arr = arr[:n]
    result = result[:n]
    log.info(f'Read {len(arr)} scenarios in {t1-t0:.2f} seconds...')
    return result, arr

def minAndMeanFitness(result:list, n_reps=10):
    """Finds the min and mean fitness for one scenario

    Args:
        result (list): List of fitnesses for each file in the scenario. The first element
        is the scenario name. This is actually the output of fitnessForScenario.
        n_reps (int, optional): Number of repetitions. Defaults to 10.

    Returns:
        list: A 3-element list. The first element is the scenario name. The second element
        is a list of min fitnesses for each fitness type. The third element is a list of
        mean fitnesses for each fitness type.
    """
    scenario = result[0]
    fitnesses = result[1:]
    fitness_no = len(fitnesses[0])
    fitness_min = [np.min([f[i] for f in fitnesses[:n_reps]]) for i in range(fitness_no)]
    fitness_mean = [np.mean([f[i] for f in fitnesses[:n_reps]]) for i in range(fitness_no)]
    return [scenario, fitness_min, fitness_mean]

def minAndMeanFitnesses(result:list, normalize=True, n_reps=10):
    """Calculates min and mean fitness for each scenario

    Args:
        result (list): List of results from fitnessForAllScenarios. Each element contains
        a list of fitnesses for each file in the scenario. The first element of each element
        is the scenario name.
        normalize (bool, optional): Whether to normalize the fitnesses. Defaults to True.
        n_reps (int, optional): Number of repetitions. Defaults to 10.
    """
    t0 = time.time()
    with Pool(8) as p:
        result = p.map(partial(minAndMeanFitness, n_reps=n_reps), result)
    t1 = time.time()

    scenarios = [r[0] for r in result]
    array = np.array([r[1:] for r in result])
    if normalize:
        array = normalizeMinMax(array)
    log.info(f'Found min and mean fitnesses of {len(result)} scenarios in {t1-t0:.2f} seconds...')
    return scenarios, array

def normalizeStd(array, axis=(0,1)):
    """ Normalizes each each of the fitnesses in the array to have mean 0 and std 1 """
    return (array - np.mean(array, axis=axis)) / np.std(array, axis=axis)

def normalizeMinMax(array, axis=(0,1)):
    """ Normalizes each each of the fitnesses in the array to have min -1 and max 1 """
    return (array - np.min(array, axis=axis)) / (np.max(array, axis=axis) - np.min(array, axis=axis)) * 2 - 1

def findFlakyScenarios(scenarios:list, array:np.array):
    diff = np.diff(array, axis=1).squeeze()
    diff_norm = np.linalg.norm(diff, axis=1)
    indices = np.argsort(diff_norm)[::-1]
    flaky_scenarios = [scenarios[i] for i in indices]
    flaky_arr = array[indices]
    flaky_norm = diff_norm[indices]
    return flaky_scenarios, flaky_arr, flaky_norm, indices

    

if __name__ == '__main__':
    # files = glob('log/log_*.txt')
    # scenarios = list(set([f[:f.rfind('_')] for f in files]))
    # result_detailed, f_arr_detailed = fitnessForAllScenarios(scenarios, normalize=False)  ##  f_arr_detailed.shape = (no_of_scenarios, no_of_repeats, no_of_fitnesses)
    # result_detailed, f_arr_detailed = fitnessForAllScenarios(platform='BeamNG', csv_file='ds_beamng.csv', normalize=False)  ##  f_arr_detailed.shape = (no_of_scenarios, no_of_repeats, no_of_fitnesses)
    result_detailed, f_arr_detailed = fitnessForAllScenarios(platform='PID', csv_file='ds_pid_new.csv', normalize=False)  ##  f_arr_detailed.shape = (no_of_scenarios, no_of_repeats, no_of_fitnesses)
    scenarios, f_arr = minAndMeanFitnesses(result_detailed)  ##  f_arr.shape = (no_of_scenarios, 2, no_of_fitnesses)
    
    df = pd.DataFrame(f_arr_detailed.reshape(-1, f_arr_detailed.shape[-1]), columns=['F1', 'F2', 'F3', 'F4'])
    ##  Visualize df statistics as a table using seaborn...
    plt.figure()
    sns.set_theme()
    sns.heatmap(df.describe()[1:].transpose(), annot=True, fmt='.2f')
    plt.title('Statistics of fitnesses')
    plt.show()

    ##  Draw histogram of fitnesses...
    plt.figure()
    sns.set_theme()
    sns.histplot(df, bins=20)
    plt.title('Histogram of fitnesses')
    plt.show()

    ##
    flaky_scenarios, flaky_arr, flaky_norm, indices = findFlakyScenarios(scenarios, f_arr)
    
    ##  Plotting histogram for flaky_norm...
    plt.figure()
    sns.set_theme()
    sns.histplot(flaky_norm, bins=20)
    plt.title('Histogram of flaky scenarios')
    plt.xlabel('Norm of fitness difference between min and mean')
    plt.show()

    