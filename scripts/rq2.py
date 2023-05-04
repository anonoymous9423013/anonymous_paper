from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import wilcoxon
from rq2_dataset import balance
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
from rq2_models import trainDecisionTree, trainRandomForest, trainSVM, trainMLP
import os
import rq1

log.basicConfig(level=log.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
mpl.rcParams['legend.fontsize'] = 25
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22


output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
thresh = 0.2
min_legend, mean_legend = "RS(n=10)", "RS(n=1)"
fitness_labels = ["F1", "F2", "F3", "F4"]
no_of_fitnesses = len(fitness_labels)
platform = "BeamNG"
csv_file = f"ds_{platform.lower()}.csv"
ds_main = os.path.join(output_folder, f"ds_fit_{platform}_{int(thresh * 100)}.csv")
ds_delta = os.path.join(output_folder, f"ds_delta_{platform}_{int(thresh * 100)}.csv")
df_main = pd.read_csv(ds_main)
probs = [
    len(df_main[df_main["Y"] == 0]) / len(df_main),
    len(df_main[df_main["Y"] == 1]) / len(df_main),
]
fitness_threshes = np.array([-1.5, -90, -1000, -10])
log.info(f"Proportions of classes: {probs}")

##  PID...
dropped_pid_5_main = ["Weather1", "Weather2", "Weather3"]
dropped_pid_5_delta = []
dropped_pid_5 = {"main": dropped_pid_5_main, "delta": dropped_pid_5_delta}
dropped_pid_10_main = ["Weather1", "Weather2", "Weather3"]
dropped_pid_10_delta = []
dropped_pid_10 = {"main": dropped_pid_10_main, "delta": dropped_pid_10_delta}
dropped_pid_20_main = []
dropped_pid_20_delta = []
dropped_pid_20 = {"main": dropped_pid_20_main, "delta": dropped_pid_20_delta}
dropped_pid = {5: dropped_pid_5, 10: dropped_pid_10, 20: dropped_pid_20}
pid_5_main_model = "mlp"
pid_5_delta_model = "mlp"
pid_10_main_model = "mlp"
pid_10_delta_model = "mlp"
pid_20_main_model = "mlp"
pid_20_delta_model = "rf"
models_pid_5 = {"main": pid_5_main_model, "delta": pid_5_delta_model}
models_pid_10 = {"main": pid_10_main_model, "delta": pid_10_delta_model}
models_pid_20 = {"main": pid_20_main_model, "delta": pid_20_delta_model}
pid_models = {5: models_pid_5, 10: models_pid_10, 20: models_pid_20}

##  Pylot...
dropped_pylot_5_main = ["Weather"]
dropped_pylot_5_delta = []
dropped_pylot_5 = {"main": dropped_pylot_5_main, "delta": dropped_pylot_5_delta}
dropped_pylot_10_main = []
dropped_pylot_10_delta = []
dropped_pylot_10 = {"main": dropped_pylot_10_main, "delta": dropped_pylot_10_delta}
dropped_pylot_20_main = ["Weather"]
dropped_pylot_20_delta = ["Weather"]
dropped_pylot_20 = {"main": dropped_pylot_20_main, "delta": dropped_pylot_20_delta}
dropped_pylot = {5: dropped_pylot_5, 10: dropped_pylot_10, 20: dropped_pylot_20}
pylot_5_main_model = "rf"
pylot_5_delta_model = "mlp"
pylot_10_main_model = "rf"
pylot_10_delta_model = "mlp"
pylot_20_main_model = "mlp"
pylot_20_delta_model = "mlp"
models_pylot_5 = {"main": pylot_5_main_model, "delta": pylot_5_delta_model}
models_pylot_10 = {"main": pylot_10_main_model, "delta": pylot_10_delta_model}
models_pylot_20 = {"main": pylot_20_main_model, "delta": pylot_20_delta_model}
pylot_models = {5: models_pylot_5, 10: models_pylot_10, 20: models_pylot_20}

##  BeamNG...
dropped_beamng_5_main = ["TrafficAmount"]
dropped_beamng_5_delta = []
dropped_beamng_5 = {"main": dropped_beamng_5_main, "delta": dropped_beamng_5_delta}
dropped_beamng_10_main = []
dropped_beamng_10_delta = ["TimeOfDay", "Weather"]
dropped_beamng_10 = {"main": dropped_beamng_10_main, "delta": dropped_beamng_10_delta}
dropped_beamng_20_main = ["TimeOfDay", "Weather"]
dropped_beamng_20_delta = ["TrafficAmount"]
dropped_beamng_20 = {"main": dropped_beamng_20_main, "delta": dropped_beamng_20_delta}
dropped_beamng = {5: dropped_beamng_5, 10: dropped_beamng_10, 20: dropped_beamng_20}
beamng_5_main_model = "rf"
beamng_5_delta_model = "rf"
beamng_10_main_model = "rf"
beamng_10_delta_model = "rf"
beamng_20_main_model = "rf"
beamng_20_delta_model = "rf"
models_beamng_5 = {"main": beamng_5_main_model, "delta": beamng_5_delta_model}
models_beamng_10 = {"main": beamng_10_main_model, "delta": beamng_10_delta_model}
models_beamng_20 = {"main": beamng_20_main_model, "delta": beamng_20_delta_model}
beamng_models = {5: models_beamng_5, 10: models_beamng_10, 20: models_beamng_20}


dropped = {"pid": dropped_pid, "pylot": dropped_pylot, "beamng": dropped_beamng}
models_platforms = {"pid": pid_models, "pylot": pylot_models, "beamng": beamng_models}

columns = list(pd.read_csv(csv_file).columns)
dropped_fit_idx = [
    columns.index(c) for c in dropped[platform.lower()][int(thresh * 100)]["main"]
]
dropped_diff_idx = [
    columns.index(c) for c in dropped[platform.lower()][int(thresh * 100)]["delta"]
]


def _trainModel(model_type, df_train, cv=5, **kwargs):
    match model_type:
        case "dt":
            return trainDecisionTree(df_train, cv=cv, **kwargs)
        case "rf":
            return trainRandomForest(df_train, cv=cv, **kwargs)
        case "svm":
            return trainSVM(df_train, cv=cv, **kwargs)
        case "mlp":
            return trainMLP(df_train, cv=cv, **kwargs)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def trainModels(df_main, df_delta, cv=5, platform="PID", **kwargs):
    models = []
    df_main_dropped = dropped[platform.lower()][int(thresh * 100)]["main"]
    df_delta_dropped = dropped[platform.lower()][int(thresh * 100)]["delta"]
    print("Dropped main:", df_main_dropped)
    print("Dropped delta:", df_delta_dropped)
    print(df_main.columns)
    df_main = df_main.drop(df_main_dropped, axis=1)
    df_delta = df_delta.drop(df_delta_dropped, axis=1)

    if "Timestep" in df_main.columns:
        df_main = df_main.drop(["Timestep"], axis=1)

    ##  Decision Tree with 1 set of fitnesses...
    df_train = df_main.sample(frac=1.0, random_state=200)
    df_train = balance(df_train, output_prefix=None)

    model_fit_type = models_platforms[platform.lower()][int(thresh * 100)]["main"]
    model_diff_type = models_platforms[platform.lower()][int(thresh * 100)]["delta"]

    scores, desc = _trainModel(model_fit_type, df_train, cv=cv, **kwargs)
    f1s = scores["test_f1"]
    models.append(scores["estimator"][np.argmax(f1s)])

    ##  Decision Tree with delta fitnesses...
    df_train = df_delta.sample(frac=1.0, random_state=200)
    df_train = balance(df_train, output_prefix=None)
    scores, desc = _trainModel(model_diff_type, df_train, cv=cv, **kwargs)
    f1s = scores["test_f1"]
    models.append(scores["estimator"][np.argmax(f1s)])
    return models


def plotRS(f_mins, labels, show=True, platform="PID"):
    """Plots the random search results in iterations...

    Args:
        f_mins (list):
        f_mean (_type_): _description_
        labels (list): The first elements are labels for f_mins, the last element is for f_mean.
        show (bool, optional): _description_. Defaults to True.
    """
    output_file = os.path.join(
        output_folder, f"rq2_rs_iters_{platform}_{int(thresh * 100)}.pdf"
    )
    t0 = time.time()
    df_mins_dict = [
        {"Iteration": [], **{fitness_labels[j]: [] for j in range(len(fitness_labels))}}
        for i in range(len(f_mins))
    ]
    for k in range(len(f_mins)):
        for i in range(len(f_mins[k])):
            for iter in range(len(f_mins[k][i])):
                df_mins_dict[k]["Iteration"].append(iter)
                for j in range(f_mins[k].shape[-1]):
                    df_mins_dict[k][fitness_labels[j]].append(f_mins[k][i, iter, j])

    df_mins = [pd.DataFrame(df_min_dict) for df_min_dict in df_mins_dict]

    loc, fontsize = "upper right", 12
    sns.set(font_scale=2.2)
    fig, ax = plt.subplots(
        nrows=2,
        ncols=2,
        # subplot_kw=dict(box_aspect=1),
        figsize=(14, 11),
        layout="constrained",
    )
    for i in range(no_of_fitnesses):
        plt.subplot(2, 2, i + 1)
        for j in range(len(df_mins)):
            sns.lineplot(
                data=df_mins[j], x="Iteration", y=fitness_labels[i], label=labels[j]
            )
        plt.xlabel("Iteration")
        plt.ylabel("")
        plt.title(fitness_labels[i])
        plt.legend(loc=loc, fontsize=fontsize)
        axx = ax[i // 2][i % 2]
        if i != 1:
            axx.get_legend().remove()
        else:
            plt.setp(axx.get_legend().get_texts(), fontsize="25", fontweight="bold")  # for legend text   
            plt.setp(axx.get_legend().get_title(), fontsize="25", fontweight="bold")  # for legend title
        for tick in axx.get_xticklabels():
            tick.set_fontweight("bold")
        for tick in axx.get_yticklabels():
            tick.set_fontweight("bold")

    plt.savefig(output_file, bbox_inches="tight")
    t1 = time.time()
    log.info(
        f"Plotting random search fitnesses took {t1-t0} seconds. Output file: {output_file}"
    )
    plt.show() if show else None


def plotBox(f_mins, labels, show=True, platform="PID"):
    output_file = os.path.join(
        output_folder, f"rq2_rs_box_{platform}_{int(thresh * 100)}.pdf"
    )
    t0 = time.time()
    df_box_dict_mins = [
        {"Method": [], **{fitness_labels[j]: [] for j in range(len(fitness_labels))}}
        for i in range(len(f_mins))
    ]
    for k in range(len(f_mins)):
        for i in range(len(f_mins[k])):
            df_box_dict_mins[k]["Method"].append(labels[k])
            for j in range(f_mins[k].shape[-1]):
                df_box_dict_mins[k][fitness_labels[j]].append(f_mins[k][i, -1, j])
    df_box_mins = [
        pd.DataFrame(df_box_dict_min) for df_box_dict_min in df_box_dict_mins
    ]
    df_box = pd.concat(df_box_mins, ignore_index=True)

    shining_green = "#00FF00"  ##  For displaying mean
    sns.set(font_scale=2.2)
    fig, ax = plt.subplots(
        nrows=1,
        ncols=no_of_fitnesses,
        subplot_kw=dict(box_aspect=1),
        figsize=(14, 11),
        layout="constrained",
    )
    for i in range(no_of_fitnesses):
        plt.subplot(2, 2, i + 1)
        boxplot = sns.boxplot(
            data=df_box,
            x="Method",
            y=fitness_labels[i],
            hue="Method",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": shining_green,
                "markeredgecolor": shining_green,
            },
            dodge=False,
        )
        plt.legend([], [], frameon=False)
        plt.xlabel("", fontweight="bold")
        plt.ylabel("", fontweight="bold")
        plt.xticks(fontweight="bold", rotation=0)
        plt.yticks(fontweight="bold")
        plt.title(fitness_labels[i], fontweight="bold")
    plt.savefig(output_file, bbox_inches="tight")
    t1 = time.time()
    log.info(f"Plotting boxplots took {t1-t0} seconds. Output file: {output_file}")
    plt.show() if show else None


def smartRandomSearchOneIteration(
    scenario, f_arr_detailed_iter, models=None, method="or", platform="PID"
):  #  RoadType and RoadID...
    scenario_fit = [
        scenario[i] for i in range(len(scenario)) if i not in dropped_fit_idx
    ]
    cnt = 1
    X = np.array([scenario_fit])
    Xfitness = np.concatenate((X[0], f_arr_detailed_iter[0]))[
        np.newaxis, :
    ]  ##  1 set of fitnesses...
    ys = []
    ps = []
    if isinstance(models, list):  ##  Machine learning methods...
        ps.append(models[0].predict(Xfitness)[0] > 0.5)
    elif isinstance(models, str) and models == "thresh":
        pass
    else:
        ps.append(np.random.rand() > probs[0])
    sceanrio_diff = [
        scenario[i] for i in range(len(scenario)) if i not in dropped_diff_idx
    ]
    X_diff = np.array([sceanrio_diff])
    for i in range(2, 10):  ##  max number of time steps we want to proceed...
        deltas = np.max(f_arr_detailed_iter[:i], axis=0) - np.min(
            f_arr_detailed_iter[:i], axis=0
        )
        Xdelta = np.concatenate((X_diff[0], deltas))[np.newaxis, :]
        ys.append(f_arr_detailed_iter[i])
        if isinstance(models, list):  ##  Machine learning methods...
            ps.append(models[1].predict(Xdelta)[0] > 0.5)
        elif isinstance(models, str) and models == "thresh":
            # flag = (np.max(f_arr_detailed_iter[:i], axis=0) > fitness_threshes) & (np.min(f_arr_detailed_iter[:i], axis=0) < fitness_threshes)
            flag = deltas >= delta_max * thresh
            if np.any(flag):
                cnt -= -1
                break
        else:
            ps.append(np.random.rand() > probs[0])

        cnt -= -1
        if method == "and" and not all(ps):
            break
        elif method == "or" and not any(ps):
            break
    f_min = np.min(ys, axis=0)
    f_mean = np.mean(ys, axis=0)
    return f_min, f_mean, cnt


def smartRandomSearchOneRun(
    scenarios, f_arr_detailed, models=None, method="or", platform="PID"
):
    f_min, f_mean, cnt = smartRandomSearchOneIteration(
        scenarios[0], f_arr_detailed[0], models, method, platform
    )
    f_mins = [f_min]
    f_means = [f_mean]
    cnts = [cnt]
    for i in range(1, len(scenarios)):
        f_min, f_mean, cnt = smartRandomSearchOneIteration(
            scenarios[i], f_arr_detailed[i], models, method, platform
        )
        f_mins.append([min(f_min[j], f_mins[-1][j]) for j in range(len(f_min))])
        f_means.append([min(f_mean[j], f_means[-1][j]) for j in range(len(f_mean))])
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)


def smartRandomSearch(
    scenarios, f_arr_detailed, runs=20, models=None, method="or", platform="PID"
):
    n_iterations = len(f_arr_detailed) // runs
    f_mins = []
    f_means = []
    cnts = []
    for i in range(runs):
        f_mins_prime, f_means_prime, cnt = smartRandomSearchOneRun(
            scenarios[i * n_iterations : (i + 1) * n_iterations],
            f_arr_detailed[i * n_iterations : (i + 1) * n_iterations],
            models,
            method,
            platform,
        )
        f_mins.append(f_mins_prime)
        f_means.append(f_means_prime)
        cnts.append(cnt)
    f_mins = np.array(f_mins)
    f_means = np.array(f_means)
    return f_mins, f_means, sum(cnts)


def mhWilcoxon(a, b):
    if np.allclose(a, b):

        class MHWilcoxon:
            def __init__(self, pvalue):
                self.pvalue = pvalue

        return MHWilcoxon(0.0)
    else:
        return wilcoxon(a, b)


def statisticalTests(
    f_mins,
    baseline,
    f_mins_labels,
    baseline_label=None,
    cnts=None,
    baseline_cnt=None,
    platform="PID",
    show=False,
    output_prename="tab-rq2-stattest_",
):
    t0 = time.time()
    csv_output_file = os.path.join(
        output_folder, f"{output_prename}{platform}_{int(thresh * 100)}.txt"
    )
    tex_output_file = os.path.join(
        output_folder, f"{output_prename}{platform}_{int(thresh * 100)}.tex"
    )

    def a12(lst1, lst2, rev=True):
        more = same = 0.0
        for x in lst1:
            for y in lst2:
                if x == y:
                    same += 1
                elif rev and x > y:
                    more += 1
                elif not rev and x < y:
                    more += 1
        return (more + 0.5 * same) / (len(lst1) * len(lst2))

    f_mins_mean = [
        np.mean(f_min, axis=0) for f_min in f_mins
    ]  ##  list of shapes (no_of_iterations, no_of_fitnesses)
    baseline_mean = np.mean(
        baseline, axis=0
    )  ##  shape (no_of_iterations, no_of_fitnesses)
    ##  Wilcoxon signed-rank test...
    res = [
        [mhWilcoxon(baseline_mean[:, i], f_min[:, i]) for i in range(no_of_fitnesses)]
        for f_min in f_mins_mean
    ]
    p_values = [[r.pvalue for r in q] for q in res]
    ##  Vargha-Delaney A^12 test...
    a12_values = [
        [a12(f_min[:, i], baseline_mean[:, i]) for i in range(no_of_fitnesses)]
        for f_min in f_mins_mean
    ]
    ##  Writing results to file...
    df_dict = {"Method": [], "Simulations": []}
    df_dict = {**df_dict, **{f"F{i+1} p-value": [] for i in range(no_of_fitnesses)}}
    df_dict = {**df_dict, **{f"F{i+1} A^12": [] for i in range(no_of_fitnesses)}}
    for i in range(len(f_mins)):
        df_dict["Method"].append(f_mins_labels[i])
        df_dict["Simulations"].append(cnts[i])
        for j in range(no_of_fitnesses):
            df_dict[f"F{j+1} p-value"].append(p_values[i][j])
            df_dict[f"F{j+1} A^12"].append(a12_values[i][j])
    df = pd.DataFrame(df_dict)
    df.to_csv(csv_output_file, index=False)
    df.to_latex(
        tex_output_file,
        index=False,
        escape=False,
        column_format="l" + "c" * (2 * no_of_fitnesses),
    )
    t1 = time.time()
    log.info(
        f"Statistical tests took {t1-t0} seconds. Results are in {csv_output_file}, {tex_output_file}..."
    )


def accuracyTable(
    f_mins,
    baseline,
    f_mins_labels,
    baseline_label=None,
    cnts=None,
    baseline_cnt=None,
    platform="PID",
    show=False,
):
    t0 = time.time()
    csv_output_file = os.path.join(
        output_folder, f"rq2-acc_{platform}_{int(thresh * 100)}.txt"
    )
    tex_output_file = os.path.join(
        output_folder, f"tab-rq2-acc_{platform}_{int(thresh * 100)}.tex"
    )
    pdf_output_file = os.path.join(
        output_folder, f"tab-rq2-acc_{platform}_{int(thresh * 100)}.pdf"
    )
    f_mins_mean = [
        np.mean(f_min[:, -1, :], axis=0) for f_min in f_mins
    ]  ##  list of shapes (no_of_fitnesses,)
    baseline_mean = np.mean(baseline[:, -1, :], axis=0)  ##  shape (no_of_fitnesses,)
    df_dict = {"Method": [], "Simulations": []}
    df_dict = {**df_dict, **{f"F{i+1}": [] for i in range(no_of_fitnesses)}}
    ##  Baseline...
    df_dict["Method"].append(baseline_label)
    df_dict["Simulations"].append(baseline_cnt)
    for i in range(no_of_fitnesses):
        df_dict[f"F{i+1}"].append(baseline_mean[i])
    ##  Other methods...
    for i in range(len(f_mins)):
        df_dict["Method"].append(f_mins_labels[i])
        df_dict["Simulations"].append(cnts[i])
        for j in range(no_of_fitnesses):
            df_dict[f"F{j+1}"].append(f_mins_mean[i][j])
    df = pd.DataFrame(df_dict)
    for col in df.columns:
        if col != "Method" and col != "Simulations":
            df[col] = df[col].apply(lambda x: f"{x:.3f} ({x / df[col][0] * 100:.1f}\%)")
    df.to_csv(csv_output_file, index=False)
    df.to_latex(
        tex_output_file,
        index=False,
        escape=False,
        column_format="l" + "c" * (no_of_fitnesses + 2),
    )
    t1 = time.time()
    log.info(
        f"Accuracy table took {t1-t0} seconds. Results are in {csv_output_file}, {tex_output_file}, and {pdf_output_file}..."
    )


def evalDeltaSteps(scenarios, f_arr_detailed, models, platform="PID"):
    result_dict = {
        "Timestep": [],
        "Baseline Presicion": [],
        "Baseline Recall": [],
        "Baseline F1": [],
        "Model Presicion": [],
        "Model Recall": [],
        "Model F1": [],
    }
    for t in range(2, 10):
        targets = []
        preds_model = []
        preds_thresh = []
        for i in range(len(scenarios)):
            scenario = scenarios[i]
            scenario = [
                scenario[i] for i in range(len(scenario)) if i not in dropped_diff_idx
            ]
            X = np.array([scenario])
            f_arr = f_arr_detailed[i, :t, :]
            deltas = np.max(f_arr, axis=0) - np.min(f_arr, axis=0)
            Xdelta = np.concatenate((X[0], deltas))[np.newaxis, :]
            y_pred_model = (models[1].predict(Xdelta)[0] > 0.5).astype(int)
            y_pred_thresh = (deltas >= delta_max * 0.05).any().astype(int)
            deltas_target = np.max(f_arr_detailed[i], axis=0) - np.min(
                f_arr_detailed[i], axis=0
            )
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
        result_dict["Timestep"].append(t)
        result_dict["Baseline Presicion"].append(precision_thresh)
        result_dict["Baseline Recall"].append(recall_thresh)
        result_dict["Baseline F1"].append(f1_thresh)
        result_dict["Model Presicion"].append(precision_model)
        result_dict["Model Recall"].append(recall_model)
        result_dict["Model F1"].append(f1_model)
    df = pd.DataFrame(result_dict)
    tex_output_file = os.path.join(
        output_folder, f"tab-rq2-delta-{platform}-{int(thresh * 100)}.tex"
    )
    df.to_latex(
        tex_output_file,
        index=False,
        escape=False,
        column_format="l" + "c" * 6,
    )
    ##  Plotting in 3 subplots...
    baseline_label = "$RS_b$"
    model_label = "$RS_{ML}$"
    sns.set(font_scale=2.2)
    xlabel = "No of Executions"
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    metrics = ["Presicion", "Recall", "F1"]
    for i, metric in enumerate(metrics):
        axs[i].plot(df["Timestep"], df[f"Baseline {metric}"], "x--", label=baseline_label, markersize=15)
        axs[i].plot(df["Timestep"], df[f"Model {metric}"], "x--", label=model_label, markersize=15)
        axs[i].set_xlabel(xlabel)
        axs[i].set_xticks(range(2, 10))
        axs[i].set_ylabel(metric)
        axs[i].set_ylim([0.4, 1.1])
        axs[i].legend()
    pdf_output_file = os.path.join(
        output_folder, f"fig-rq2-delta-{platform}_{int(thresh * 100)}.pdf"
    )
    plt.savefig(pdf_output_file, bbox_inches="tight")
    plt.show()
    log.info(f"Table saved to {tex_output_file}...")
    log.info(f"Plot saved to {pdf_output_file}...")


if __name__ == "__main__":
    ##  Training models...
    # platform = "PID"
    csv_file = f"ds_{platform.lower()}.csv"
    ds_main = os.path.join(output_folder, f"ds_fit_{platform}_{int(thresh * 100)}.csv")
    ds_delta = os.path.join(
        output_folder, f"ds_delta_{platform}_{int(thresh * 100)}.csv"
    )
    df_main = pd.read_csv(ds_main)
    df_main = df_main.sample(frac=1, random_state=42).reset_index(drop=True)
    df_delta = pd.read_csv(ds_delta)
    df_delta = df_delta.sample(frac=1, random_state=42).reset_index(drop=True)
    df_main = df_main.drop(["Y0", "Y1", "Y2", "Y3"], axis=1)
    df_delta = df_delta.drop(["Y0", "Y1", "Y2", "Y3"], axis=1)
    models = trainModels(df_main, df_delta, platform=platform)
    log.info("Models trained...")
    assert len(models) == 2
    ##  Loading fitnesses...
    result_detailed, f_arr_detailed = utils.fitnessForAllScenarios(
        platform=platform, csv_file=csv_file, normalize=False
    )  ##  f_arr_detailed.shape = (no_of_scenarios, no_of_repeats, no_of_fitnesses)
    deltas = np.max(f_arr_detailed, axis=1) - np.min(
        f_arr_detailed, axis=1
    )  ##  (no_of_scenarios, no_of_fitnesses)
    delta_max = np.max(deltas, axis=0)  ##  (no_of_fitnesses,)
    scenarios, __ = utils.minAndMeanFitnesses(result_detailed, normalize=True)

    ##  fastFitness manual baseline ...
    (
        f_min_baseline_manual,
        f_mean_baseline_manual,
        cnt_baseline_manual,
    ) = smartRandomSearch(
        scenarios, f_arr_detailed, models="thresh", method="baseline", platform=platform
    )

    ##  fastFitness with models with and...
    f_min_smart_and, f_mean_smart_and, cnt_and = smartRandomSearch(
        scenarios, f_arr_detailed, models=models, method="and", platform=platform
    )

    ##  Random search for 10 repetitions...
    scenarios, f_arr10 = utils.minAndMeanFitnesses(
        result_detailed, normalize=False, n_reps=10
    )
    f_min10, f_mean10 = rq1.randomSearch(f_arr10, runs=20)
    min_ = np.min(f_arr10, axis=(0, 1))
    max_ = np.max(f_arr10, axis=(0, 1))

    ##  Normalizing fitnesses for smart random search...
    f_min_baseline_manual = (f_min_baseline_manual - min_) / (max_ - min_)
    f_min_smart_and = (f_min_smart_and - min_) / (max_ - min_)
    f_min10 = (f_min10 - min_) / (max_ - min_)
    f_mean10 = (f_mean10 - min_) / (max_ - min_)

    plotRS(
        [f_min_baseline_manual, f_min_smart_and, f_min10, f_mean10],
        ["$RS_b$", "$RS_{{ML}}$", "$RS_{n=10}$", "$RS_{n=1}$"],
        show=True,
        platform=platform,
    )
    plotBox(
        [f_min_baseline_manual, f_min_smart_and, f_min10, f_mean10],
        ["$RS_b$", "$RS_{{ML}}$", "$RS_{n=10}$", "$RS_{n=1}$"],
        show=True,
        platform=platform,
    )

    log.info(f"Number of iterations for baseline manual: {cnt_baseline_manual}")
    log.info(f"Number of iterations for smart RS with models AND: {cnt_and}")

    cnts = [cnt_baseline_manual, cnt_and, 10000]
    cnt_baseline = 10000
    statisticalTests(
        [f_min_baseline_manual, f_min_smart_and],
        f_mean10,
        ["Baseline-Manual", "fastFitness"],
        "RS(n=10)",
        cnts,
        cnt_baseline,
        platform=platform,
    )
    statisticalTests(
        [
            f_min_baseline_manual,
        ],
        f_min_smart_and,
        [
            "Baseline-Manual",
        ],
        "fastFitness",
        [
            cnts[0],
        ],
        cnts[1],
        platform=platform,
        output_prename="rq2_baseline_vs_fastFitness_stattest_",
    )

    accuracyTable(
        [f_min_baseline_manual, f_min_smart_and],
        f_min10,
        ["Baseline-Manual", "fastFitness"],
        "RS(n=10)",
        cnts,
        cnt_baseline,
        platform=platform,
    )
    evalDeltaSteps(scenarios, f_arr_detailed, models, platform=platform)
