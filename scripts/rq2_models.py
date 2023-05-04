from matplotlib import pyplot as plt
from rq2_dataset import balance
import logging as log
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import warnings

warnings.filterwarnings("ignore")

log.basicConfig(level=log.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def trainDecisionTree(df, cv=5, **kwargs):
    max_depth = kwargs.get("max_depth", 5)
    model = DecisionTreeClassifier(max_depth=max_depth)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["precision", "recall", "f1"], return_estimator=True
    )
    desc = "Decision Tree"
    return scores, desc


def trainRandomForest(df, cv=5, **kwargs):
    max_depth = kwargs.get("max_depth", 20)
    n_estimators = kwargs.get("n_estimators", 20)
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["precision", "recall", "f1"], return_estimator=True
    )
    desc = "Random Forest"
    return scores, desc


def trainSVM(df, cv=5, **kwargs):
    kernel = kwargs.get("kernel", "rbf")
    model = SVC(kernel=kernel)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["precision", "recall", "f1"], return_estimator=True
    )
    desc = f"SVM with {kernel} kernel"
    return scores, desc


def trainMLP(df, cv=5, **kwargs):
    hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (50, 100))
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, learning_rate="adaptive"
    )
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scores = cross_validate(
        model, X, y, cv=cv, scoring=["precision", "recall", "f1"], return_estimator=True
    )
    desc = f"MLP with {hidden_layer_sizes} hidden layer sizes"
    return scores, desc


def _train(train, test, cv, approach, results_dict, filehandler, **kwargs):
    if approach == "dt":
        scores, desc = trainDecisionTree(train, cv=cv, **kwargs)
    elif approach == "rf":
        scores, desc = trainRandomForest(train, cv=cv, **kwargs)
    elif approach == "svm":
        scores, desc = trainSVM(train, cv=cv, **kwargs)
    elif approach == "mlp":
        scores, desc = trainMLP(train, cv=cv, **kwargs)
    models = scores["estimator"]
    predictions = [model.predict(test.iloc[:, :-1].values) for model in models]
    precisions = [
        precision_score(test.iloc[:, -1].values, prediction)
        for prediction in predictions
    ]
    recalls = [
        recall_score(test.iloc[:, -1].values, prediction) for prediction in predictions
    ]
    f1s = [f1_score(test.iloc[:, -1].values, prediction) for prediction in predictions]
    results_dict["Method"].append(approach)
    results_dict["Input"].append(kwargs["input"])
    argmax = np.argmax(f1s)
    results_dict["Precision"].append(precisions[argmax])
    results_dict["Recall"].append(recalls[argmax])
    results_dict["F1"].append(f1s[argmax])
    filehandler.write("*" * 25 + " With all the features  " + "*" * 25 + "\n")
    filehandler.write(f"Approach: {desc}\n")
    filehandler.write(f"Inputs: {list(df_main.columns[:-1])}\n")
    filehandler.write(f"Output: {list(df_main.columns[-1])}\n")
    filehandler.write(
        f'Number of fitnesses included in input: {kwargs.get("fitnesses_len", 0)}\n'
    )
    filehandler.write(f'Precision scores: {scores["test_precision"]}\n')
    filehandler.write(f'Mean precision score: {scores["test_precision"].mean()}\n')
    filehandler.write(f'Recall scores: {scores["test_recall"]}\n')
    filehandler.write(f'Mean recall score: {scores["test_recall"].mean()}\n')
    filehandler.write(f'F1 scores: {scores["test_f1"]}\n')
    filehandler.write(f'Mean F1 score: {scores["test_f1"].mean()}\n')
    filehandler.write(f"Test set precision: {precisions}\n")
    filehandler.write(f"Test set mean precision: {np.mean(precisions)}\n")
    filehandler.write(f"Test set best precision: {np.max(precisions)}\n")
    filehandler.write(f"Test set recall: {recalls}\n")
    filehandler.write(f"Test set mean recall: {np.mean(recalls)}\n")
    filehandler.write(f"Test set best recall: {np.max(recalls)}\n")
    filehandler.write(f"Test set F1: {f1s}\n")
    filehandler.write(f"Test set mean F1: {np.mean(f1s)}\n")
    filehandler.write(f"Test set best F1: {np.max(f1s)}\n")


def train(
    df_main, cv=5, approach="dt", output_file="rq2.txt", platform="PID", **kwargs
):
    results_dict = {"Method": [], "Input": [], "Precision": [], "Recall": [], "F1": []}
    df_train = df_main.sample(frac=0.8, random_state=200)
    df_test = df_main.drop(df_train.index).reset_index(drop=True)
    df_train = balance(df_train, output_prefix=None)
    train = df_train.copy()
    test = df_test.copy()
    main_input = kwargs.get("input", "")
    with open(output_file, "a") as f:
        ##  With all the features...
        _train(train, test, cv, approach, results_dict, f, **kwargs)
        ##  Without weather features...
        kwargs["input"] = main_input + r" w\o weather"
        if platform == "PID":
            train = df_train.drop(["Weather1", "Weather2", "Weather3"], axis=1)
            test = df_test.drop(["Weather1", "Weather2", "Weather3"], axis=1)
        elif platform == "Pylot":
            train = df_train.drop(["Weather"], axis=1)
            test = df_test.drop(["Weather"], axis=1)
        elif platform.lower == "beamng":
            train = df_train.drop(["TimeOfDay", "Weather"], axis=1)
            test = df_test.drop(["TimeOfDay", "Weather"], axis=1)
        _train(train, test, cv, approach, results_dict, f, **kwargs)
        ##  Without blueprints...
        kwargs["input"] = main_input + r" w\\o blueprints"
        if platform == "PID":
            train = df_train.drop(["Ego", "NonEgo"], axis=1)
            test = df_test.drop(["Ego", "NonEgo"], axis=1)
        elif platform == "Pylot":
            train = df_train.drop(["RoadType", "RoadID"], axis=1)
            test = df_test.drop(["RoadType", "RoadID"], axis=1)
        elif platform.lower == "beamng":
            train = df_train.drop(["TrafficAmount"], axis=1)
            test = df_test.drop(["TrafficAmount"], axis=1)
        _train(train, test, cv, approach, results_dict, f, **kwargs)
        ##  Without weather and blueprints...
        kwargs["input"] = main_input + r" w\\o weather, blueprints"
        if platform == "PID":
            train = df_train.drop(
                ["Ego", "NonEgo", "Weather1", "Weather2", "Weather3"], axis=1
            )
            test = df_test.drop(
                ["Ego", "NonEgo", "Weather1", "Weather2", "Weather3"], axis=1
            )
        elif platform == "Pylot":
            train = df_train.drop(["Weather", "RoadType", "RoadID"], axis=1)
            test = df_test.drop(["Weather", "RoadType", "RoadID"], axis=1)
        elif platform.lower == "beamng":
            train = df_train.drop(["Weather", "TimeOfDay", "TrafficAmount"], axis=1)
            test = df_test.drop(["Weather", "TimeOfDay", "TrafficAmount"], axis=1)
        _train(train, test, cv, approach, results_dict, f, **kwargs)

    results_df = pd.DataFrame(results_dict)
    return results_df


def visualizeNumberOfClasses(df):
    sns.set_theme()
    plt.subplots(1, 5, figsize=(20, 5), layout="tight")
    for i in range(4):
        plt.subplot(1, 5, i + 1)
        sns.countplot(x=f"Y{i}", data=df)
        plt.xlabel(f"Fitness {i+1}")
    plt.subplot(1, 5, 5)
    sns.countplot(x="Y", data=df)
    plt.xlabel("Overall")
    log.info(
        f'Ratio of classes in the dataset: \n{df["Y"].value_counts(normalize=True)}'
    )
    plt.show()


if __name__ == "__main__":
    platform = "PID"
    output_folder = "output"
    thresh = 0.05
    ds_main = os.path.join(output_folder, f"ds_fit_{platform}_{int(thresh * 100)}.csv")
    ds_delta = os.path.join(
        output_folder, f"ds_delta_{platform}_{int(thresh * 100)}.csv"
    )
    df_main = pd.read_csv(ds_main)
    df_main = df_main.sample(frac=1, random_state=42).reset_index(drop=True)
    df_delta = pd.read_csv(ds_delta)
    df_delta = df_delta.sample(frac=1, random_state=42).reset_index(drop=True)
    ##  Visualize number of classes...
    visualizeNumberOfClasses(df_main)
    visualizeNumberOfClasses(df_delta)
    ##  Data preprocessing...
    df_main = df_main.drop(["Y0", "Y1", "Y2", "Y3"], axis=1)
    df_delta = df_delta.drop(["Y0", "Y1", "Y2", "Y3"], axis=1)
    ##  Training models...
    log_file = os.path.join(output_folder, f"rq2_{platform}_{int(thresh * 100)}.txt")
    os.remove(log_file) if os.path.exists(log_file) else None
    approaches = ["dt", "rf", "svm", "mlp"]
    results_df = pd.DataFrame(columns=["Method", "Input", "Precision", "Recall", "F1"])
    for approach in approaches:
        log.info(f"Tranining {approach}...")
        ##  With 1 set of fitnesses...
        df = df_main.copy()
        res = train(
            df,
            cv=5,
            approach=approach,
            output_file=log_file,
            max_depth=5,
            fitnesses_len=1,
            input="With 1 set of fitnesses",
            platform=platform,
        )
        results_df = results_df.append(res, ignore_index=True)
        ##  With delta fitnesses...
        df = df_delta.copy()
        res = train(
            df,
            cv=5,
            approach=approach,
            output_file=log_file,
            max_depth=5,
            fitnesses_len=2,
            input="With delta fitnesses",
            platform=platform,
        )
        results_df = results_df.append(res, ignore_index=True)

    results_df["Method"] = results_df["Method"].map(
        {"dt": "Decision Tree", "svm": "SVM", "mlp": "MLP", "rf": " Random Forest"}
    )
    results_df.sort_values(by=["F1"], ascending=False, inplace=True)
    csv_file = os.path.join(
        output_folder, f"rq2_models_{platform}_{int(thresh * 100)}.csv"
    )
    results_df.to_csv(csv_file, index=False)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    tex_output_file = os.path.join(
        output_folder, f"tab-rq2-models-{platform}-{int(thresh * 100)}.tex"
    )
    results_df.to_latex(
        tex_output_file,
        index=False,
        escape=False,
        float_format=lambda x: "%.2f" % x,
    )
    log.info(f"Finished training models for {platform} platform!")
    log.info(
        f"A summary of the results can be found in the following files: {csv_file} and {tex_output_file}"
    )
