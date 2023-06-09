from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import logging as log
from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import utils
import time
import os

log.basicConfig(level=log.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def generateFitDataset(
    scenarios,
    f_arr_detailed: np.array,
    flaky_variation_thresh: list,
    fitness_window_len=3,
    n_rep=10,
    output_file="dataset.csv",
    platform="PID",
):
    t0 = time.time()
    no_of_fitnesses = f_arr_detailed.shape[-1]
    ##  Calculate labels...
    deltas = np.max(f_arr_detailed, axis=1) - np.min(
        f_arr_detailed, axis=1
    )  ##  (no_of_scenarios, no_of_fitnesses)
    delta_max = np.max(deltas, axis=0)  ##  (no_of_fitnesses,)
    Y = ((deltas / delta_max) >= flaky_variation_thresh).astype(
        int
    )  ##  (no_of_scenarios, no_of_fitnesses)
    ##  Calculate features...
    if platform == "PID":
        columns = [
            "Weather1",
            "Weather2",
            "Weather3",
            "Ego",
            "NonEgo",
            "Front",
            "Back",
            "Opposite",
        ]
    elif platform == "Pylot":
        columns = [
            "RoadType",
            "RoadID",
            "ScenarioLength",
            "VehicleFront",
            "VehicleAdjacent",
            "VehicleOpposite",
            "VehicleFrontTwoWheeled",
            "VehicleAdjacentTwoWheeled",
            "VehicleOppositeTwoWheeled",
            "TimeOfDay",
            "Weather",
            "NumberOfPeople",
            "TargetSpeed",
            "Trees",
            "Buildings",
            "Task",
        ]
    elif platform.lower() == "beamng":
        columns = [
            "TargetIndex",
            "TimeOfDay",
            "EgoSpeed",
            "TrafficAmount",
            "KeepLane",
            "Weather",
        ]
    columns.extend(
        [f"F{i}_{j}" for i in range(no_of_fitnesses) for j in range(fitness_window_len)]
    )
    columns.extend([f"Y{i}" for i in range(no_of_fitnesses)] + ["Y"])
    data = []
    for i in range(len(scenarios)):
        scenario = scenarios[i].copy()
        data.append(scenario)
        for j in range(fitness_window_len):
            data[-1].extend(f_arr_detailed[i][j].tolist())
        for j in range(no_of_fitnesses):
            data[-1].append(Y[i][j])
        data[-1].append(np.max(Y[i]))
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    t1 = time.time()
    log.info(f"Output: {output_file} generated in {t1-t0:.2f} seconds...")
    return df


def balance(df, target="Y", output_prefix=None, platform="PID"):
    targets = ["Y"]
    X = df.drop(targets, axis=1).values
    y = df[target].values
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    df_res = pd.DataFrame(X_res, columns=df.drop(targets, axis=1).columns)
    df_res[target] = y_res
    if output_prefix is not None:
        output_file = f"{output_prefix}_{platform}.csv"
        df_res.to_csv(output_file, index=False)
    return df_res


def generateDeltaDataset(
    scenarios,
    f_arr_detailed: np.array,
    flaky_variation_thresh: list,
    fitness_window_len=3,
    n_rep=10,
    output_file="dataset_delta.csv",
    platform="PID",
):
    t0 = time.time()
    no_of_fitnesses = f_arr_detailed.shape[-1]
    ##  Calculate labels...
    deltas = np.max(f_arr_detailed, axis=1) - np.min(
        f_arr_detailed, axis=1
    )  ##  (no_of_scenarios, no_of_fitnesses)
    delta_max = np.max(deltas, axis=0)  ##  (no_of_fitnesses,)
    Y = ((deltas / delta_max) >= flaky_variation_thresh).astype(
        int
    )  ##  (no_of_scenarios, no_of_fitnesses)
    ##  Calculate features...
    if platform == "PID":
        columns = [
            "Weather1",
            "Weather2",
            "Weather3",
            "Ego",
            "NonEgo",
            "Front",
            "Back",
            "Opposite",
        ]
    elif platform == "Pylot":
        columns = [
            "RoadType",
            "RoadID",
            "ScenarioLength",
            "VehicleFront",
            "VehicleAdjacent",
            "VehicleOpposite",
            "VehicleFrontTwoWheeled",
            "VehicleAdjacentTwoWheeled",
            "VehicleOppositeTwoWheeled",
            "TimeOfDay",
            "Weather",
            "NumberOfPeople",
            "TargetSpeed",
            "Trees",
            "Buildings",
            "Task",
        ]
    elif platform.lower() == "beamng":
        columns = [
            "TargetIndex",
            "TimeOfDay",
            "EgoSpeed",
            "TrafficAmount",
            "KeepLane",
            "Weather",
        ]
    columns.extend([f"Delta_F{i}" for i in range(no_of_fitnesses)])
    columns.extend([f"Y{i}" for i in range(no_of_fitnesses)] + ["Y"])
    data = []
    for i in range(len(scenarios)):
        scenario = scenarios[i].copy()
        for j in range(2, fitness_window_len + 1):
            data.append(scenario.copy())
            delta = np.max(f_arr_detailed[i, :j], axis=0) - np.min(
                f_arr_detailed[i, :j], axis=0
            )  ##  Delta fitness up to j...
            assert delta.shape == (no_of_fitnesses,)
            data[-1].extend(delta.tolist())
            for k in range(no_of_fitnesses):
                data[-1].append(Y[i][k])
            data[-1].append(np.max(Y[i]))
    df = pd.DataFrame(data, columns=columns)
    t1 = time.time()
    log.info("Shape of delta dataset before removing duplicates: {}".format(df.shape))
    df = df.drop_duplicates()
    df.to_csv(output_file, index=False)
    log.info("Shape of delta dataset after removing duplicates: {}".format(df.shape))
    log.info(f"Output: {output_file} generated in {t1-t0:.2f} seconds...")
    return df


if __name__ == "__main__":
    output_folder = "output"
    platform = "PID"
    csv_file = f"ds_{platform.lower()}.csv"
    thresh = 0.05

    result_detailed, f_arr_detailed = utils.fitnessForAllScenarios(
        platform=platform, csv_file=csv_file, normalize=False
    )  ##  f_arr_detailed.shape = (no_of_scenarios, no_of_repeats, no_of_fitnesses)
    result_detailed, f_arr_detailed = result_detailed[:1000], f_arr_detailed[:1000]
    scenarios, f_arr = utils.minAndMeanFitnesses(
        result_detailed
    )  ##  f_arr.shape = (no_of_scenarios, 2, no_of_fitnesses)
    flaky_variation_thresh = [thresh for i in range(4)]
    df = generateFitDataset(
        scenarios,
        f_arr_detailed,
        flaky_variation_thresh,
        1,
        platform=platform,
        output_file=os.path.join(
            output_folder, f"ds_fit_{platform}_{int(thresh * 100)}.csv"
        ),
    )

    ##  Number of each class...
    for i in range(4):
        print(f'Number of flaky fitness {i}: {df[df[f"Y{i}"]==1].shape[0]}')
    print(f'Number of overall flaky scenarios: {df[df["Y"]==1].shape[0]}')
    log.info(f"Shape of fit dataset before balancing: {df.shape}")
    df = balance(
        df,
        target="Y",
        output_prefix=os.path.join(
            output_folder, f"ds_fit_balanced_{int(thresh * 100)}"
        ),
        platform=platform,
    )
    log.info(f"Shape of fit dataset after balancing: {df.shape}")
    df_delta = generateDeltaDataset(
        scenarios,
        f_arr_detailed,
        flaky_variation_thresh,
        10,
        platform=platform,
        output_file=os.path.join(
            output_folder, f"ds_delta_{platform}_{int(thresh * 100)}.csv"
        ),
    )
    log.info("Shape of delta dataset: {}".format(df_delta.shape))

    ##  Number of each class...
    for i in range(4):
        print(f'Number of flaky fitness {i}: {df_delta[df_delta[f"Y{i}"]==1].shape[0]}')
    print(f'Number of overall flaky scenarios: {df_delta[df_delta["Y"]==1].shape[0]}')
