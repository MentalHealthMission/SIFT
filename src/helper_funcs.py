import datetime
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_boxplot(df: pd.DataFrame, title: str):
    """
    Draws a boxplot of the values in the dataframe with the assigned title.
    """
    _, ax = plt.subplots()
    ax.boxplot(
        df,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0", alpha=0.9),
        medianprops=dict(color="orange", linewidth=2),
    )
    median_value = np.median(df)
    ax.text(
        1.1,  # x position (slightly right of box)
        median_value,  # y position at the median
        f"Median = {median_value:.3f}",
        va="center",
        fontsize=10,
    )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.gca().set_xticks([])

    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.title(title)
    plt.show()


def get_file_paths(
    input_folder,
    csv_name,
    Folder_structure,
    site_list,
):
    """
    Get a list of the paths to all the files with name csv_name in input_folder in the subfolders
    found in site_list.
    """
    all_paths = []
    if Folder_structure == 1:
        for site in site_list:
            participant_list = [
                f.name for f in os.scandir(input_folder + site) if f.is_dir()
            ]
            for participant in participant_list:
                subfolders = [
                    f.name
                    for f in os.scandir(input_folder + site + "/" + participant)
                    if f.is_dir()
                ]
                if csv_name in subfolders:
                    file_name = (
                        input_folder
                        + site
                        + "/"
                        + participant
                        + "/"
                        + csv_name
                        + "/"
                        + csv_name
                        + ".csv.gz"
                    )
                    if os.path.isfile(file_name):
                        all_paths.append(file_name)

    if Folder_structure == 2:
        for site in site_list:
            participant_list = [
                f.name for f in os.scandir(input_folder + site) if f.is_dir()
            ]
            for participant in participant_list:
                if csv_name + ".csv" in os.listdir(
                    input_folder + site + "/" + participant
                ):
                    all_paths.append(
                        input_folder
                        + site
                        + "/"
                        + participant
                        + "/"
                        + csv_name
                        + ".csv"
                    )

    print(len(all_paths), "files found")
    return all_paths


def summary_stats(LIST, thresh):
    """
    Returns summary statistics for the values in LIST
    """
    counts = Counter(LIST)
    mode, mode_count = counts.most_common(1)[0]
    less_than_mode = sum(1 for x in LIST if x < mode - thresh)
    close_to_mode = sum(1 for x in LIST if (x > mode - thresh and x < mode + thresh))
    av = sum(LIST) / len(LIST)
    med = statistics.median(LIST)
    min_list = min(LIST)
    max_list = max(LIST)
    summary_list = [
        av,
        med,
        min_list,
        max_list,
        mode,
        mode_count / (len(LIST)),
        less_than_mode / (len(LIST)),
        close_to_mode / (len(LIST)),
    ]

    return summary_list


def all_summary_stats(cleaned_list: list):
    """
    Returns summary statistics for the values in cleaned_list
    """
    cleaned_list.sort()
    LQT_ind = int(len(cleaned_list) * 0.25)
    UQT_ind = int(len(cleaned_list) * 0.75)
    P1_ind = int(len(cleaned_list) * 0.01)
    P99_ind = int(len(cleaned_list) * 0.99)
    LQT = cleaned_list[LQT_ind]
    UQT = cleaned_list[UQT_ind]
    P1 = cleaned_list[P1_ind]
    P99 = cleaned_list[P99_ind]
    med = statistics.median(cleaned_list)
    min_list = min(cleaned_list)
    max_list = max(cleaned_list)

    return [min_list, P1, LQT, med, UQT, P99, max_list]


def get_participant_and_site(file_path: str):
    """
    Get the participant and site from file_path
    """
    path = Path(file_path)
    if file_path[-3:] == "csv":
        participant = path.parent.name
        site = path.parents[1].name
    if file_path[-3:] == ".gz":
        participant = path.parents[1].name
        site = path.parents[2].name

    return participant, site


def df_filter(df, filter_dict):
    """
    Filters the rows in df based on values in filter_dict
    """
    if filter_dict is not None:
        # This option will filter the df, keeping only rows where allowed values
        # are in the specified columns
        df = df[
            pd.concat(
                [df[col].isin(values) for col, values in filter_dict.items()], axis=1
            ).all(axis=1)
        ].copy()

    return df


def convert_to_unix_time(df: pd.DataFrame, cols: list, time_zone="utc"):

    for col in cols:

        df[col] = (
    pd.to_datetime(df[col])
      .dt.tz_localize(time_zone)
      .astype('int64') / 10**9
)

    return df