import os

import pandas as pd

from helper_funcs import convert_to_unix_time, df_filter


def investigate_sleep_blocks(
    files_list: list[str],
    timestamp_col: str,
    sleep_level_col: str,
    awake_string: str,
    gap_thresh: float,
    duration_col=None,
    end_time_col=None,
    convert_to_unix=None,
    filter_dict=None,
    time_zone="Europe/London",
):
    """
    Returns a list of the length of each 'block' of sleep across all the files in files_list.
    """
    all_block_durations = []
    all_TSTs = []
    
    for path in files_list:

        #Read in file and filter to useful rows if neccesary.
        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except: 
            print(path + " file cannot be read")
            continue

        df = df_filter(df, filter_dict)

        if len(df) > 0:

            # Prepare the df for calcualting sleep blocks.
            if convert_to_unix is not None:
                df = convert_to_unix_time(df, convert_to_unix[0], convert_to_unix[1])
            
            df[timestamp_col] = df[timestamp_col].astype(float) # Do this to prevent warning further down.
            
            # Clean timestamp errors
            df = df.sort_values(by=timestamp_col).reset_index(drop=True)
            if end_time_col is None:
                df['end_time'] = df[timestamp_col]+df[duration_col]
            else:
                df['end_time'] = df[end_time_col]
            df=clean_errors_with_durations(df, False, 30, timestamp_col, sleep_level_col, 'first', 'end_time')
            df['duration'] = df['end_time'] - df[timestamp_col]
            
            #Remove awake datapoints.
            df=df[df[sleep_level_col]!=awake_string].copy()
            
            # create a column 'group' that assigns the same number to all datapoints in the same sleep block
            gaps = df[timestamp_col]-df['duration'].shift().fillna(0)-df[timestamp_col].shift().fillna(0) # time between timestamp and previous end time
            group_ids = (gaps > gap_thresh).cumsum()  # Start a new group when difference > thresh
            df["group"] = group_ids

            # aggregate to get start and end time of each block
            df = df.groupby("group", as_index=False).agg(
                {
                    timestamp_col: "first",
                    'end_time': "last",
                    'duration': "sum",
                }
            )

            #Calculate the block duration, get max in each day (PSP), then add all non zero PSPs to list. 
            df['block_duration'] = df['end_time'] - df[timestamp_col]
            df['block_duration'] = df['block_duration'] / 3600
            df[timestamp_col] = (pd.to_datetime(df['end_time'], unit="s", utc=True).dt.tz_convert(time_zone).dt.floor('D').astype('int64') // 10**9)
            max_blocks = df.loc[df.groupby(timestamp_col)['block_duration'].idxmax(), [timestamp_col, 'block_duration', 'duration']]
            all_block_durations.extend(max_blocks['block_duration'])
            all_TSTs.extend(max_blocks['duration']/3600)

    return all_block_durations, all_TSTs


def find_time_of_timestamps(
    all_file_paths, timestamp_col, convert_to_unix=None, filter_dict=None, time_zone="Europe/London"
):
    """
    Returns a dictionary that reports how often each time of day occurs in the timestamp_col column
    over all the files in all_file_paths
    """
    all_hours = []
    for path in all_file_paths:
        # read in file
        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except Exception as e:
            print(path + " file cannot be read, error: " + str(e))
            continue
        df = df_filter(df, filter_dict)
        # convert to unix time if necessary
        if convert_to_unix is not None:
            df = convert_to_unix_time(df, convert_to_unix)
        # Add hour of timestamp to list 'all_hours'
        df["value.time.day"] =pd.to_datetime(df[timestamp_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
        df["hour"] = df["value.time.day"].dt.strftime("%H:%M:%S")
        all_hours = all_hours + list(df["hour"])

    # Convert 'all_hours' into dictionary summarising how often each hour occurred
    d = dict.fromkeys(all_hours, 0)
    for val in all_hours:
        d[val] += 1
    return d


def time_gap_freqs(
    all_file_paths, output_path, time_stamp="value.time", filter_dict=None
):
    """
    Counts time gap frequencies.
    """
    all_data = pd.DataFrame()
    for path in all_file_paths:
        # read in file
        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except Exception as e:
            print(path + " file cannot be read, error: " + str(e))
            continue

        df = df_filter(df, filter_dict)
        # get rid of all columns except timestamp
        df = df[[time_stamp]]
        df = df[~df[time_stamp].duplicated(keep="first")]
        df = df.sort_values(by=time_stamp).reset_index(drop=True)
        df["gap"] = df[time_stamp].diff().fillna(0)
        df = df[["gap"]]
        all_data = pd.concat([all_data, df], ignore_index=True)

    counts_df = all_data.value_counts().reset_index()
    counts_df["fraction"] = counts_df["count"] / len(all_data)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    counts_df.to_csv(output_path + "time_gaps.csv", index=True)
    df_first15 = counts_df.head(15)

    return df_first15

def clean_errors_with_durations(
    df, STG_fix, STG, time_stamp_col, measurement_col, meas_agg, end_time_col
):
    """
    Cleans df of all timestamp errors according to the following rules:
    1.datapoints with duration of 0 deleted
    2.if duration overlaps next datapoint, it is capped to the time gap between this datapoint and
    the next (the maximum possible duration)
    3.if there are multiple durations for a timestamp, the highest duration that does not overlap
    next datapoint is taken as correct duration, if all durations overlap than the maximum possible
    duration used.
    4. In the case of RT+CM, measured value is calculated according to meas_agg from all timestamps
    that originally had the 'correct' duration (i.e the one that was there after the rule above),
    or all timestamps if all datapoints originally overlapped. If all durations were the same
    originally, then 'correct' measurement is just calculated from all timestamps.
    5. if STG_fix is true, STG errors are treated as RT+CM errors - make gap sum instead? In this case,
    timestamp_agg always has to be min for data with durations to avoid making gaps.
    """
    df = get_group_ids(df, time_stamp_col, STG_fix, STG)
    next_group_time = df.groupby("group")[time_stamp_col].first()
    df["next_group_time"] = df["group"].map(next_group_time.shift(-1))

    df.loc[df[end_time_col] > df["next_group_time"], end_time_col] = 0
    df["max end time"] = df.groupby("group")[end_time_col].transform(
        "max"
    )  # get a column that is the max in the group
    df.loc[df[end_time_col] < df["max end time"], end_time_col] = 0
    sum_ = df.groupby("group")[end_time_col].sum()
    df["sum"] = df["group"].map(sum_)
    df["use_datapoint"] = ~((df[end_time_col] == 0) & (df["sum"] != 0))
    df[end_time_col] = df.groupby("group")[end_time_col].transform("max")
    df.loc[df[end_time_col] == 0, end_time_col] = df["next_group_time"]
    mean_nonzero = (
        df[df["use_datapoint"]].groupby("group")[measurement_col].agg(meas_agg)
    )
    df[measurement_col] = df["group"].map(mean_nonzero)

    df = df.groupby("group", as_index=False).agg(
        {
            end_time_col: "first",
            measurement_col: "first",
            time_stamp_col: "min",
            
        }
    )
    return df

def get_group_ids(df, time_stamp_col, STG_fix, STG):
    """
    Add a column to df that assigns a 'group' to each row - all rows with same
    timestamp, or timestamps within STG if STG_fix is True, will be assigned
    same group.
    """
    gaps = df[time_stamp_col].diff().fillna(0)
    if STG_fix:
        group_ids = (gaps >= STG).cumsum()  # Start a new group when difference > STG
    if not STG_fix:
        group_ids = (gaps > 0).cumsum()  # Start a new group when difference > 0
    df["group"] = group_ids

    return df