import pandas as pd

from helper_funcs import convert_to_unix_time, df_filter


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


def calculate_errors(df, time_stamp_col, measurement_col, STG):
    """
    Adds columns to df that indicate whether each row has a timestamp error
    """
    df = df[[time_stamp_col, measurement_col]]
    df = df.drop_duplicates(keep="first")
    df = df.sort_values(by=time_stamp_col)
    # Add fields for all the timestamp errors for the measurement col
    df["gap"] = df[time_stamp_col] - df[time_stamp_col].shift()
    df["values_different"] = df[measurement_col] != df[measurement_col].shift()
    df["RT+CM"] = ((df["gap"] == 0) & (df["values_different"])).astype(int)
    df["STG+CM"] = (
        (df["gap"] > 0) & (df["gap"] < STG) & (df["values_different"])
    ).astype(int)
    df["STG-CM"] = (
        (df["gap"] > 0) & (df["gap"] < STG) & (~df["values_different"])
    ).astype(int)

    return df


def calculate_errors_with_duration(
    df, time_stamp_col, measurement_col, duration_col, STG, EAS_thresh
):
    """
    Adds columns to df that indicate whether each row has a timestamp error
    """
    df = df[[time_stamp_col, measurement_col, duration_col]]
    df = df.drop_duplicates(keep="first")
    df = df.sort_values(by=[time_stamp_col, duration_col])
    df["gap"] = df[time_stamp_col] - df[time_stamp_col].shift()
    df["calc_end_time"] = df[time_stamp_col] + df[duration_col]
    # df['EAS']=((df['gap'] >=STG)& ((df['calc_end_time'].shift()-df[time_stamp_col])>EAS_thresh)).astype(int)
    df["EAS"] = (
        ((df[time_stamp_col].shift(-1) - df[time_stamp_col]) >= STG)
        & (df["calc_end_time"] - df[time_stamp_col].shift(-1) > EAS_thresh)
    ).astype(int)
    df["cols_different"] = (df[duration_col] != df[duration_col].shift()) | (
        df[measurement_col] != df[measurement_col].shift()
    )
    df["RT+CM"] = ((df["gap"] == 0) & (df["cols_different"])).astype(int)
    df["STG+CM"] = (
        (df["gap"] > 0) & (df["gap"] < STG) & (df["cols_different"])
    ).astype(int)
    df["STG-CM"] = (
        (df["gap"] > 0) & (df["gap"] < STG) & (~df["cols_different"])
    ).astype(int)

    return df


def calculate_error_with_endtime(
    df, time_stamp_col, measurement_col, end_time_col, STG, EAS_thresh
):
    """
    Adds columns to df that indicate whether each row has a timestamp error
    """
    df = df[[time_stamp_col, measurement_col, end_time_col]]
    df = df.drop_duplicates(keep="first")
    df = df.sort_values(by=[time_stamp_col, end_time_col])
    df["gap"] = df[time_stamp_col] - df[time_stamp_col].shift()
    df["EAS"] = (
        ((df[time_stamp_col].shift(-1) - df[time_stamp_col]) >= STG)
        & (df[end_time_col] - df[time_stamp_col].shift(-1) > EAS_thresh)
    ).astype(int)
    df["cols_different"] = (df[end_time_col] != df[end_time_col].shift()) | (
        df[measurement_col] != df[measurement_col].shift()
    )
    df["RT+CM"] = ((df["gap"] == 0) & (df["cols_different"])).astype(int)
    df["STG+CM"] = (
        (df["gap"] > 0) & (df["gap"] < STG) & (df["cols_different"])
    ).astype(int)
    df["STG-CM"] = (
        (df["gap"] > 0) & (df["gap"] < STG) & (~df["cols_different"])
    ).astype(int)

    return df


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
            "RT+CM": "max",
            "STG+CM": "max",
            "STG-CM": "max",
            "EAS": "max",
        }
    )
    return df


def get_timestamp_errors_and_clean(
    df,
    interval,
    time_stamp_col,
    measurement_col,
    STG,
    EAS_thresh=None,
    STG_fix=False,
    meas_agg="mean",
    end_time_col=None,
    duration_col=None,
    filter_dict=None,
    convert_to_unix=None,
    included_errors=["RT+CM", "STG+CM", "STG-CM", 'EAS'],
    time_zone="Europe/London",
):
    """
    Produces metadata that can accompany any features extracted from the field
    'measurement_col' in the input df. This includes a total count of datapoints per
    interval and counts of each type of timestamp error per interval, calculated
    based on the timestamp field, 'time_stamp_col'. Timestamp errors are calculated
    from STG, end_time_col, and duration_col. The output is extended using extended_index
    if it is set. For more info on the timestamp errors, see Data analysis book introduction.
    Also cleans the input df to fix RT+CM errors according to values of time_stamp_agg
    and meas_agg, and cleans STG errors if STG_fix is set to True.
    Returns:
        df: A cleaned version of the input df
        output_series: a dataframe containing metadata at the specified interval
    """
    # TODO consider also cleaning (and counting) when end is before start on same datapoint
    # TODO add option for worst case EAS (ascending order) or best case EAS (descending order)
    # TODO check if there are any errors before calling the cleaning function (will require calculating total error earlier)
    # TODO think about whether you can extend this to having more than one measurement col
    # TODO consider included a threshold for EAS as an option so EAS is only recorded if it is above that threshold
    df = df_filter(df, filter_dict)
    # convert to unix time if neccessary
    if convert_to_unix is not None:
        df = convert_to_unix_time(df, convert_to_unix, time_zone)
    if end_time_col == None and duration_col == None:
        df = calculate_errors(df, time_stamp_col, measurement_col, STG)
        df = get_group_ids(df, time_stamp_col, STG_fix, STG)
        # Clean the input df of RT+CM and STG
        df = df.groupby("group", as_index=False).agg(
            {
                measurement_col: meas_agg,
                time_stamp_col: "min",
                "RT+CM": "max",
                "STG+CM": "max",
                "STG-CM": "max",
            }
        )
        if "EAS" in included_errors:
            included_errors.remove("EAS")
    if end_time_col != None:
        df = calculate_error_with_endtime(
            df, time_stamp_col, measurement_col, end_time_col, STG, EAS_thresh
        )
        df = clean_errors_with_durations(
            df, STG_fix, STG, time_stamp_col, measurement_col, meas_agg, end_time_col
        )
    if duration_col != None:
        df = calculate_errors_with_duration(
            df, time_stamp_col, measurement_col, duration_col, STG, EAS_thresh
        )
        df = clean_errors_with_durations(
            df, STG_fix, STG, time_stamp_col, measurement_col, meas_agg, "calc_end_time"
        )
        df[duration_col] = df["calc_end_time"] - df[time_stamp_col]

    # Convert df to right format for resampling
    df["value.time.day"] =pd.to_datetime(df[time_stamp_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
    index_to_drop = df[
        df["value.time.day"].dt.year == 1970
    ].index  # Remove rows where the year is 1970, which indicates no data was recorded
    df.drop(index_to_drop, inplace=True)
    df.set_index("value.time.day", inplace=True)

    # Create a new column in df for total errors
    df["total timestamps with any error"] = df[included_errors].max(axis=1)

    # Extract metadata features from df
    df_errors = df.loc[:, included_errors + ["total timestamps with any error"]].copy()
    df_errors["total counts"] = 1
    output_series = df_errors.resample(interval).sum()

    return df, output_series
