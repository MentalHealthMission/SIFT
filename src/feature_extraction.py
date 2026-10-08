import numpy as np
import pandas as pd
from clean_and_extract_features import get_timestamp_errors_and_clean
from helper_funcs import convert_to_unix_time, df_filter

def round_timestamp_to_midnight(df, timestamp_col,time_zone="Europe/London"):
    """
    Rounds all values in timestamp_col to the nearest midnight 
    """
    df["value.time.day"] =pd.to_datetime(df[timestamp_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
    df["hour"] = df["value.time.day"].dt.hour + df["value.time.day"].dt.minute / 60
    df["hour"] = df["hour"].apply(lambda x: x - 24 if x > 12 else x)
    # Round timestamp to nearest midnight
    df["rounded"] = df["value.time.day"].apply(
        lambda ts: (ts + pd.Timedelta(minutes=720)).normalize()
    )
    df[timestamp_col] = df["rounded"].astype("int64") // 10**9
    cleaned_df, features = get_timestamp_errors_and_clean(
        df=df,
        interval="D",
        time_stamp_col=timestamp_col,
        measurement_col="hour",
        STG=86400,
        meas_agg="mean",
    )
    hours = get_fixed_series(
        cleaned_df, "D", "mean", "hour", timestamp_col, "hour of datapoint"
    )

    return df, cleaned_df, hours


def get_extra_HR_metadata_features(
    cleaned_df,
    timestamp_col,
    meas_col,
    max_gap,
    interval,
    low_thresh=30,
    upper_thresh=250,
    end_time_col=None,
    duration_col=None,
    included_errors=["RT+CM", "STG+CM", "STG-CM", "EAS"],
    time_zone="Europe/London",
):
    """
    Returns 'metadata_features', a df with all the required metadata features for HR (including 
    number filtered and coverage) at the frequency given by 'interval' for input dataframe 
    cleaned_df. Also returns an updated version of cleaned_df that includes the filtered steps 
    column and number filtered column.
    The meas_col is the name of the heart rate column, timestamp_col is the name of the timestamp 
    column, max_gap is the maximum expected gap between datapoints, low_thresh and upper_thresh are
    the thresholds for filtering HR data, end_time_col and duration_col are the names of the end 
    time and duration columns if they are included in the input df, included_errors is a list 
    of the error columns to include in the metadata features, and time_zone is the timezone that 
    the timestamps are in.
    """
    #Ensure index is a datetime object in the right timezone
    cleaned_df["value.time.day"] =pd.to_datetime(cleaned_df[timestamp_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
    cleaned_df.set_index("value.time.day", inplace=True)

    # Calculate filtered steps and get count
    cleaned_df["filtered"] = cleaned_df[meas_col].clip(lower=low_thresh, upper=upper_thresh)
    cleaned_df["Number filtered"] = (cleaned_df["filtered"] != cleaned_df[meas_col]).astype(int)

    # Create a new column in df for total errors
    if (end_time_col == None and duration_col == None):  # This is just in case EAS was in included_errors by mistake
        if "EAS" in included_errors:
            included_errors.remove("EAS")

    included_errors.append("Number filtered")
    cleaned_df["total timestamps with any error"] = cleaned_df[included_errors].max(axis=1)

    # Extract metadata features from df
    df_errors = cleaned_df.loc[
        :, included_errors + ["total timestamps with any error"]
    ].copy()
    df_errors["total counts"] = 1
    features = df_errors.resample(interval).sum()

    coverage_features = get_coverage(
        cleaned_df.copy(),
        timestamp_col,
        max_gap,
        interval,
        "Coverage (secs) from all datapoints",
        end_time_col,
        duration_col,
        time_zone
    )
    only_clean = cleaned_df[cleaned_df["total timestamps with any error"] == 0]
    coverage_features_filtered = get_coverage(
        only_clean.copy(),
        timestamp_col,
        max_gap,
        interval,
        "Coverage (secs) from clean datapoints",
        end_time_col,
        duration_col,
        time_zone
    )

    # merge all metadata
    metadata_features = pd.concat(
        [features, coverage_features, coverage_features_filtered], axis=1
    )
    metadata_features.index.name = cleaned_df.index.name

    return metadata_features, cleaned_df

def get_fixed_series(
    df,
    interval,
    agg,
    meas_col,
    timestamp_col,
    new_name,
    extended_index=None,
    time_type="unix",
    time_zone="Europe/London"
):
    """
    Takes in an input df and produces an output df that contains a resampled series of the
    field meas_col with the specified interval, resampling method defined by agg. The name
    of the field in the output df is set by new_name, and is extended by extended_index if
    it is set.
    Returns:
        output_series: a df containing a resampled series for a field of the input df.
    """

    if time_type=='unix':
        df["value.time.day"] =pd.to_datetime(df[timestamp_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
        timestamp_col = "value.time.day"
    index_to_drop = df[
        df[timestamp_col].dt.year == 1970
    ].index  # Remove rows where the year is 1970, which indicates no data was recorded
    df.drop(index_to_drop, inplace=True)
    df.set_index(timestamp_col, inplace=True)

    # Carry out the resampling
    df = df.loc[:, [meas_col]].copy()
    if agg == "count":
        output_series = df.resample(interval).count()
    if agg == "max":
        output_series = df.resample(interval).max()
    if agg == "min":
        output_series = df.resample(interval).min()
    if agg == "sum":
        output_series = df.resample(interval).sum()
    if agg == "mean":
        output_series = df.resample(interval).mean()

    # reindex the output series so it is the required length
    if extended_index is not None:
        output_series = output_series.reindex(extended_index).fillna(0)

    # rename the measurement column
    output_series.rename(columns={meas_col: new_name}, inplace=True)

    return output_series



def find_durations(df, start_col, end_col, interval, meas_col=None, time_zone="Europe/London"):
    """
    Returns a version of df with end_col and start_col converted to datetime objects and extra column
    'seconds_diff' which gives the duration of each datapoint. Any datapoints overlapping the interval
    (e.g hour/day) have been split into separate datapoints.
    meas_col is a numerical measurement column that, if not None, will be split proportionally between 
    the two new datapoints when a datapoint is split by the interval. 
    The incoming df should already be cleaned such that the values in end_col are always after
    the values in start_col
    """
    if meas_col is not None:
        df["duration"] = df[end_col] - df[start_col]
    # Convert df to right format for resampling
    df[start_col] = pd.to_datetime(df[start_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
    df[end_col] = pd.to_datetime(df[end_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
    df = split_intervals(df, interval, start_col, end_col)
    df["seconds_diff"] = (df[end_col] - df[start_col]).dt.total_seconds()
    if meas_col is not None:
        df[meas_col] = df[meas_col] * df["seconds_diff"] / df["duration"]

    return df


def split_intervals(df, freq, start_col, end_col):
    """
    Returns a version of df where any rows where start_col and end_col are in different intervals
    (e.g different hours/days) have been split into separate rows, with the start_col and end_col values 
    adjusted so that each row now only covers a single interval. The interval is defined by freq, 
    which should be a pandas offset alias (e.g 'H' for hourly, 'D' for daily etc).
    """
    df = df.copy()
    # Create interval boundaries per row
    df["boundary"] = df.apply(
        lambda r: pd.date_range(
            r[start_col].floor(freq), r[end_col].ceil(freq), freq=freq
        ),
        axis=1,
    )
    # Explode
    df = df.explode("boundary", ignore_index=True)
    # Compute new start
    df["new_start"] = df[[start_col, "boundary"]].max(axis=1)
    # Compute next boundary
    df["next_boundary"] = df["boundary"] + pd.tseries.frequencies.to_offset(freq)
    # Compute new end
    df["new_end"] = df[[end_col, "next_boundary"]].min(axis=1)
    # Keep only valid intervals
    df = df[df["new_start"] < df["new_end"]]
    # Clean up
    df = df.drop(columns=[start_col, end_col, "boundary", "next_boundary"])
    df = df.rename(columns={"new_start": start_col, "new_end": end_col})

    return df.reset_index(drop=True)

def weighted_average(
    df,
    timestamp_col,
    meas_col,
    max_time_gap,
    interval,
    col_name,
    end_time_col=None,
    duration_col=None,
    time_zone="Europe/London"
):
    """
    Returns 'final_df', a df that reports the weighted average of meas_col for the input interval for the
    data in the input df.
    timestamp_col is the name of the timestamp column, max_gap is the maximum expected gap between 
    datapoints, col_name is the name of the column to store the weighted average in the output df, 
    end_time_col and duration_col are the names of the end time and duration columns if they 
    are included in the input df, and time_zone is the timezone that the timestamps are in.
    """
    df, timestamp_col, duration_col=get_adjusted_timings(
        df, 
        timestamp_col, 
        max_time_gap, 
        interval, 
        end_time_col, 
        duration_col, 
        time_zone
    )
    df["weighted"] = df[meas_col] * df[duration_col]
    total_duration = get_fixed_series(
        df.copy(), interval, "sum", duration_col, timestamp_col, "total_duration", time_type="datetime"
    )  
    total_weighted = get_fixed_series(
        df.copy(), interval, "sum", "weighted", timestamp_col, "total_weighted", time_type="datetime"
    )
    final_df = pd.concat([total_duration, total_weighted], axis=1)
    final_df[col_name] = np.where(
        final_df["total_duration"] == 0,
        -1,
        final_df["total_weighted"] / final_df["total_duration"],
    )
    final_df = final_df[[col_name]]

    return final_df


def get_coverage(
    df,
    timestamp_col,
    max_time_gap,
    interval,
    col_name,
    end_time_col=None,
    duration_col=None,
    time_zone="Europe/London",
):
    """
    Returns a dataframe 'total_duration' that reports the amount of time covered by a datapoint each 
    interval (e.g hour/day) for the data in the input df.
    timestamp_col is the name of the timestamp column, max_gap is the maximum expected gap between 
    datapoints, col_name is the name of the column to store the weighted average in the output df, 
    end_time_col and duration_col are the names of the end time and duration columns if they 
    are included in the input df, and time_zone is the timezone that the timestamps are in.
    """

    df, timestamp_col, duration_col=get_adjusted_timings(
        df, 
        timestamp_col, 
        max_time_gap, 
        interval, 
        end_time_col, 
        duration_col, 
        time_zone
    )
    total_duration = get_fixed_series(
        df, interval, "sum", duration_col, timestamp_col, col_name, time_type="datetime"
    ) 

    return total_duration


def get_adjusted_timings(df, timestamp_col, max_time_gap, interval, end_time_col=None, duration_col=None, time_zone="Europe/London"):
    """
    Returns a version of df where rows overlapping the interval are split into multiple rows and a new 
    column 'seconds_diff' has been created which gives the updated duration of each datapoint. The 
    timestamp_col (and end_time_col if there is one) is also converted to datetime. 
    If end_time_col and duration_col are both None, the start time and end time are calculated for 
    each datapoint using the neighbouring datapoints and max_time_gap (the expected maximum gap between 
    datapoints).
    """
    if duration_col != None:
        df["end_time"] = df[timestamp_col] + df[duration_col]
        df = find_durations(df, timestamp_col, "end_time", interval, time_zone=time_zone)
    if end_time_col == None and duration_col == None:
        df["start_time_1"] = df[timestamp_col] - (max_time_gap / 2)
        df["start_time_2"] = 0.5 * (
            df[timestamp_col] + df[timestamp_col].shift().fillna(0)
        )
        df["start_time_col"] = df[["start_time_1", "start_time_2"]].max(axis=1)
        df["end_time_1"] = df[timestamp_col] + (max_time_gap / 2)
        df["end_time_2"] = 0.5 * (
            df[timestamp_col] + df[timestamp_col].shift(-1).fillna(100000000000000)
        )
        df["end_time_col"] = df[["end_time_1", "end_time_2"]].min(axis=1)
        df = find_durations(df, "start_time_col", "end_time_col", interval, time_zone=time_zone)
        timestamp_col = "start_time_col" 
    if duration_col == None and end_time_col != None:
        df = find_durations(df, timestamp_col, end_time_col, interval, time_zone=time_zone)
    duration_col = "seconds_diff"

    return df, timestamp_col, duration_col


def get_daily_sleep_features(
    df,
    timestamp_col: str,
    sleep_level_col: str,
    awake_string: str,
    light_string:str,
    deep_string:str,
    rem_string:str,
    gap_thresh: float,
    duration_col=None,
    end_time_col=None,
    filter_dict=None,
    time_zone="Europe/London",
):
    """
    Returns daily sleep features for incoming df, see sleep chapters for full details on 
    features extracted. timestamp_col is the name of the column that gives the start time 
    of the datapoint, duration_col is the name of the column that gives the duration, 
    end_time_col is the name of the column that gives the end_time, and sleep_level_col is the name
    of the column that gives the sleep level. Either duration_col or end_time_col should be 
    None, and the other should be a string. awake_string, light_string, deep_string, and 
    rem_string are the entries in the sleep level column that represent those states of sleep. 
    gap_thresh is the threshold for the time between datapoints that indicates a new sleep block.
    filter_dict is a dictionary that specifies any filters to apply to the input df before extracting
    features, and time_zone is the timezone that the timestamps are in.
    """
   
    # Prepare the df
    df = df_filter(df, filter_dict)

    # Get end time and duration columns
    if end_time_col is None:
        df['end_time'] = df[timestamp_col]+df[duration_col]
    else:
        df['end_time'] = df[end_time_col]
    df['duration'] = df['end_time'] - df[timestamp_col]

    #Get extended index (this is used during FE in case of lone awake datapoints)
    counts_raw = get_fixed_series(df, 'D', "count", sleep_level_col, timestamp_col, "counts",time_type="unix",time_zone=time_zone)
    extended_index = pd.date_range(start=counts_raw.index.min(), end=counts_raw.index.max(), freq='D')
    
    #Remove awake datapoints.
    df=df[df[sleep_level_col]!=awake_string].copy()

    #Extract overall and individual sleep stage TSTs for day
    df_for_TST=find_durations(df.copy(), timestamp_col, 'end_time', 'D', time_zone=time_zone)
    TST_=get_fixed_series(df_for_TST.copy(),'D','sum','seconds_diff',timestamp_col,'Daily TST',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    df_for_TST_light=df_for_TST[df_for_TST[sleep_level_col]==light_string]
    TST_light=get_fixed_series(df_for_TST_light.copy(),'D','sum','seconds_diff',timestamp_col,'Total time in light sleep',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    df_for_TST_deep=df_for_TST[df_for_TST[sleep_level_col]==deep_string]
    TST_deep=get_fixed_series(df_for_TST_deep.copy(),'D','sum','seconds_diff',timestamp_col,'Total time in deep sleep',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    df_for_TST_rem=df_for_TST[df_for_TST[sleep_level_col]==rem_string]
    TST_rem=get_fixed_series(df_for_TST_rem.copy(),'D','sum','seconds_diff',timestamp_col,'Total time in rem sleep',extended_index=extended_index,time_type="unix",time_zone=time_zone)
   
    # create a column 'group' that assigns the same number to all datapoints in the same sleep block
    df[timestamp_col] = df[timestamp_col].astype(float) # Do this to prevent warning further down.
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    gaps = df[timestamp_col]-df['duration'].shift().fillna(0)-df[timestamp_col].shift().fillna(0) # time between timestamp and previous end time
    group_ids = (gaps > gap_thresh).cumsum()  # Start a new group when difference > thresh
    df["group"] = group_ids

    #create a col 'awakening' that is 1 if the next timestamp is bigger than the end time col
    df['awakening'] = (df["end_time"] != df[timestamp_col].shift(-1)).astype(int)
    df.loc[df.index[-1], 'awakening'] = 1 # we do this because we delete 1 from this row later as we have counted an extra awakening at the end of each bloack, so we want to do that for the last block too

    # aggregate to get start and end time of each block
    df = df.groupby("group", as_index=False).agg(
        {
            timestamp_col: "first",
            'end_time': "last",
            'duration': "sum",
            'awakening': "sum"
        }
    )
    
    #Extract total number of sleep episodes per day
    total_naps=get_fixed_series(df.copy(),'D','count','duration','end_time','Total sleep episodes',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    total_sleep_episodes_duration=get_fixed_series(df.copy(),'D','sum','duration','end_time','Total sleep episodes duration',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    #subract 1 from the awakening col
    df['awakening'] = df['awakening'] -1

    #Calculate the block duration, get max in each day (PSP), then add all non zero PSPs to list. 
    df['block_duration'] = df['end_time'] - df[timestamp_col]
    
    #df[timestamp_col] = (pd.to_datetime(df[timestamp_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.floor('D').astype('int64') // 10**9)
    df['midnight'] = (pd.to_datetime(df['end_time'], unit="s", utc=True).dt.tz_convert(time_zone).dt.floor('D').astype('int64') // 10**9)
    max_blocks = df.loc[df.groupby('midnight')['block_duration'].idxmax(), ['midnight', 'block_duration', 'duration', 'awakening', 'end_time']]
    max_blocks['time awake']= max_blocks['block_duration']-max_blocks['duration']
    max_blocks['wake up time']=max_blocks['end_time']-max_blocks['midnight']

    #extract all PSP related features
    PSP_length=get_fixed_series(max_blocks.copy(),'D','sum','block_duration','end_time','PSP length',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    PSP_TST=get_fixed_series(max_blocks.copy(),'D','sum','duration','end_time','PSP TST',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    PSP_time_awake=get_fixed_series(max_blocks.copy(),'D','sum','time awake','end_time','Time awake in PSP',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    PSP_number_awakenings=get_fixed_series(max_blocks.copy(),'D','sum','awakening','end_time','Number of awakenings during PSP',extended_index=extended_index,time_type="unix",time_zone=time_zone)
    PSP_wake_up_time=get_fixed_series(max_blocks.copy(),'D','sum','wake up time','end_time','PSP wake up time',extended_index=extended_index,time_type="unix",time_zone=time_zone)

    all_features = pd.concat(
    [
        TST_,
        TST_light,
        TST_deep,
        TST_rem,
        total_sleep_episodes_duration,
        total_naps,
        PSP_length,
        PSP_TST,
        PSP_time_awake,
        PSP_number_awakenings,
        PSP_wake_up_time
    ],
    axis=1,
    )
    
    all_features['Nap duration'] = all_features['Total sleep episodes duration']-all_features['PSP TST']
    all_features = all_features.drop(columns=['Total sleep episodes duration'])
    all_features['Sleep efficiency'] = all_features['PSP TST']/all_features['PSP length']

    return all_features


def general_steps_cleaning_and_FE(
    df,
    interval,
    meas_col,
    timestamp_col,
    STG,
    EAS_thresh,
    convert_to_unix=None,
    meas_agg="mean",
    duration_col=None,
    SPS=4,
    filter_min=50,
    end_time_col=None,
    STG_fix=False,
    cumulative=False,
    max_time_gap=None,
    device_col=None,
    filter_dict=None,
    round_to_midnight=False,
    distribute_steps=False,
    included_errors=["RT+CM", "STG+CM", "STG-CM", 'EAS'],
    time_zone="Europe/London",
    segments=False
    ):
    
    df = df.sort_values(by=timestamp_col)

    if round_to_midnight:
        # TODO think about whether below is correct
        df, df_hour_cleaned, hour_features = round_timestamp_to_midnight(
            df, timestamp_col, time_zone
        )
        if end_time_col is not None:
            df[end_time_col] = df[timestamp_col] + 86400

    df = df_filter(df, filter_dict)
    # convert to unix time if neccessary
    if convert_to_unix is not None:
        df = convert_to_unix_time(df, convert_to_unix)

    counts_raw = get_fixed_series(
        df, interval, "count", meas_col, timestamp_col, "total raw datapoints",time_type="unix", time_zone=time_zone
    )
    extended_index = pd.date_range(
        start=counts_raw.index.min(), end=counts_raw.index.max(), freq=interval
    )

    if cumulative != True:
        df = df[df[meas_col] > 0].copy()
        df = df.reset_index(drop=True)

    if cumulative and (device_col is not None):
        # Fix where transition happens if there is a single RT+CM or STG+CM
        df["dif_device"] = (df[[device_col]] != df[[device_col]].shift()).any(axis=1)
        df.iloc[0, df.columns.get_loc("dif_device")] = False
        df["time gap"] = df[timestamp_col].diff().fillna(0)
        mask = (
            df["dif_device"]
            & (df["time gap"] < STG)
            & df["dif_device"].shift(-1, fill_value=False)
            & (df["time gap"].shift(-1, fill_value=0) > STG)
            & df["dif_device"].shift(fill_value=False)
            & (df["time gap"].shift(fill_value=0) > STG)
        )
        prev_ts = df[timestamp_col].shift().fillna(0)
        df.loc[mask, timestamp_col] = prev_ts[mask] - STG
        df = df.sort_values(by=timestamp_col)
        df["dif_device"] = (df[[device_col]] != df[[device_col]].shift()).any(axis=1)
        df.iloc[0, df.columns.get_loc("dif_device")] = False
        df["time gap"] = df[timestamp_col].diff().fillna(0)
        mask = (
            df["dif_device"]
            & (df["time gap"] < STG)
            & ~df["dif_device"].shift(-1, fill_value=False)
            & (df["time gap"].shift(-1, fill_value=0) > STG)
            & ~df["dif_device"].shift(fill_value=False)
            & (df["time gap"].shift(fill_value=0) > STG)
        )
        prev_ts = df[timestamp_col].shift().fillna(0)
        df.loc[mask, timestamp_col] = prev_ts[mask] + STG
        # Produce a cleaned version that has a device col so that this can be merged later.
        df_clean_device, features = get_timestamp_errors_and_clean(
            df,
            interval,
            timestamp_col,
            device_col,
            STG,
            EAS_thresh,
            STG_fix=STG_fix,
            meas_agg="first",
            end_time_col=end_time_col,
            duration_col=duration_col,
            time_zone=time_zone
        )
    if end_time_col == None and duration_col == None:
        df["previous_time_stamp"] = df[timestamp_col].shift().fillna(0)
        df_clean_pts, features = get_timestamp_errors_and_clean(
            df,
            interval,
            timestamp_col,
            "previous_time_stamp",
            STG,
            STG_fix=STG_fix,
            meas_agg="min",
            time_zone=time_zone
        )

    cleaned_df, features = get_timestamp_errors_and_clean(
        df=df,
        interval=interval,
        time_stamp_col=timestamp_col,
        measurement_col=meas_col,
        EAS_thresh=EAS_thresh,
        STG=STG,
        meas_agg=meas_agg,
        duration_col=duration_col,
        end_time_col=end_time_col,
        STG_fix=STG_fix,
        time_zone=time_zone
    )
    if end_time_col == None and duration_col == None:
        cleaned_df["previous_time_stamp"] = df_clean_pts["previous_time_stamp"]
    if cumulative and (device_col is not None):
        cleaned_df[device_col] = df_clean_device[device_col]
    if round_to_midnight:
        cleaned_df["hour"] = df_hour_cleaned["hour"]
    if cumulative == True:
        cleaned_df["new steps"] = cleaned_df[meas_col].diff().fillna(0)
        meas_col = "new steps"
        cleaned_df["STG-CM"] = 0
        cleaned_df["RT+CM"] = (
            (cleaned_df["RT+CM"].shift() == 1) | (cleaned_df["RT+CM"] == 1)
        ).astype(int)
        cleaned_df["STG+CM"] = (
            (cleaned_df["STG+CM"].shift() == 1) | (cleaned_df["STG+CM"] == 1)
        ).astype(int)
        if device_col is not None:
            cleaned_df["same_device"] = (
                cleaned_df[[device_col]] == cleaned_df[[device_col]].shift()
            ).any(axis=1)
            cleaned_df.iloc[0, cleaned_df.columns.get_loc("same_device")] = True
            cleaned_df = cleaned_df[cleaned_df["same_device"] == True].copy()
        cleaned_df = cleaned_df[cleaned_df[meas_col] > 0].copy()
        if max_time_gap is not None:
            cleaned_df['time_diff'] = cleaned_df[timestamp_col] - cleaned_df['previous_time_stamp']
            cleaned_df=cleaned_df[cleaned_df['time_diff'] <= max_time_gap].copy()
    if duration_col != None:  # TODO need to consider if end time=timestamp
        cleaned_df["filtered steps"] = cleaned_df[meas_col].clip(
            upper=SPS * cleaned_df[duration_col]
        )
    if end_time_col != None:  # TODO need to consider if duration==0
        cleaned_df["filtered steps"] = cleaned_df[meas_col].clip(
            upper=SPS * (cleaned_df[end_time_col] - cleaned_df[timestamp_col])
        )
    if end_time_col == None and duration_col == None:
        cleaned_df["allowed steps"] = (
            cleaned_df[timestamp_col] - cleaned_df["previous_time_stamp"]
        ) * SPS
        cleaned_df["filtered steps"] = cleaned_df[["allowed steps", meas_col]].min(
            axis=1
        )
    cleaned_df["filtered steps"] = np.where(
        cleaned_df[meas_col] < filter_min,
        cleaned_df[meas_col],
        cleaned_df["filtered steps"].clip(lower=filter_min),
    )
    cleaned_df["Number filtered"] = (
        cleaned_df["filtered steps"] != cleaned_df[meas_col]
    ).astype(int)
    # Create a new column in df for total errors
    cleaned_df["total timestamps with any error"] = cleaned_df[
        included_errors + ["Number filtered"]
    ].max(axis=1)
    # Extract metadata features from df
    df_errors = cleaned_df.loc[
        :, included_errors + ["Number filtered", "total timestamps with any error"]
    ].copy()
    df_errors["total counts"] = 1
    features = df_errors.resample(interval).sum()
    features = features.reindex(extended_index).fillna(0)
    if round_to_midnight:
        features = pd.concat(
            [features, hour_features["hour of datapoint"], counts_raw], axis=1
        )
    else:
        features = pd.concat([features, counts_raw], axis=1)
    features.index.name = cleaned_df.index.name

    if distribute_steps:
        if end_time_col is None:
            end_time_col = "end_time_col"
            if duration_col is None:
                cleaned_df["end_time_col"] = cleaned_df[timestamp_col]
                cleaned_df[timestamp_col] = cleaned_df["previous_time_stamp"]
            else:
                cleaned_df["end_time_col"] = (
                    cleaned_df[timestamp_col] + cleaned_df[duration_col]
                )
        #df_for_steps = cleaned_df.copy()
        #df_for_filtered = cleaned_df.copy()
        cleaned_df_steps = find_durations(
            cleaned_df.copy(), timestamp_col, end_time_col, interval, meas_col, time_zone
        )
        cleaned_df_filtered = find_durations(
            cleaned_df.copy(), timestamp_col, end_time_col, interval, "filtered steps", time_zone
        )
        total_filtered = get_fixed_series(
            cleaned_df_filtered,
            interval,
            "sum",
            "filtered steps",
            timestamp_col,
            "Total steps (with filtering)",
            time_type="unix",
            time_zone=time_zone
        )
        total_unfiltered = get_fixed_series(
            cleaned_df_steps,
            interval,
            "sum",
            meas_col,
            timestamp_col,
            "Total steps (without filtering)",
            time_type="unix",
            time_zone=time_zone
        )
        if segments:
            cleaned_df_steps = find_durations(
            cleaned_df.copy(), timestamp_col, end_time_col, 'h', meas_col, time_zone
        )
            cleaned_df_filtered = find_durations(
            cleaned_df.copy(), timestamp_col, end_time_col, 'h', "filtered steps", time_zone
        )
            seg1,seg2,seg3,seg4=get_daily_segment_features(cleaned_df_steps,
                                    timestamp_col,
                                    meas_col, 
                                    ['total steps (unfiltered) 12am-6am', 'total steps (unfiltered) 6am-12pm', 'total steps (unfiltered) 12pm-6pm', 'total steps (unfiltered) 6pm-12am'], 
                                    time_zone=time_zone
                                    )
            seg1_filtered,seg2_filtered,seg3_filtered,seg4_filtered=get_daily_segment_features(cleaned_df_filtered,
                                    timestamp_col,
                                    "filtered steps", 
                                    ['total steps (filtered) 12am-6am', 'total steps (filtered) 6am-12pm', 'total steps (filtered) 12pm-6pm', 'total steps (filtered) 6pm-12am'], 
                                    time_zone=time_zone
                                    )
    else:
        total_filtered = get_fixed_series(
            cleaned_df,
            interval,
            "sum",
            "filtered steps",
            timestamp_col,
            "Total steps (with filtering)",
            time_type="unix",
            time_zone=time_zone
        )
        total_unfiltered = get_fixed_series(
            cleaned_df,
            interval,
            "sum",
            meas_col,
            timestamp_col,
            "Total steps (without filtering)",
            time_type="unix",
            time_zone=time_zone
        )
        if segments:
            seg1,seg2,seg3,seg4 = get_daily_segment_features(cleaned_df.copy(),
                                    timestamp_col,
                                    meas_col, 
                                    ['total steps (unfiltered) 12am-6am', 'total steps (unfiltered) 6am-12pm', 'total steps (unfiltered) 12pm-6pm', 'total steps (unfiltered) 6pm-12am'], 
                                    time_zone=time_zone
                                    )
            seg1_filtered,seg2_filtered,seg3_filtered,seg4_filtered=get_daily_segment_features(cleaned_df.copy(),
                                    timestamp_col,
                                    "filtered steps", 
                                    ['total steps (filtered) 12am-6am', 'total steps (filtered) 6am-12pm', 'total steps (filtered) 12pm-6pm', 'total steps (filtered) 6pm-12am'], 
                                    time_zone=time_zone
                                    )
    if segments:
        total_steps = pd.concat([total_filtered, total_unfiltered,seg1,seg2,seg3,seg4,seg1_filtered,seg2_filtered,seg3_filtered,seg4_filtered], axis=1)
    else:
        total_steps = pd.concat([total_filtered, total_unfiltered], axis=1)
    total_steps = total_steps.reindex(extended_index).fillna(0)
    total_steps.index.name = cleaned_df.index.name

    return features, cleaned_df, total_steps


def get_daily_segment_features(df,
                                timestamp_col,
                                meas_col, 
                                new_names, 
                                agg='sum',
                                extended_index=None,
                                time_zone="Europe/London",
                                time_type="unix"
                                ):

    # convert to unix if neccessary, otherwise set dt as timestamp_col
    if time_type=='unix':
        df["dt"] =pd.to_datetime(df[timestamp_col], unit="s", utc=True).dt.tz_convert(time_zone).dt.tz_localize(None)
        timestamp_col = "dt"

    df_0_to_6 = df[(df[timestamp_col].dt.hour >= 0) & (df[timestamp_col].dt.hour < 6)]
    df_6_to_12 = df[(df[timestamp_col].dt.hour >= 6) & (df[timestamp_col].dt.hour < 12)]
    df_12_to_18 = df[(df[timestamp_col].dt.hour >= 12) & (df[timestamp_col].dt.hour < 18)]
    df_18_to_24 = df[(df[timestamp_col].dt.hour >= 18) & (df[timestamp_col].dt.hour < 24)]

    features_seg1 = get_fixed_series(
    df_0_to_6.copy(),
    'D',
    agg,
    meas_col,
    timestamp_col,
    new_names[0],
    extended_index=extended_index,
    time_type="dt",
    )

    features_seg2 = get_fixed_series(
    df_6_to_12.copy(),
    'D',
    agg,
    meas_col,
    timestamp_col,
    new_names[1],
    extended_index=extended_index,
    time_type="dt",
    )

    features_seg3 = get_fixed_series(
    df_12_to_18.copy(),
    'D',
    agg,
    meas_col,
    timestamp_col,
    new_names[2],
    extended_index=extended_index,
    time_type="dt",
    )

    features_seg4 = get_fixed_series(
    df_18_to_24.copy(),
    'D',
    agg,
    meas_col,
    timestamp_col,
    new_names[3],
    extended_index=extended_index,
    time_type="dt",
    )

    return features_seg1, features_seg2, features_seg3, features_seg4

