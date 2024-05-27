import os
import torch
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_data_df(
        data_dir,
        train_window,
        val_window,
        test_window,
        base_date=datetime(1900, 1, 1)
):
    timestamps_list = []
    dates_list = []
    nth_days_list = []
    pathes_list = []
    split_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".nc"):
                file_path = os.path.join(root, file)
                ncdf = nc.Dataset(file_path, "r")
                year = int(file.split(".")[0])
                for t, time in enumerate(ncdf.variables["time"][:]):
                    date = datetime(year, 1, 1) + timedelta(days=t)
                    timestamp = (date - base_date).days
                    if train_window[0] <= date <= train_window[1]:
                        split = "train"
                    elif val_window[0] <= date <= val_window[1]:
                        split = "val"
                    elif test_window[0] <= date <= test_window[1]:
                        split = "test"
                    else:
                        split = "other"
                    timestamps_list.append(timestamp)
                    dates_list.append(date)
                    nth_days_list.append(t)
                    pathes_list.append(file_path)
                    split_list.append(split)
                ncdf.close()
    df = pd.DataFrame({
        "timestamp": timestamps_list,
        "date": dates_list,
        "nth_day": nth_days_list,
        "path": pathes_list,
        "split": split_list
    })
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df

"""def get_date_path_df(per_year_path,
                          train_window,
                          val_window,
                          test_window
                          ):
    dp_df = pd.DataFrame(columns=["date", "split", "path"])
    for root, dirs, files in os.walk(per_year_path):
        for file in files:
            if file.endswith(".nc"):
                year_ds = nc.Dataset(os.path.join(root, file))
                for i, day in enumerate(year_ds.variables["time"]):
                    year = int(file.split(".")[0])
                    date = datetime(year, 1, 1) + timedelta(days=i)
                    if train_window[0] <= date <= train_window[1]:
                        split = "train"
                    elif val_window[0] <= date <= val_window[1]:
                        split = "val"
                    elif test_window[0] <= date <= test_window[1]:
                        split = "test"
                    else:
                        split = "other"
                    path = os.path.join(root, file)
                    temp_df = pd.DataFrame({"date": [date], "split": [split], "path": [path]})
                    dp_df = (dp_df.copy() if temp_df.empty else temp_df.copy() if dp_df.empty else pd.concat([dp_df, temp_df]))
                year_ds.close()
    dp_df.sort_values("date", inplace=True)
    dp_df.set_index("date", inplace=True)
    dp_df.to_csv(os.path.join(per_year_path, "date_path_df.csv"))
    return dp_df"""