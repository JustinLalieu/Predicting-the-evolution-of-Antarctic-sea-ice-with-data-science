import time
import torch
import numpy as np
import pandas as pd
import xarray as xr
#import netCDF4 as nc
from torch.utils.data import Dataset
from datetime import datetime, timedelta



class GridLoader(Dataset):
    def __init__(
            self,
            data_df,
            split,
            input_vars,
            target_vars,
            grid_coords,
            grid_shape,
            normalize,
            standardize
    ): 
        self.data_df = data_df
        self.split = split
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.grid_coords = grid_coords
        self.grid_shape = grid_shape
        self.normalize = normalize
        self.standardize = standardize
        
        self.split_df = self.data_df[self.data_df["split"] == self.split]
        self.file_pathes = self.data_df["path"].unique()
        self.datasets = [xr.open_dataset(file_path) for file_path in self.file_pathes]

        #self.mask = self.get_mask()

        self.min_max_dict = {'SIC_NEMO': {'min': 0.0, 'max': 100.0}, 'SNTHIC_NEMO': {'min': 0.0, 'max': 503.26926}, 'MLD_NEMO': {'min': 0.0, 'max': 451996.06}}
        self.mean_std_dict = {'SIC_NEMO': {'mean': 12.985211499810287, 'std': 36.04501543432098}, 'SNTHIC_NEMO': {'mean': 3.803279445586731, 'std': 11.894388267703121}, 'MLD_NEMO': {'mean': 11090.49358046236, 'std': 9023.29259886038}}
        self.norm_mean_std_dict = {'SIC_NEMO': {'mean': 0.10409109749254702, 'std': 0.2899808938060342}, 'SNTHIC_NEMO': {'mean': 0.006015777002221595, 'std': 0.018835558220642458}, 'MLD_NEMO': {'mean': 0.01931211203517341, 'std': 0.0153130332346179}}

    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.split_df.iloc[idx]
        current_timestamp = row.name
        input_data = []
        for input_var in self.input_vars:
            if "CLIM" in input_var:
                input_data.append(self.get_clim_var(current_timestamp, input_var))
            else:
                input_data.append(self.get_var(current_timestamp, input_var))
        input_data_fin = torch.cat(input_data, dim=0)
        target_data = []
        for target_var in self.target_vars:
            target_data.append(self.get_var(current_timestamp, target_var))
        target_data_fin = torch.cat(target_data, dim=0)
        return input_data_fin, target_data_fin
    
    # SIC_NEMO_CLIM_5x365
    def get_clim_var(self, timestamp, var):
        min_timestamp = self.data_df.index.min()
        var_name = var.split("_")[0] + "_" + var.split("_")[1]
        infos = var.split("_")[-1]
        n_step = int(infos.split("x")[0])
        step_size = int(infos.split("x")[1])
        final_var = torch.zeros(self.grid_shape, self.grid_shape)
        for step in range(1, n_step + 1):
            target_timestamp = timestamp - step_size * step
            while target_timestamp < min_timestamp:
                target_timestamp += 365
            row = self.data_df.loc[target_timestamp]
            path = row["path"]
            nth_day = row["nth_day"]
            nth_path = np.where(self.file_pathes == path)[0][0]
            dataset = self.datasets[nth_path]
            step_var = dataset.variables[var_name][nth_day][
                self.grid_coords[0]:self.grid_coords[0]+self.grid_shape,
                self.grid_coords[1]:self.grid_coords[1]+self.grid_shape]
            final_var += step_var
        final_var /= n_step
        final_var_ma = final_var.where(final_var < 100000000000000, 0)
        mask = final_var_ma == 0
        if self.normalize and not self.standardize:
            normalized_data = (final_var_ma - self.min_max_dict[var_name]['min']) / (self.min_max_dict[var_name]['max'] - self.min_max_dict[var_name]['min'])
            final_var_tensor = torch.tensor(normalized_data.data)
        elif self.standardize and not self.normalize:
            standardized_data = (final_var_ma - self.mean_std_dict[var_name]['mean']) / self.mean_std_dict[var_name]['std']
            final_var_tensor = torch.tensor(standardized_data.data)
        elif self.normalize and self.standardize:
            normalized_data = (final_var_ma - self.min_max_dict[var_name]['min']) / (self.min_max_dict[var_name]['max'] - self.min_max_dict[var_name]['min'])
            standardized_data = (normalized_data - self.norm_mean_std_dict[var_name]['mean']) / self.norm_mean_std_dict[var_name]['std']
            final_var_tensor = torch.tensor(standardized_data.data)
        else:
            final_var_tensor = torch.tensor(final_var_ma.data)
        tens = final_var_tensor.unsqueeze(0)
        return tens
            
    
    def get_var(self, timestamp, var):
        if "+" in var:
            var_name, offset_str = var.split("+")
            offset = int(offset_str)
            target_timestamp = timestamp + offset
        elif "-" in var:
            var_name, offset_str = var.split("-")
            offset = int(offset_str)
            target_timestamp = timestamp - offset
        row = self.data_df.loc[target_timestamp]
        path = row["path"]
        nth_day = row["nth_day"]
        nth_path = np.where(self.file_pathes == path)[0][0]
        dataset = self.datasets[nth_path]
        var_data = dataset.variables[var_name][nth_day][
            self.grid_coords[0]:self.grid_coords[0]+self.grid_shape,
            self.grid_coords[1]:self.grid_coords[1]+self.grid_shape]
        var_data_ma = var_data.where(var_data < 100000000000000, 0)
        if self.normalize and not self.standardize:
            normalized_data = (var_data_ma - self.min_max_dict[var_name]['min']) / (self.min_max_dict[var_name]['max'] - self.min_max_dict[var_name]['min'])
            var_data_tensor = torch.tensor(normalized_data.data)
        elif self.standardize and not self.normalize:
            standardized_data = (var_data_ma - self.mean_std_dict[var_name]['mean']) / self.mean_std_dict[var_name]['std']
            var_data_tensor = torch.tensor(standardized_data.data)
        elif self.normalize and self.standardize:
            normalized_data = (var_data_ma - self.min_max_dict[var_name]['min']) / (self.min_max_dict[var_name]['max'] - self.min_max_dict[var_name]['min'])
            standardized_data = (normalized_data - self.norm_mean_std_dict[var_name]['mean']) / self.norm_mean_std_dict[var_name]['std']
            var_data_tensor = torch.tensor(standardized_data.data)
        else:
            var_data_tensor = torch.tensor(var_data_ma.data)
        tens = var_data_tensor.unsqueeze(0)

        return tens

        



        