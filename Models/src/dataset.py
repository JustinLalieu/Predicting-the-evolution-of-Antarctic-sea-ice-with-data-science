import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.loader import GridLoader


class IceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_df,
        batch_size,
        num_workers,
        input_vars,
        target_vars,
        grid_coords,
        grid_shape,
        normalize,
        standardize,
        shuffle_train,
        shuffle_val,
    ):
        super(IceDataModule, self).__init__()
        self.data_df = data_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.grid_coords = grid_coords
        self.grid_shape = grid_shape
        self.normalize = normalize
        self.standardize = standardize
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or None:
            self.train_dataset = GridLoader(
                self.data_df,
                "train",
                self.input_vars,
                self.target_vars,
                self.grid_coords,
                self.grid_shape,
                self.normalize,
                self.standardize,
            )
            self.val_dataset = GridLoader(
                self.data_df,
                "val",
                self.input_vars,
                self.target_vars,
                self.grid_coords,
                self.grid_shape,
                self.normalize,
                self.standardize,
            )
        if stage == "test":
            self.test_dataset = GridLoader(
                self.data_df,
                "test",
                self.input_vars,
                self.target_vars,
                self.grid_coords,
                self.grid_shape,
                self.normalize,
                self.standardize,
            )
        if stage == "predict":
            self.test_dataset = GridLoader(
                self.data_df,
                "test",
                self.input_vars,
                self.target_vars,
                self.grid_coords,
                self.grid_shape,
                self.normalize,
                self.standardize,
            )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
    
    def train_dataloader_inference(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
