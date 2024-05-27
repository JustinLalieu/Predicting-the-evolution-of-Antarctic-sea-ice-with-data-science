import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

from src.architectures.Unet5 import Unet5
from src.architectures.IceNetLike import IceNetLike
from src.architectures.UnetS import UnetS
from src.architectures.Linear import Linear
from src.architectures.LinearB import LinearB
from src.architectures.Persistence import Persistence
from src.architectures.LinearMat import LinearMat

import config_gridder as config_single
from src.utils import get_data_df
from src.gridder import Gridder
from src.callbacks import PrintCallback, EarlyStopping, DeviceStatsMonitor, ModelSummary

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input_vars", type=str, nargs="+", default=config_single.input_vars)
    parser.add_argument(
        "--target_vars", type=str, nargs="+", default=config_single.target_vars
    )
    parser.add_argument("--grid_stride", type=int, default=config_single.grid_stride)
    parser.add_argument("--grid_shape", type=int, default=config_single.grid_shape)
    parser.add_argument(
        "--architecture_name", type=str, default=config_single.architecture_name
    )
    parser.add_argument("--learning_rate", type=float, default=config_single.learning_rate)
    parser.add_argument("--batch_size", type=int, default=config_single.batch_size)
    parser.add_argument("--max_epochs", type=int, default=config_single.max_epochs)
    parser.add_argument("--num_workers", type=int, default=config_single.num_workers)
    parser.add_argument("--loss_function", type=str, default=config_single.loss_function)
    parser.add_argument("--normalize", type=bool, default=config_single.normalize)
    parser.add_argument("--standardize", type=bool, default=config_single.standardize)
    parser.add_argument("--optimizer_str", type=str, default=config_single.optimizer_str)
    parser.add_argument("--post_processing", type=bool, default=config_single.post_processing)
    parser.add_argument("--shuffle_train", type=bool, default=config_single.shuffle_train)
    parser.add_argument("--shuffle_val", type=bool, default=config_single.shuffle_val)
    parser.add_argument("--to_mask", type=bool, default=config_single.to_mask)
    args = parser.parse_args()

    input_vars = args.input_vars
    target_vars = args.target_vars
    grid_stride = args.grid_stride
    grid_shape = args.grid_shape
    loss_function = args.loss_function
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    num_workers = args.num_workers
    normalize = args.normalize
    standardize = args.standardize
    shuffle_train = args.shuffle_train
    shuffle_val = args.shuffle_val
    optimizer_str = args.optimizer_str
    post_processing = args.post_processing
    to_mask = args.to_mask

    architecture_name = args.architecture_name
    if architecture_name == "Unet5":
        architecture = Unet5
    elif architecture_name == "UnetS":
        architecture = UnetS
    elif architecture_name == "IceNetLike":
        architecture = IceNetLike
    elif architecture_name == "Linear":
        architecture = Linear
    elif architecture_name == "LinearB":
        architecture = LinearB
    elif architecture_name == "Persistence":
        architecture = Persistence
    elif architecture_name == "LinearMat":
        architecture = LinearMat
    

    data_dir = config_single.data_dir
    train_window = config_single.train_window
    val_window = config_single.val_window
    test_window = config_single.test_window

    data_df = get_data_df(data_dir, train_window, val_window, test_window)

    gridder = Gridder(
        data_df=data_df,
        architecture=architecture,
        input_vars=input_vars,
        target_vars=target_vars,
        grid_shape=grid_shape,
        grid_stride=grid_stride,
        loss_function=loss_function,
        max_epochs=max_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        standardize=standardize,
        train_window=train_window,
        val_window=val_window,
        test_window=test_window,
        shuffle_train=shuffle_train,
        shuffle_val=shuffle_val,
        optimizer_str=optimizer_str,
        post_processing=post_processing,
        to_mask=to_mask,
    )

    gridder.train_all_models(learning_rate=learning_rate)
    
    train_predictions = gridder.predict_all_models_on_training()
    val_predictions = gridder.predict_all_models_on_validation()
    test_predictions = gridder.predict_all_models()

    gridder.recompose(train_predictions, "train", save_as_nc=True)
    gridder.recompose(val_predictions, "val", save_as_nc=True)
    gridder.recompose(test_predictions, "test", save_as_nc=True)
