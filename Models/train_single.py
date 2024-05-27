import os
import sys
import torch
import numpy as np
import wandb
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import WandbLogger

from src.architectures.Unet5 import Unet5
from src.architectures.IceNetLike import IceNetLike
from src.architectures.UnetS import UnetS
from src.architectures.Linear import Linear
from src.architectures.LinearB import LinearB
from src.architectures.Persistence import Persistence
from src.architectures.LinearMat import LinearMat

import config_single as config_single
from src.utils import get_data_df
from src.dataset import IceDataModule
from src.model import IceModel
from src.callbacks import PrintCallback, EarlyStopping, DeviceStatsMonitor, ModelSummary, ModelCheckpoint

from torch.utils.viz._cycles import warn_tensor_cycles

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":

    #warn_tensor_cycles()
    #torch.cuda.memory._record_memory_history(max_entries=100000000)

    parser = ArgumentParser()
    parser.add_argument("--input_vars", type=str, nargs="+", default=config_single.input_vars)
    parser.add_argument(
        "--target_vars", type=str, nargs="+", default=config_single.target_vars
    )
    parser.add_argument("--grid_coords", type=tuple, default=config_single.grid_coords)
    parser.add_argument("--grid_shape", type=int, default=config_single.grid_shape)
    parser.add_argument(
        "--architecture_name", type=str, default=config_single.architecture_name
    )
    parser.add_argument("--learning_rate", type=float, default=config_single.learning_rate)
    parser.add_argument("--batch_size", type=int, default=config_single.batch_size)
    parser.add_argument("--max_epochs", type=int, default=config_single.max_epochs)
    parser.add_argument("--num_workers", type=int, default=config_single.num_workers)
    parser.add_argument("--normalize", type=bool, default=config_single.normalize)
    parser.add_argument("--standardize", type=bool, default=config_single.standardize)
    parser.add_argument("--loss_function", type=str, default=config_single.loss_function)
    parser.add_argument("--shuffle_train", type=bool, default=config_single.shuffle_train)
    parser.add_argument("--shuffle_val", type=bool, default=config_single.shuffle_val)
    parser.add_argument("--optimizer_str", type=str, default=config_single.optimizer_str)
    parser.add_argument("--post_processing", type=bool, default=config_single.post_processing)
    parser.add_argument("--to_mask", type=str, default=config_single.to_mask)
    args = parser.parse_args()

    input_vars = args.input_vars
    target_vars = args.target_vars
    grid_coords = args.grid_coords
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

    norm_str = "norm" if normalize else "nonorm"
    stand_str = "stand" if standardize else "nostand"

    data_df = get_data_df(data_dir, train_window, val_window, test_window)

    data_module = IceDataModule(
        data_df=data_df,
        batch_size=batch_size,
        num_workers=num_workers,
        input_vars=input_vars,
        target_vars=target_vars,
        grid_coords=grid_coords,
        grid_shape=grid_shape,
        normalize=normalize,
        standardize=standardize,
        shuffle_train=shuffle_train,
        shuffle_val=shuffle_val,
    )

    in_neurons = len(input_vars) * grid_shape**2
    out_neurons = len(target_vars) * grid_shape**2

    model = IceModel(
        architecture=architecture,
        learning_rate=learning_rate,
        in_channels=len(input_vars),
        out_channels=len(target_vars),
        in_neurons=in_neurons,
        out_neurons=out_neurons,
        input_vars=input_vars,
        target_vars=target_vars,
        normalize=normalize,
        standardize=standardize,
        loss_function=loss_function,
        grid_shape=grid_shape,
        grid_stride=0,
        max_epochs=max_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=shuffle_train,
        shuffle_val=shuffle_val,
        optimizer_str=optimizer_str,
        post_processing=post_processing,
        to_mask=to_mask,
    )
    
    model_scope = "S_" + str(grid_shape) + "_" + str(grid_coords) + "_" + str(target_vars)
    model_specifications = f"{architecture.__name__}_{learning_rate}_{loss_function}_{norm_str}_{stand_str}_{input_vars}_{shuffle_train}"

    #tb_logger = TensorBoardLogger(model_scope, name=model_specifications)
    logger = WandbLogger(project="iceberg")
    #logger.watch(model, log="all")
    #logger = [tb_logger, wandb_logger]
    #profiler = PyTorchProfiler(
        #on_trace_ready=torch.profiler.tensorboard_trace_handler(logger.log_dir),
        #activites=[
        #    torch.profiler.ProfilerActivity.CPU,
        #    torch.profiler.ProfilerActivity.CUDA,
        #],
        #profile_memory=True,
        #schedule=torch.profiler.schedule(wait=1, warmup=1, active=20, repeat=1),
    #)

    trainer = pl.Trainer(
        default_root_dir=f"{model_scope}/{model_specifications}",
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=max_epochs,
        #callbacks=[
        #    EarlyStopping(monitor="val_loss"),
        #],
        logger=logger,
        #profiler=profiler,
        val_check_interval=0.25,
        #limit_val_batches=1.0,
        #num_sanity_val_steps=2,
    )

    #if os.path.exists(f"{model_scope}/{model_specifications}"):
    #    trainer.fit(model, ckpt_path=f"{model_scope}/{model_specifications}")
    trainer.fit(model, data_module)

    #torch.cuda.memory._record_memory_history(enabled=None)
    
    #predictions = trainer.predict(model, data_module)
    #trainer.tune(model, train_loader, val_loader)
    #trainer.validate(model, data_module)
    #trainer.test(model, data_module)
