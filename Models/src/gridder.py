from datetime import datetime
import netCDF4 as nc
import numpy as np
import wandb
import pytorch_lightning as pl
#from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from src.model import IceModel
from src.dataset import IceDataModule
from src.callbacks import EarlyStopping

class Gridder():
    def __init__(
            self,
            data_df,
            architecture,
            input_vars,
            target_vars,
            grid_shape,
            grid_stride,
            loss_function,
            max_epochs,
            batch_size,
            num_workers,
            normalize,
            standardize,
            train_window,
            val_window,
            test_window,
            shuffle_train,
            shuffle_val,
            optimizer_str,
            post_processing,
            to_mask,
    ):
        self.data_df = data_df
        self.architecture = architecture
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.grid_shape = grid_shape
        self.grid_stride = grid_stride
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.standardize = standardize
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.grid_coords = self.get_grid_coords()
        self.trainers = self.get_slots()
        self.models = self.get_slots()
        self.datamodules = self.get_slots()
        self.norm_str = "norm" if self.normalize else "nonorm"
        self.stand_str = "stand" if self.standardize else "nostand"
        self.input_vars_str = self.make_input_vars_str()
        self.optimizer_str = optimizer_str
        self.post_processing = post_processing
        self.pp_str = "PP" if self.post_processing else ""
        self.to_mask = to_mask

    def make_input_vars_str(self):
        vars_dict = {}
        for var in self.input_vars:
            if var.split("-")[0] not in vars_dict:
                vars_dict[var.split("-")[0]] = []
            vars_dict[var.split("-")[0]].append(var.split("-")[1])
        input_vars_str = ""
        for key in vars_dict:
            input_vars_str += key + "-"
            for var in vars_dict[key]:
                input_vars_str += var + "-"
            input_vars_str = input_vars_str[:-1] + "+"
        return input_vars_str[:-1]
        
    def get_grid_coords(self):
        grids = []
        for i in range(0, 432-(self.grid_shape-self.grid_stride), self.grid_stride):
            for j in range(0, 432-(self.grid_shape-self.grid_stride), self.grid_stride):
                grids.append((i, j))
        return grids
    
    def get_slots(self):
        slots = []
        for _ in range(len(self.grid_coords)):
            slots.append(None)
        return slots
    
    def train_single_model(
            self,
            grid_id,
            learning_rate=0.00001,
    ):
        self.learning_rate = learning_rate
        self.datamodules[grid_id] = IceDataModule(
            data_df=self.data_df,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            grid_coords=self.grid_coords[grid_id],
            grid_shape=self.grid_shape,
            normalize=self.normalize,
            standardize=self.standardize,
            shuffle_train=self.shuffle_train,
            shuffle_val=self.shuffle_val,
        )
        self.models[grid_id] = IceModel(
            architecture=self.architecture,
            learning_rate=learning_rate,
            in_channels=len(self.input_vars),
            out_channels=len(self.target_vars),
            in_neurons=len(self.input_vars) * self.grid_shape**2,
            out_neurons=len(self.target_vars) * self.grid_shape**2,
            input_vars=self.input_vars,
            target_vars=self.target_vars,
            normalize=self.normalize,
            standardize=self.standardize,
            loss_function=self.loss_function,
            grid_shape=self.grid_shape,
            grid_stride=self.grid_stride,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle_train=self.shuffle_train,
            shuffle_val=self.shuffle_val,
            optimizer_str=self.optimizer_str,
            post_processing=self.post_processing,
            to_mask=self.to_mask,
        )

        model_scope = "G_" + str(self.grid_shape) + "_" + str(self.grid_stride) + "_" + str(self.target_vars)
        model_specifications = f"{self.grid_coords[grid_id]}_{self.architecture.__name__}_{learning_rate}_{self.loss_function}_{self.norm_str}_{self.stand_str}_{self.input_vars_str}_{self.shuffle_train}"

        logger = TensorBoardLogger(model_scope, name=model_specifications)

        #logger = WandbLogger(project="iceberg", log_model="all")
        #profiler = PyTorchProfiler()

        #logger = [tb_logger, wandb_logger]

        self.trainers[grid_id] = pl.Trainer(
            default_root_dir=f"{model_scope}/{model_specifications}",
            accelerator="auto",
            #devices="auto",
            #strategy="auto",
            max_epochs=self.max_epochs,
            logger=logger,
            #profiler=profiler,
            #callbacks=[
            #    EarlyStopping(monitor="val_loss"),
            #],
            enable_checkpointing=False
        )
        self.trainers[grid_id].fit(
            self.models[grid_id],
            self.datamodules[grid_id],
        )

    def train_all_models(
            self,
            learning_rate=0.00001,
    ):
        for grid_id in range(len(self.grid_coords)):
            self.train_single_model(grid_id, learning_rate)

    def predict_single_model(self, grid_id):
        self.datamodules[grid_id].setup("test")
        preds = self.trainers[grid_id].predict(self.models[grid_id], self.datamodules[grid_id].test_dataloader())
        return preds
    
    def predict_all_models(self):
        all_preds = []
        for grid_id in range(len(self.grid_coords)):
            all_preds.append(self.predict_single_model(grid_id))
        return all_preds
    
    def predict_single_model_on_validation(self, grid_id):
        self.datamodules[grid_id].setup("val")
        preds = self.trainers[grid_id].predict(self.models[grid_id], self.datamodules[grid_id].val_dataloader())
        return preds
    
    def predict_all_models_on_validation(self):
        all_preds = []
        for grid_id in range(len(self.grid_coords)):
            all_preds.append(self.predict_single_model_on_validation(grid_id))
        return all_preds
    
    def predict_single_model_on_training(self, grid_id):
        self.datamodules[grid_id].setup("train")
        preds = self.trainers[grid_id].predict(self.models[grid_id], self.datamodules[grid_id].train_dataloader_inference())
        return preds
    
    def predict_all_models_on_training(self):
        all_preds = []
        for grid_id in range(len(self.grid_coords)):
            all_preds.append(self.predict_single_model_on_training(grid_id))
        return all_preds
    
    def recompose(self, preds, split, save_as_nc=True):
        recomposed = []
        recomposed_count = []
        n_preds = 0
        for grid_preds in preds:
            for batch in grid_preds:
                n_preds += len(batch)
        n_preds = n_preds // len(preds)
        for i in range(n_preds):
            recomposed.append(np.zeros((432, 432)))
            recomposed_count.append(np.zeros((432, 432)))
        for gid, grid_preds in enumerate(preds):
            pid = 0
            for b, batch in enumerate(grid_preds):
                for p, pred in enumerate(batch):
                    x, y = self.grid_coords[gid]
                    recomposed[pid][x:x+self.grid_shape, y:y+self.grid_shape] += np.array(pred[0])
                    recomposed_count[pid][x:x+self.grid_shape, y:y+self.grid_shape] += 1
                    pid += 1
        for i in range(len(recomposed)):
            recomposed[i] /= recomposed_count[i]
            recomposed[i] = np.clip(recomposed[i], 0, 100)
        now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if save_as_nc:
            for i, var in enumerate(self.target_vars):
                with nc.Dataset(f"NCs/{split}_{self.architecture.__name__}_{self.to_mask}_{self.pp_str}_{self.grid_shape}_{self.grid_stride}_{self.loss_function}_{self.learning_rate}_{self.max_epochs}_{self.norm_str}_{self.stand_str}_{self.input_vars_str}_{now_str}.nc", "w") as ds:
                    ds.createDimension("x", 432)
                    ds.createDimension("y", 432)
                    ds.createDimension("time", None)
                    ds.createVariable(var, "f4", ("time", "x", "y"), compression="zlib")
                    ds[var][:] = recomposed


    