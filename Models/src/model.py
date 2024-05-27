import sys
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
import torchinfo
from torchinfo import summary
import matplotlib.pyplot as plt
import gc
import pytorch_lightning as pl
import wandb

from src.metrics import MSE, MAE, RMSE, NRMSE, IIEE, NIIEE, BACC, NSE, ANOM_CORR_COEF, Hybrid

class IceModel(pl.LightningModule):
    def __init__(
        self,
        architecture,
        learning_rate,
        in_channels,
        out_channels,
        in_neurons,
        out_neurons,
        input_vars,
        target_vars,
        normalize,
        standardize,
        loss_function,
        grid_shape,
        grid_stride,
        max_epochs,
        batch_size,
        num_workers,
        shuffle_train,
        shuffle_val,
        optimizer_str,
        post_processing,
        to_mask,
    ):
        super(IceModel, self).__init__()
        self.architecture = architecture(
            in_channels,
            hidden_dim=64,
            kernel_size=(3, 3),
            num_layers=5,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.architecture_name = self.architecture.architecture_name
        self.learning_rate = learning_rate
        self.optimizer_str = optimizer_str

        self.target_var = "SIC_NEMO"

        self.normalize = normalize
        self.standardize = standardize
        self.post_processing = post_processing
        self.min_max_dict = {'SIC_NEMO': {'min': 0.0, 'max': 100.0}, 'SNTHIC_NEMO': {'min': 0.0, 'max': 503.26926}, 'MLD_NEMO': {'min': 0.0, 'max': 451996.06}}
        self.mean_std_dict = {'SIC_NEMO': {'mean': 12.985211499810287, 'std': 36.04501543432098}, 'SNTHIC_NEMO': {'mean': 3.803279445586731, 'std': 11.894388267703121}, 'MLD_NEMO': {'mean': 11090.49358046236, 'std': 9023.29259886038}}
        self.norm_mean_std_dict = {'SIC_NEMO': {'mean': 0.10409109749254702, 'std': 0.2899808938060342}, 'SNTHIC_NEMO': {'mean': 0.006015777002221595, 'std': 0.018835558220642458}, 'MLD_NEMO': {'mean': 0.01931211203517341, 'std': 0.0153130332346179}}

        lfs_split = loss_function.split("_")
        self.loss_function_str = lfs_split[0]
        if self.loss_function_str == "Hybrid":
            alpha = float(lfs_split[1])
            beta = float(lfs_split[2])


        if self.loss_function_str == "MSE":
            self.loss_function = MSE()
        elif self.loss_function_str == "Hybrid":
            self.loss_function = Hybrid(alpha, beta)
        elif self.loss_function_str == "MAE":
            self.loss_function = MAE()
        elif self.loss_function_str == "RMSE":
            self.loss_function = RMSE()
        elif self.loss_function_str == "NRMSE":
            self.loss_function = NRMSE()
        elif self.loss_function_str == "IIEE":
            self.loss_function = IIEE()
        elif self.loss_function_str == "NIIEE":
            self.loss_function = NIIEE()
        elif self.loss_function_str == "BACC":
            self.loss_function = BACC()
        elif self.loss_function_str == "NSE":
            self.loss_function = NSE()
        else:
            raise ValueError(f"Loss function {loss_function} not recognized")
        
        self.metrics = {
            "mse": MSE(),
            "mae": MAE(),
            "rmse": RMSE(),
            "nrmse": NRMSE(),
            "iiee": IIEE(),
            "niiee": NIIEE(),
            "bacc": BACC(),
            "nse": NSE(),
            "anom_corr_coef": ANOM_CORR_COEF(),
            "hybrid_1_100": Hybrid(1, 100)
        }

        self.grid_shape = grid_shape
        self.grid_stride = grid_stride
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        self.to_mask = to_mask
        if self.to_mask == "after_pred" or self.to_mask == "in_metrics":
            mask_path = "/home/ucl/ingi/jlalieu/Thesic/code/src/mask.txt"
            self.mask = torch.zeros((grid_shape, grid_shape), dtype=bool)
            with open(mask_path, "r") as f:
                for r, row in enumerate(f.readlines()):
                    for c, col in enumerate(row.split()):
                        if col == "True":
                            self.mask[r, c] = True
                        else:
                            self.mask[r, c] = False
            self.mask = self.mask.unsqueeze(0).unsqueeze(0)

        self.save_hyperparameters()
        print(self.hparams)
        
        input_shape = 432
        summary(self.architecture, (1, in_channels, input_shape, input_shape))
    
    def configure_optimizers(self):
        if self.optimizer_str == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_str == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_str == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_str == "adagrad":
            opt = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_str == "adadelta":
            opt = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_str == "rmsprop":
            opt = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_str == "adamax":
            opt = torch.optim.Adamax(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer_str} not recognized")
        return opt
    
    def forward(self, x):
        return self.architecture(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.normalize and not self.standardize:
            y_hat = y_hat * self.min_max_dict[self.target_var]["max"]
            y = y * self.min_max_dict[self.target_var]["max"]
        elif self.standardize and not self.normalize:
            y_hat = y_hat * self.mean_std_dict[self.target_var]["std"] + self.mean_std_dict[self.target_var]["mean"]
            y = y * self.mean_std_dict[self.target_var]["std"] + self.mean_std_dict[self.target_var]["mean"]
        elif self.normalize and self.standardize:
            y_hat = y_hat * self.norm_mean_std_dict[self.target_var]["std"] + self.norm_mean_std_dict[self.target_var]["mean"]
            y = y * self.norm_mean_std_dict[self.target_var]["std"] + self.norm_mean_std_dict[self.target_var]["mean"]
            y_hat = y_hat * self.min_max_dict[self.target_var]["max"]
            y = y * self.min_max_dict[self.target_var]["max"]
        if self.post_processing:
            pass
        if self.to_mask == "after_pred":
            device = y_hat.device
            batch_mask = self.mask.repeat(y_hat.shape[0], 1, 1, 1).to(device)
            y_hat = y_hat * batch_mask
            y = y * batch_mask
        if self.to_mask == "in_metrics":
            loss = self.loss_function(y_hat, y, self.mask)
        else:
            loss = self.loss_function(y_hat, y)
        loss = loss.float()
        loss.requires_grad_()
        metrics = self.compute_metrics(y_hat, y)
        #log gradients
        #for name, param in self.architecture.named_parameters():
        #    self.logger.experiment.log({f"{name}_grad": wandb.Histogram(param.grad)})
        return loss, y_hat, y, metrics
    
    def on_train_start(self):
        torch.autograd.set_detect_anomaly(True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, y_hat, y, metrics = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("mem_alloc", torch.cuda.memory_allocated() / 1e9, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        #self.log("mem_reserv", torch.cuda.memory_reserved() / 1e9, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        #self.log("mem_cached", torch.cuda.memory_cached() / 1e9, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            metric_value = metric_value.float()
            self.log(f"train_{metric_name}", metric_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if batch_idx % 100 == 0 and False:
            x_one = x[:8]
            y_one = y[:8]
            y_hat_one = y_hat[:8]
            x_image = x_one[:, 0, :, :].unsqueeze(1)
            y_image = y_one[:, 0, :, :].unsqueeze(1)
            y_hat_image = y_hat_one[:, 0, :, :].unsqueeze(1)
            fig, ax = plt.subplots(3, 8, figsize=(24, 8))
            for i in range(8):
                ax[0, i].imshow(x_image[i].squeeze().detach().cpu().numpy())
                ax[0, i].set_title(f"Input {i}")
                ax[1, i].imshow(y_image[i].squeeze().detach().cpu().numpy())
                ax[1, i].set_title(f"Truth {i}")
                ax[2, i].imshow(y_hat_image[i].squeeze().detach().cpu().numpy())
                ax[2, i].set_title(f"Prediction {i}")
            self.logger.experiment.log({"train_images": wandb.Image(fig)})
            plt.close(fig)
        if not isinstance(loss, torch.Tensor):
            print(loss, type(loss))
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y, metrics = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            metric_value = metric_value.float()
            self.log(f"val_{metric_name}", metric_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y, metrics = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            metric_value = metric_value.float()
            self.log(f"test_{metric_name}", metric_value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        if self.normalize and not self.standardize:
            y_hat = y_hat * self.min_max_dict[self.target_var]["max"]
        elif self.standardize and not self.normalize:
            y_hat = y_hat * self.mean_std_dict[self.target_var]["std"] + self.mean_std_dict[self.target_var]["mean"]
        elif self.normalize and self.standardize:
            y_hat = y_hat * self.norm_mean_std_dict[self.target_var]["std"] + self.norm_mean_std_dict[self.target_var]["mean"]
            y_hat = y_hat * self.min_max_dict[self.target_var]["max"]
        return y_hat
    
    def compute_metrics(self, y_hat, y):
        metrics = {}
        for y_hat_sample, y_sample in zip(y_hat, y):
            y_hat_sample = y_hat_sample.squeeze()
            y_sample = y_sample.squeeze()
            for metric_name, metric_fn in self.metrics.items():
                metrics[metric_name] = metric_fn(y_hat_sample, y_sample)
        return metrics