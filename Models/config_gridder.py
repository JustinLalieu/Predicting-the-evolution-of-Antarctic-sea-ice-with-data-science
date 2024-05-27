import sys
from datetime import datetime
from src.architectures.Unet5 import Unet5
from src.architectures.IceNetLike import IceNetLike
from src.architectures.UnetS import UnetS
from src.architectures.Linear import Linear
from src.architectures.LinearB import LinearB
from src.architectures.Persistence import Persistence
from src.architectures.LinearMat import LinearMat

#data_dir = "/home/ucl/ingi/jlalieu/Thesic/data/PER_YEAR"
data_dir = "/Users/justinlalieu/Unif/Thesis/data/PER_YEAR"

target_vars = ["SIC_NEMO+0"]

input_vars = [
    #"SIC_NEMO-30", 
    #"SIC_NEMO-60", 
    #"SIC_NEMO-90", 
    #"SIC_NEMO-365",
    "SIC_NEMO_CLIM-_15x365",
    #"SNTHIC_NEMO-30", 
    #"SNTHIC_NEMO-60", 
    #"SNTHIC_NEMO-90", 
    #"SNTHIC_NEMO-365",
    #"MLD_NEMO-30", 
    #"MLD_NEMO-60",
    #"MLD_NEMO-90", 
    #"MLD_NEMO-365",
    ]
architecture_name = "Persistence"
learning_rate = 0.0001
max_epochs = 0
to_mask = ""

normalize = True
standardize = True
post_processing = False # FAAAAAALSSSEEEEEE
grid_shape = 432
grid_stride = 1  # At least greater than 0
loss_function = "MAE"
num_workers = 10
batch_size = 32
shuffle_train = True
shuffle_val = False
optimizer_str = "adam"

train_window = (datetime(1961, 1, 1), datetime(2016, 12, 31))
val_window = (datetime(2017, 1, 1), datetime(2019, 12, 31))
test_window = (datetime(2020, 1, 1), datetime(2023, 11, 30))

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

