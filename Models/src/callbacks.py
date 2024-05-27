from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    DeviceStatsMonitor,
    ModelSummary,
)


class PrintCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is ended!")
