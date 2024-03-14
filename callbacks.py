from keras.callbacks import Callback
import os
from datetime import datetime
from keras import saving


class ExportModel(Callback):
    def __init__(self, model_name, directory='./weights', monitor='val_loss'):
        super(ExportModel, self).__init__()
        self.directory = directory
        self.monitor = monitor
        self.best_val_loss = float('inf')
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        if val_loss is None:
            return

        if val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        str_val_loss = str(val_loss).replace('.,)( ', '_')
        filename = os.path.join(self.directory, f'model_{self.model_name}.keras')

        self.model.save(filename, overwrite=True)
        saving.save_model(self.model, filename)

        print("\nModel saved at epoch", epoch + 1, "to", filename)

