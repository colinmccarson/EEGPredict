from keras.callbacks import Callback
import os
from datetime import datetime


class ExportModelWeights(Callback):
    def __init__(self, directory='./weights', monitor='val_loss'):
        super(ExportModelWeights, self).__init__()
        self.directory = directory
        self.monitor = monitor
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        if val_loss is None:
            return

        if val_loss >= self.best_val_loss:
            return

        self.best_val_loss = val_loss

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        str_val_loss = str(val_loss).replace('.', '_')
        filename = os.path.join(self.directory, f'weights_t{timestamp}_v{str_val_loss}.weights.h5')

        self.model.save_weights(filename, overwrite=True)
        print("Model weights saved at epoch", epoch + 1, "to", filename)

