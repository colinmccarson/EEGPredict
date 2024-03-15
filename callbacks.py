from keras.callbacks import Callback
import os
from datetime import datetime
from keras import saving


class ExportModel(Callback):
    def __init__(self, model_name, directory='./weights', monitor='val_accuracy'):
        super(ExportModel, self).__init__()
        self.directory = directory
        self.monitor = monitor
        self.best_val_acc = 0
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get(self.monitor)
        if val_acc is None:
            return

        if val_acc <= self.best_val_acc:
            return

        self.best_val_acc = val_acc

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        str_val_loss = str(val_acc).replace('.', '_')
        filename = os.path.join(self.directory, f'model_{self.model_name}_v{str_val_loss}.keras')

        self.model.save(filename, overwrite=True)
        saving.save_model(self.model, filename)

        print("\nModel saved at epoch", epoch + 1, "to", filename)

class ExportModelTest70(Callback):
    def __init__(self, model_name, xtest, ytest, directory='./weights', monitor='val_accuracy'):
        super(ExportModelTest70, self).__init__()
        self.directory = directory
        self.monitor = monitor
        self.best_val_acc = 0
        self.model_name = model_name
        self.xtest = xtest
        self.ytest = ytest

    def on_epoch_end(self, epoch, logs=None):
        if self.model.evaluate(self.xtest, self.ytest, verbose=0)[1] >= .7:
            filename = os.path.join(self.directory, f'model_{self.model_name}_test70.keras')
            saving.save_model(self.model, filename)
            print("\nModel saved at epoch", epoch + 1, "to", filename)

