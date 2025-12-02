import tensorflow as tf
import time

class PrintEpochMetrics(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        m = int(epoch_time//60)
        s = int(epoch_time % 60)
        
        train_loss = logs['loss']
        train_acc = logs['accuracy'] * 100
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy') * 100
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {m}m {s}s')
        print(f'\tTrain Loss: {train_loss:.5f} | Train Accuracy = {train_acc:.4f} ')
        print(f'\t Val. Loss: {val_loss:.5f} | Valid Accuracy = {val_acc:.4f} ')