from keras.callbacks import ModelCheckpoint
class ComplexModelCheckpoint(ModelCheckpoint):
    def __init__(self, models, filepaths, monitor='val_loss', verbose=0,
                     save_best_only=False, save_weights_only=False,
                     mode='auto', period=1):                     
        super(ComplexModelCheckpoint, self).__init__(filepaths[0], monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
            
        self.models = models
        self.filepaths = filepaths

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            for i in range(len(self.filepaths)):
                self.filepaths[i] = self.filepaths[i].format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current                            
                        for i in range(len(self.models)):
                            if self.save_weights_only:
                                self.models[i].save_weights(self.filepaths[i], overwrite=True)
                            else:
                                self.models[i].save(self.filepaths[i], overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                for i in range(len(self.models)):
                    if self.save_weights_only:
                        self.models[i].save_weights(self.filepaths[i], overwrite=True)
                    else:
                        self.models[i].save(self.filepaths[i], overwrite=True)