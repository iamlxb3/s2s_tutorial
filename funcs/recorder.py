"""
记录没一个epoch模型的表现，然后再决定要不要保存模型，一般是保存当前最好的一个模型。
"""

class EpochRecorder():
    def __init__(self):
        self._epoch_train_losses = []
        self._epoch_val_losses = []

    @property
    def lowest_val_loss(self):
        if self._epoch_val_losses:
            lowest_val_loss = min(self._epoch_val_losses)
            index = self._epoch_val_losses.index(lowest_val_loss)
        else:
            lowest_val_loss = float('inf')
            index = -1
        return tuple((lowest_val_loss, index))

    def val_loss_update(self, loss):
        self._epoch_val_losses.append(loss)

    def train_loss_update(self, loss):
        self._epoch_train_losses.append(loss)
