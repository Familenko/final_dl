import matplotlib.pyplot as plt

import torch
from IPython.display import display, HTML
from abc import ABC, abstractmethod


def cprint(text, color="red", bold=True):
    style = f"color: {color};"
    if bold:
        style += " font-weight: bold;"
    display(HTML(f"<span style='{style}'>{text}</span>"))


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch, metrics, model):
        pass


class EarlyStoppingCallback(Callback):
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def on_epoch_end(self, epoch, metrics, model):
        current_score = metrics['test_loss'][-1]
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            cprint(f"Early stopping triggered at epoch {epoch + 1}.", color="red", bold=False)


class ModelCheckpointCallback(Callback):
    def __init__(self, path='best_model.pt', min_delta=0.01):
        self.path = path
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = 0
        self.best_model = None

    def on_epoch_end(self, epoch, metrics, model):
        current_score = metrics['test_loss'][-1]
        
        if self.best_score is None or current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_model = model.state_dict()
            torch.save(self.best_model, self.path)
            cprint(f"Model saved at epoch {epoch + 1} with loss {current_score:.4f}.", color="green", bold=False)


class ReduceLROnPlateauCallback(Callback):
    def __init__(self, optimizer, patience=3, factor=0.5, min_lr=1e-6, min_delta=0.01):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def on_epoch_end(self, epoch, metrics, model):
        current_score = metrics['test_loss'][-1]
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                if old_lr == self.min_lr:
                    return
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                cprint(f"Learning rate reduced on epoch {epoch + 1} from {old_lr:.6f} to {new_lr:.6f}.", color="orange", bold=False)
            self.counter = 0


class LRSchedulerCallback(Callback):
    def __init__(self, optimizer, lr_min=1e-5, lr_max=1e-3, epochs_total=100, growth_rate=10):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epochs_total = epochs_total
        self.growth_rate = growth_rate

        self.lr = []
        self.loss = []

    def schedule(self, epoch):
        progress = epoch / self.epochs_total
        lr = self.lr_min * (self.lr_max / self.lr_min) ** (progress ** self.growth_rate)
        return lr

    def on_epoch_end(self, epoch, metrics, model):
        new_lr = self.schedule(epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        metrics['lr'].append(new_lr)

        self.lr.append(new_lr)
        self.loss.append(metrics['train_loss'][-1])

    def plot_lr(self, x1=None, x2=None, y1=None, y2=None):

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.semilogx(self.lr, self.loss)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")

        x_min = x1 if x1 is not None else self.lr_min
        x_max = x2 if x2 is not None else self.lr_max
        y_min = y1 if y1 is not None else 0
        y_max = y2 if y2 is not None else max(self.loss) * 1.1

        plt.axis([x_min, x_max, y_min, y_max])
        plt.show()
