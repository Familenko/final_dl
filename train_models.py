from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassRecall,
                                         MulticlassPrecision,
                                         BinaryAccuracy,
                                         BinaryRecall,
                                         BinaryPrecision)
from torchmetrics import MeanSquaredError, MeanAbsoluteError


class Trainer(ABC):
    def __init__(self):
        self.metrics = defaultdict(list)
        self.early_stop = False

    @abstractmethod
    def _init_metrics(self):
        pass
    @abstractmethod
    def _reset_metrics(self):
        pass
    @abstractmethod
    def _update_metrics(self, train_loss, test_loss, len_train_data, len_test_data):
        pass
    @abstractmethod
    def _print_metrics(self, epoch, info_every_iter, num_epoch, show_val_metrics):
        pass

    @abstractmethod
    def _train_step(self, train_dataloader):
        pass
    @abstractmethod
    def _eval_step(self, test_dataloader):
        pass

    @abstractmethod
    def _callback_process_epoch(self, epoch):
        pass

    @abstractmethod
    def _callback_process_batch(self, batch, loss):
        pass

    def fit(self, train_dataloader, test_dataloader, num_epoch=10, info_every_iter=1, show_val_metrics=True):
        for epoch in tqdm(range(num_epoch)):

            if self.early_stop:
                break

            self._reset_metrics()

            train_loss = self._train_step(train_dataloader)
            test_loss = self._eval_step(test_dataloader)

            self._update_metrics(train_loss, test_loss, 
                                len(train_dataloader.dataset), len(test_dataloader.dataset))
            
            self._print_metrics(epoch, info_every_iter, num_epoch, show_val_metrics)

            self._callback_process_epoch(epoch)

    def plot_metrics(self, metric=None):
        if metric:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics[f"train_{metric}"], label=f"Train {metric}")
            plt.plot(self.metrics[f"test_{metric}"], label=f"Test {metric}")
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.title(metric)
            plt.legend()
            plt.show()

        else:
            num_metrics = len(self.metrics) // 2
            cols = 2
            rows = (num_metrics + 1) // cols

            fig, axs = plt.subplots(rows, cols, figsize=(10, 6))
            fig.suptitle('Train/Test Metrics', fontsize=16)
            axs = axs.flatten()

            for i, (key, values) in enumerate(self.metrics.items()):
                if 'train_' in key:
                    metric_name = key.replace('train_', '')
                    test_key = f'test_{metric_name}'
                    if test_key in self.metrics:
                        axs[i].plot(values, label=f'Train {metric_name}')
                        axs[i].plot(self.metrics[test_key], label=f'Test {metric_name}')
                        axs[i].set_title(metric_name.capitalize())
                        axs[i].legend()
                else:
                    continue

            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()


class ClassifierTrainer(Trainer):
    def __init__(self, model, criterion, optimizer,
                 num_classes, device = None, callbacks: list = []):
        
        super().__init__()

        self.num_classes = num_classes
        self.device = device if device else torch.device('cpu')

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.callbacks = callbacks

        self._init_metrics()

    def _init_metrics(self):
        metrics_device = torch.device("cpu")

        if self.num_classes == 2:
            self.train_accuracy = BinaryAccuracy().to(metrics_device)
            self.train_recall = BinaryRecall().to(metrics_device)
            self.train_precision = BinaryPrecision().to(metrics_device)

            self.val_accuracy = BinaryAccuracy().to(metrics_device)
            self.val_recall = BinaryRecall().to(metrics_device)
            self.val_precision = BinaryPrecision().to(metrics_device)
        else:
            self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='macro', ignore_index=-100).to(metrics_device)
            self.train_recall = MulticlassRecall(num_classes=self.num_classes, average='macro', ignore_index=-100).to(metrics_device)
            self.train_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro', ignore_index=-100).to(metrics_device)

            self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='macro', ignore_index=-100).to(metrics_device)
            self.val_recall = MulticlassRecall(num_classes=self.num_classes, average='macro', ignore_index=-100).to(metrics_device)
            self.val_precision = MulticlassPrecision(num_classes=self.num_classes, average='macro', ignore_index=-100).to(metrics_device)


    def _reset_metrics(self):
        self.train_accuracy.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        self.val_accuracy.reset()
        self.val_recall.reset()
        self.val_precision.reset()

    def _update_metrics(self, train_loss, test_loss, len_train_data, len_test_data):
        train_loss /= len_train_data
        train_acc = self.train_accuracy.compute().item()
        train_rec = self.train_recall.compute().item()
        train_prec = self.train_precision.compute().item()
        
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_accuracy'].append(train_acc)
        self.metrics['train_recalls'].append(train_rec)
        self.metrics['train_precisions'].append(train_prec)

        test_loss /= len_test_data
        test_acc = self.val_accuracy.compute().item()
        test_rec = self.val_recall.compute().item()
        test_prec = self.val_precision.compute().item()
        
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_accuracy'].append(test_acc)
        self.metrics['test_recalls'].append(test_rec)
        self.metrics['test_precisions'].append(test_prec)

    def _print_metrics(self, epoch, info_every_iter, num_epoch, show_val_metrics):

        if (epoch + 1) % info_every_iter == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                    f"Train Loss: {self.metrics['train_loss'][-1]:.4f} " +
                    f"Acc: {self.metrics['train_accuracy'][-1]:.4f} " +
                    f"Rec: {self.metrics['train_recalls'][-1]:.4f} " +
                    f"Prec: {self.metrics['train_precisions'][-1]:.4f}")
                
            if show_val_metrics:
                print(f"Epoch [{epoch + 1}/{num_epoch}] " +
                        f"Val Loss: {self.metrics['test_loss'][-1]:.4f} " +
                        f"Acc: {self.metrics['test_accuracy'][-1]:.4f} " +
                        f"Rec: {self.metrics['test_recalls'][-1]:.4f} " +
                        f"Prec: {self.metrics['test_precisions'][-1]:.4f}")

    def _define_outputs(self, outputs):
        if self.num_classes == 2 and isinstance(self.criterion, (torch.nn.BCELoss, torch.nn.BCEWithLogitsLoss)) or outputs.shape[1] == 1:
            outputs = outputs.squeeze()
        elif self.num_classes == 2 and isinstance(self.criterion, nn.CrossEntropyLoss):
            outputs = outputs[:, 1]

        return outputs

    def _train_step(self, train_dataloader):
        self.model.train()
        train_loss = 0.0

        for batch_i, batch in enumerate(train_dataloader, start=1):
            inputs, targets, attention_mask = self._parse_batch(batch)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs, attention_mask=attention_mask) if attention_mask is not None else self.model(inputs)

            # Обробка виходів
            original_shape = outputs.shape
            if outputs.dim() == 3:
                # NER: [batch, seq_len, num_classes]
                batch_size, seq_len, _ = outputs.shape
                outputs = outputs.view(batch_size * seq_len, self.num_classes)
                targets = targets.view(batch_size * seq_len)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(-1)
                    # Фільтрація паддінгів
                    outputs = outputs[attention_mask.bool()]
                    targets = targets[attention_mask.bool()]
            elif outputs.dim() == 2:
                # Класифікація: [batch, num_classes] або [batch, 1]
                outputs = self._define_outputs(outputs)
            else:
                raise ValueError(f"Unsupported output shape: {original_shape}")

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            if isinstance(inputs, tuple):
                batch_size = inputs[0].size(0)
            else:
                batch_size = inputs.size(0)
            train_loss += loss.item() * batch_size

            # Метрики
            if outputs.dim() == 1 or outputs.shape[1] == 1:
                preds = (outputs > 0.5).long()
            else:
                preds = outputs.argmax(dim=1)

            self.train_accuracy.update(preds.cpu(), targets.cpu())
            self.train_recall.update(preds.cpu(), targets.cpu())
            self.train_precision.update(preds.cpu(), targets.cpu())

            self._callback_process_batch(batch_i, loss.item())

        return train_loss

    def _eval_step(self, test_dataloader):
        self.model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_dataloader:
                inputs, targets, attention_mask = self._parse_batch(batch)

                outputs = self.model(inputs, attention_mask=attention_mask) if attention_mask is not None else self.model(inputs)
                original_shape = outputs.shape

                if outputs.dim() == 3:
                    # NER: [batch, seq_len, num_classes]
                    batch_size, seq_len, _ = outputs.shape
                    outputs = outputs.view(batch_size * seq_len, self.num_classes)
                    targets = targets.view(batch_size * seq_len)

                    if attention_mask is not None:
                        attention_mask = attention_mask.view(-1)
                        outputs = outputs[attention_mask.bool()]
                        targets = targets[attention_mask.bool()]
                elif outputs.dim() == 2:
                    # Класифікація: [batch, num_classes] або [batch, 1]
                    outputs = self._define_outputs(outputs)
                else:
                    raise ValueError(f"Unsupported output shape: {original_shape}")

                loss = self.criterion(outputs, targets)
                if isinstance(inputs, tuple):
                    batch_size = inputs[0].size(0)
                else:
                    batch_size = inputs.size(0)
                test_loss += loss.item() * batch_size

                # Метрики
                if outputs.dim() == 1 or outputs.shape[1] == 1:
                    preds = (outputs > 0.5).long()
                else:
                    preds = outputs.argmax(dim=1)

                self.val_accuracy.update(preds.cpu(), targets.cpu())
                self.val_recall.update(preds.cpu(), targets.cpu())
                self.val_precision.update(preds.cpu(), targets.cpu())

        return test_loss

    def _parse_batch(self, batch):
        """
        Універсальна обробка батчу.
        Повертає (inputs, targets, attention_mask)
        """
        if isinstance(batch, dict):
            inputs = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            targets = batch['labels'].to(self.device)
        elif len(batch) == 3:
            inputs, attention_mask, targets = batch
            inputs, attention_mask, targets = inputs.to(self.device), attention_mask.to(self.device), targets.to(self.device)
        elif len(batch) == 2:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            attention_mask = None
        else:
            raise ValueError("Batch format not supported.")

        inputs = inputs.float()
        targets = targets.float() if self.num_classes == 2 else targets.long()
        return inputs, targets, attention_mask
        
    def _callback_process_epoch(self, epoch):
        if self.callbacks:
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, self.metrics, self.model)
                if hasattr(cb, 'early_stop') and cb.early_stop:
                    self.early_stop = True

    def _callback_process_batch(self, batch, loss):
        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, 'on_batch_end'):
                    cb.on_batch_end(batch, loss, self.model)
                    if hasattr(cb, 'early_stop') and cb.early_stop:
                        self.early_stop = True


class RegressorTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, 
                 device=None, callbacks: list = []):
        
        super().__init__()
        
        self.device = device if device else torch.device('cpu')

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.callbacks = callbacks

        self._init_metrics()

    def _init_metrics(self):
        self.metrics = defaultdict(list)
        
        metrics_device = torch.device("cpu")

        self.train_mse = MeanSquaredError().to(metrics_device)
        self.train_mae = MeanAbsoluteError().to(metrics_device)
        self.val_mse = MeanSquaredError().to(metrics_device)
        self.val_mae = MeanAbsoluteError().to(metrics_device)

    def _reset_metrics(self):
        self.train_mse.reset()
        self.train_mae.reset()
        self.val_mse.reset()
        self.val_mae.reset()

    def _update_metrics(self, train_loss, test_loss, len_train_data, len_test_data):
        train_loss /= len_train_data
        train_mse = self.train_mse.compute().item()
        train_mae = self.train_mae.compute().item()

        test_loss /= len_test_data
        val_mse = self.val_mse.compute().item()
        val_mae = self.val_mae.compute().item()

        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_mse'].append(train_mse)
        self.metrics['train_mae'].append(train_mae)
        self.metrics['train_rmse'].append(np.sqrt(train_mse))

        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_mse'].append(val_mse)
        self.metrics['test_mae'].append(val_mae)
        self.metrics['test_rmse'].append(np.sqrt(val_mse))

    def _print_metrics(self, epoch, info_every_iter, num_epoch, show_val_metrics):
        if (epoch + 1) % info_every_iter == 0:
            print(f"Epoch [{epoch + 1}/{num_epoch}] "
                  f"Train Loss: {self.metrics['train_loss'][-1]:.4f} "
                  f"MSE: {self.metrics['train_mse'][-1]:.4f} "
                  f"MAE: {self.metrics['train_mae'][-1]:.4f} "
                  f"RMSE: {self.metrics['train_rmse'][-1]:.4f}")

            if show_val_metrics:
                print(f"Epoch [{epoch + 1}/{num_epoch}] "
                      f"Val Loss: {self.metrics['test_loss'][-1]:.4f} "
                      f"MSE: {self.metrics['test_mse'][-1]:.4f} "
                      f"MAE: {self.metrics['test_mae'][-1]:.4f} "
                      f"RMSE: {self.metrics['test_rmse'][-1]:.4f}")

    def _train_step(self, train_dataloader):
        self.model.train()
        train_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = inputs.float(), targets.float()
            targets = targets.view(-1, 1)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            self.train_mse.update(outputs.cpu().squeeze(), targets.cpu().squeeze())
            self.train_mae.update(outputs.cpu().squeeze(), targets.cpu().squeeze())

        return train_loss

    def _eval_step(self, test_dataloader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = inputs.float(), targets.float()
                targets = targets.view(-1, 1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                self.val_mse.update(outputs.cpu().squeeze(), targets.cpu().squeeze())
                self.val_mae.update(outputs.cpu().squeeze(), targets.cpu().squeeze())

        return val_loss

    def _callback_process_epoch(self, epoch):
        if self.callbacks:
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, self.metrics, self.model)
                if hasattr(cb, 'early_stop') and cb.early_stop:
                    self.early_stop = True

    def _callback_process_batch(self, batch, loss):
        if self.callbacks:
            for cb in self.callbacks:
                if hasattr(cb, 'on_batch_end'):
                    cb.on_batch_end(batch, loss, self.model)
                    if hasattr(cb, 'early_stop') and cb.early_stop:
                        self.early_stop = True
