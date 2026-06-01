import torch
import os
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, metric):
        if self.mode == 'min':
            if metric < self.best - self.min_delta:
                self.best = metric
                self.counter = 0
                return True
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                return False
        else:
            if metric > self.best + self.min_delta:
                self.best = metric
                self.counter = 0
                return True
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                return False


class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_top_k=3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_k = keep_top_k
        self.saved = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model_state, optimizer_state, epoch, val_loss, config, tag=''):
        filename = f"ckpt_epoch{epoch:04d}_loss{val_loss:.5f}"
        if tag:
            filename += f"_{tag}"
        filename += '.pt'
        filepath = os.path.join(self.checkpoint_dir, filename)

        torch.save({
            'epoch': epoch,
            'model_state': model_state,
            'optim_state': optimizer_state,
            'val_loss': val_loss,
            'config': config,
        }, filepath)

        self.saved.append((val_loss, filepath))
        self.saved.sort(key=lambda x: x[0])

        while len(self.saved) > self.keep_top_k:
            _, old_path = self.saved.pop()
            if os.path.exists(old_path):
                os.remove(old_path)

        return filepath

    def get_best_path(self):
        if not self.saved:
            return None
        return self.saved[0][1]


def log_metrics(writer, metrics, step, prefix=''):
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}{key}', value, step)


def save_training_config(config, output_path):
    import yaml
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
