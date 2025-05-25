import numpy as np
import torch
# import matplotlib.pyplot as plt
# import pandas as pd
# import math

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, attr_name, stage_name):
        score = -val_loss  # Lower val_loss is better, so we negate it
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, attr_name, stage_name)
        elif score > self.best_score + self.delta:  # Changed < to >
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, attr_name, stage_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, attr_name, stage_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model ...')
        checkpoint_path = f"{path}/{stage_name}_{attr_name}_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        self.val_loss_min = val_loss