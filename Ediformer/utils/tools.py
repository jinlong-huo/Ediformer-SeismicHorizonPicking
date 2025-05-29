import os
import torch
from datetime import datetime

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def save_checkpoint(self, val_loss, model, path, name, stage_name):
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Add timestamp for unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(path, f'{stage_name}_{name}_checkpoint_{timestamp}.pth')
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'timestamp': timestamp
        }, checkpoint_path)
        
        if self.verbose:
            print(f'Validation loss improved. Saving model to {checkpoint_path}')