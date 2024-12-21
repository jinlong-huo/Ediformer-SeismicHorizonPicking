    
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class SimpleDataset(Dataset):
    """
    Custom Dataset for Horizon Picking data
    
    Parameters:
    -----------
    data : torch.Tensor
        Input data tensor
    labels : torch.Tensor
        Corresponding labels tensor
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class HorizonDataFactory:
    """
    Data processing factory for Horizon Picking dataset
    
    Handles data loading, preprocessing, and splitting for multiple attributes
    """
    def __init__(self, 
                 attr_dirs: dict,  # Dictionary with attribute names as keys and directories as values
                 kernel_size: tuple = (1, 288, 64),
                 stride: tuple = (1, 64, 32),
                 train_ratio: float = 0.8,
                 batch_size: int = 16,
                 random_seed: int = 0):
        """
        Initialize the data factory
        
        Parameters:
        -----------
        attr_dirs : dict
            Dictionary mapping attribute names to their respective data directories
        kernel_size : tuple, optional
            Size of the sliding window/kernel (default: (1, 288, 64))
        stride : tuple, optional
            Stride for the sliding window (default: (1, 64, 32))
        train_ratio : float, optional
            Proportion of data to use for training (default: 0.8)
        batch_size : int, optional
            Batch size for DataLoaders (default: 16)
        random_seed : int, optional
            Seed for reproducibility (default: 0)
        """
        self.attr_dirs = attr_dirs
        self.kc, self.kh, self.kw = kernel_size
        self.dc, self.dh, self.dw = stride
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Initialize storage for dataloaders
        self.attribute_dataloaders = {}
        print("============ Begin processing dataset ========\n")
        
        # Process each attribute's data
        for attr_name, dir_paths in self.attr_dirs.items():
            data_path = dir_paths['data']
            label_path = dir_paths['label']
            self._process_attribute_data(attr_name, data_path, label_path)
            
            
    def _process_attribute_data(self, attr_name, data_path, label_path):
        """
        Process and create dataloaders for a specific attribute
        """
        # Load data
        data = np.load(data_path)
        labels = np.load(label_path)
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        data = (data - mean) / std
        
        # Convert to tensors
        data = torch.tensor(data, dtype=torch.float) # 571551 288
        labels = torch.tensor(labels, dtype=torch.float)
        
        # Reshape
        data = data.reshape(-1, 951, 288)                # 601 951 288
        labels = labels.reshape(-1, 951, 288)
        
        # To align with dip attr we remove last inline of data and labels
        data = data[:600, :, :]
        labels = labels[:600, :, :]
        
        # Change here to adjust data volume you use
        data = data[::100, ::10, :]
        labels = labels[::100, ::10, :]
        
        # data = data[::5, ::5, :]
        # labels = labels[::5, ::5, :]
        
        # Add batch dimension and permute
        data = np.swapaxes(data, -1, 1)
        labels = np.swapaxes(labels, -1, 1)
        
        data = data[np.newaxis, :]    # b, c, 288, 10
        labels = labels[np.newaxis, :]  # b, c, 288, 10
        
        # Pad data to be divisible by kernel size
        data = F.pad(data, [
            data.size(3) % self.kw // 2, data.size(3) % self.kw // 2,
            data.size(2) % self.kh // 2, data.size(2) % self.kh // 2,
            data.size(1) % self.kc // 2, data.size(1) % self.kc // 2
        ])
        
        # Apply sliding window (unfold)
        data = data.unfold(1, self.kc, self.dc).unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        data = data.contiguous().view(-1, self.kc, self.kh, self.kw)
        data = data.reshape(data.shape[0], -1, self.kh, self.kw)
        
        # Pad and unfold labels
        labels = F.pad(labels, [
            labels.size(3) % self.kw // 2, labels.size(3) % self.kw // 2,
            labels.size(2) % self.kh // 2, labels.size(2) % self.kh // 2,
            labels.size(1) % self.kc // 2, labels.size(1) % self.kc // 2
        ])
        labels = labels.unfold(1, self.kc, self.dc).unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        labels = labels.contiguous().view(-1, self.kc, self.kh, self.kw)
        labels = labels.reshape(labels.shape[0], -1, self.kh, self.kw)
        
        print(f"Processed {attr_name} Data size: ", data.shape)
        print(f"Processed {attr_name} Labels size: {labels.shape}\n")
        
        # Create dataset
        # data = data.permute(0, 1, -1, 2)
        # labels = labels.permute(0, 1, -1, 2)
        
        full_dataset = SimpleDataset(data, labels)
        
        # Split dataset
        length = len(full_dataset)
        train_size = int(self.train_ratio * length)
        val_size = length - train_size
        generator = torch.Generator().manual_seed(self.random_seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        # Create dataloaders 
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(full_dataset, batch_size=self.batch_size)
        
        # Store dataloaders
        self.attribute_dataloaders[attr_name] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
    
    def get_dataloaders(self, attribute_names):
        """
        Return dataloaders for specified attributes
        
        Parameters:
        -----------
        attribute_names : list
            List of attribute names to retrieve dataloaders for
        
        Returns:
        --------
        list of tuples
            Dataloaders for each attribute (train, val, test)
        """
        loaders = {}
        for attr in attribute_names:
            if attr not in self.attribute_dataloaders:
                raise ValueError(f"Attribute '{attr}' not found in processed attributes.")
            loaders[attr] = self.attribute_dataloaders[attr]
            
        print("============ Process dataset done ============\n")
        return loaders


# def main():
#     attr_dirs = {
#         "freq": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy", "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"},
#         "phase": {"data": "/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_phase.npy", "label": "/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy"}
#     }
    
#     data_factory = HorizonDataFactory(attr_dirs=attr_dirs, kernel_size=(1, 288, 64), stride=(1, 64, 32))
    
#     dataloaders = data_factory.get_dataloaders(["freq", "phase"])
    
#     train_loaders = [loaders["train"] for _, loaders in dataloaders.items()]
#     val_loaders = [loaders["val"] for _, loaders in dataloaders.items()]
#     test_loaders = [loaders["test"] for _, loaders in dataloaders.items()]
#     # attribute_val_loaders = [attribute_dataloaders["val"] for attr, loaders in attribute_dataloaders.items()]
#     # attribute_test_loaders = [attribute_dataloaders["test"] for attr, loaders in attribute_dataloaders.items()]
    
#     # for idx, loader in enumerate(train_loaders):
#     #     print(f"Train Loader for Attribute {list(dataloaders.keys())[idx]}:")
#     #     print(f"  Number of batches: {len(loader)}")
#     # for attr, loaders in dataloaders.items():
#     #     print(f"Attribute: {attr}")
#     print(f"Train size: {len(train_loaders)}")
#     print(f"Train size: {len(val_loaders)}")
#     print(f"Train size: {len(test_loaders)}")
    
# if __name__ == "__main__":
#     main()
