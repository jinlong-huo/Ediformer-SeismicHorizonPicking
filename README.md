# Seismic Horizon Picking with Transformer-Based Models

A deep learning framework for automated seismic horizon detection using transformer-based architectures and ensemble learning methods.

## Overview

This project implements advanced neural network models for seismic horizon picking, combining:
- **DexiNed**: [Dense Extreme Inception Network for edge detection](https://arxiv.org/abs/2112.02250) ```(The model I modified for horizon picking)```
- **Diformer**: [Transformer-based model with attention mechanisms](https://arxiv.org/abs/2205.12956) ```(I replace the dense blocks with mixer attention)```
- **Ensemble Learning**: [Multi-attribute fusion for improved accuracy](https://ieeexplore.ieee.org/document/9709337) ```(Ensemble base model diformer with a Sparse unet as fusion model)```
- **Memory-Efficient Fusion**: Optimized attention mechanisms for large-scale processing ```(The fusion model is modified to be memory-efficient)```

## Environment Requirements

### Dependencies
```bash
# Core ML frameworks
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.20.0
scikit-learn>=0.24.0

# Image processing and visualization
matplotlib>=3.3.0
imageio>=2.9.0
seaborn>=0.11.0

# Model utilities
timm>=0.4.0
torchsummary>=1.5.0

# Data processing
pandas>=1.3.0

# Optional: For advanced features
tensorboard>=2.5.0  # For training visualization

Basically, if you lack some package just install in your env.
```

<!-- ### Hardware Requirements
You need to make sure you have enough resources to run the project, i.e., do not conflict with other's running projects.
- **GPU**: CUDA-compatible GPU with ≥8GB VRAM recommended
- **RAM**: ≥16GB system memory
- **Storage**: ≥50GB free space for datasets and models -->



## Project Structure

```
├── models/
│   ├── dexined.py          # DexiNed edge detection model
│   ├── diformer.py         # Transformer-based seismic model
│   ├── memfusion.py        # Memory-efficient fusion model
│   └── DOD_ensemble.py     # Ensemble model components
├── utils/
│   ├── datafactory.py      # Data processing utilities
│   └── tools.py           # Training utilities (early stopping, etc.)
├── training/
│   ├── DFormer_patch_main.py    # Patch-based training script
│   ├── DFormer_seis.py          # Seismic-specific training
│   ├── train_seismic.py         # General seismic training
│   └── ensemseis.py             # Ensemble training pipeline
└── ensembler.py            # Main ensemble orchestrator
```

## Models

### 1. DexiNed (Dense Extreme Inception Network)
- **Purpose**: Edge detection for seismic horizons
- **Input**: `[batch_size, 1, height, width]` - Single-channel seismic data
- **Output**: List of 7 tensors (multi-scale edge maps) + ensemble result
- **Typical dimensions**: `[batch_size, 1, 16, 288]`

### 2. Diformer (Transformer-based Model)
- **Purpose**: Seismic horizon detection with attention mechanisms
- **Input**: `[batch_size, 1, height, width]` - Seismic patches
- **Output**: `[batch_size, 7, height, width]` - 7-class horizon predictions
- **Features**: Combines dense blocks with transformer attention

### 3. Ensemble Learning System
- **Purpose**: Multi-attribute fusion for improved accuracy
- **Attributes**: Seismic, phase, dip, frequency, amplitude, coherence, etc.
- **Output**: Combined predictions with enhanced performance

## Quick Start

### 1. Data Preparation
Organize your seismic data as NumPy arrays:
```python
# Required data format
seismic_data.shape    # (n_samples, height, width) or (height, width, depth)
seismic_labels.shape  # Same dimensions with integer class labels (0-6)
```

### 2. Single Model Training

#### DexiNed Training
```bash
python DFormer_patch_main.py --train True --num_epochs 20 --batch_size 10
```

#### Diformer Training
```bash
python train_seismic.py --is_testing False
```

### 3. Ensemble Training
```bash
python ensemseis.py --is_training True --num_epochs 100 --batch_size 36
```

### 4. Testing/Inference
```bash
# Single model testing
python DFormer_patch_main.py --test True

# Ensemble testing
python ensemseis.py --is_testing True
```

## Configuration

### Data Paths Configuration
Edit the `attr_dirs` dictionary in your training scripts:

```python
attr_dirs = {
    "seismic": {
        "data": "/path/to/seismic_data.npy", 
        "label": "/path/to/labels.npy"
    },
    "phase": {
        "data": "/path/to/phase_data.npy", 
        "label": "/path/to/labels.npy"
    },
    # Add more attributes as needed
}
```

### Model Parameters
Key hyperparameters you can adjust:

```python
# Model architecture
embed_dims = [72, 36, 36, 36]  # Transformer dimensions
num_heads = 2                   # Attention heads
num_classes = 7                 # Horizon classes

# Training
batch_size = 36
learning_rate = 0.001
num_epochs = 100
patch_size = (1, 288, 16)      # (channels, height, width)
```

## Input Data Requirements

### Data Format
- **File type**: NumPy arrays (.npy files)
- **Data type**: Float32 or Float64
- **Normalization**: Data will be automatically normalized using RobustScaler

### Seismic Data
- **Shape**: `(n_samples, height, width)` where:
  - `n_samples`: Number of seismic sections/volumes
  - `height`: Vertical dimension (time/depth samples)
  - `width`: Horizontal dimension (traces/channels)
- **Typical dimensions**: `(601, 951, 288)` → patches of `(batch, 1, 16, 288)`

### Labels
- **Shape**: Same as input data
- **Values**: Integer class labels (0-6 for 7-class horizon detection)
- **Classes**:
  - 0: Background/no horizon
  - 1-6: Different horizon types/confidence levels

### Multi-Attribute Data
For ensemble learning, provide multiple seismic attributes:
- **Seismic amplitude**
- **Instantaneous phase**
- **Dip angle**
- **Dominant frequency**
- **RMS amplitude**
- **Coherence**
- **Complex trace attributes**

## Output

### Model Predictions
- **Shape**: `[batch_size, num_classes, height, width]`
- **Format**: Class probability distributions or predicted class indices
- **Files**: Saved as `.npy` files with timestamps

### Ensemble Results
- **Individual predictions**: Separate files for each attribute model
- **Fused predictions**: Combined results from ensemble fusion
- **Metrics**: Accuracy, precision, recall, F1-scores saved as text files

### Typical Output Files
```
outputs/
├── 2025_01_XX_predictions.npy           # Main predictions
├── 2025_01_XX_labels.npy               # Ground truth labels  
├── 2025_01_XX_metrics.txt              # Performance metrics
├── attention_weights/                   # Attention visualizations
└── checkpoints/                        # Trained model weights
    ├── meta_seismic_checkpoint.pth
    ├── meta_phase_checkpoint.pth
    └── fusion_combined_checkpoint.pth
```

<!-- ## Advanced Features

### Memory-Efficient Processing
The framework includes optimized components for large datasets:
- Sparse attention mechanisms
- Lightweight channel attention  
- Grouped convolutions
- Gradient checkpointing

### Attention Visualization
```python
# Extract and visualize attention maps
model = Diformer(dim=[72, 36, 36, 36], num_heads=2)
output, attn_results = model(input_tensor)

# attn_results contains attention weights from each transformer block
import matplotlib.pyplot as plt
plt.imshow(attn_results[0].numpy(), cmap='viridis')
plt.title("Attention Weights - Block 1")
plt.show()
```

### Cross-Validation
Built-in 5-fold cross-validation for robust model evaluation:
```python
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True)
# Automatically handled in training scripts
```

## Performance Optimization

### GPU Configuration
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Multi-GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Mixed Precision Training
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input_data)
    loss = criterion(output, labels)
``` -->
<!-- 
## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Data Loading Errors**
   - Check file paths in `attr_dirs`
   - Verify NumPy array dimensions
   - Ensure consistent data types

3. **Model Convergence Issues**
   - Adjust learning rate
   - Check data normalization
   - Verify label encoding (0-6 for 7 classes)

### Performance Tips
- Use multiple GPUs for ensemble training
- Implement data augmentation for small datasets
- Monitor validation loss for early stopping
- Use TensorBoard for training visualization -->

## Citation

If you use this code in your research, please cite:
```bibtex
@article{liu2024seismic,
  title={Seismic attributes aided horizon interpretation using an ensemble dense inception transformer network},
  author={Liu, Naihao and Huo, Jinlong and Li, Zhuo and Wu, Hao and Lou, Yihuai and Gao, Jinghuai},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--10},
  year={2024},
  publisher={IEEE}
}
```
## Codes for Paper
[Seismic Attributes Aided Horizon Interpretation Using an Ensemble Dense Inception Transformer Network](https://doi.org/10.1109/TGRS.2024.3349687) or >[Google Drive](https://drive.google.com/file/d/1HQQIZkRjQfuteo0zcAJ5NT2TxK3KuyfU/view?usp=drive_link)

## License

[[Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0)]



