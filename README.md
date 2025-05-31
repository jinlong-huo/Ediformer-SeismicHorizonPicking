# Ediformer: Seismic Horizon Picking with Transformer-Based Models

A deep learning framework for automated seismic horizon detection using transformer-based architectures and ensemble learning methods.

## 🚀 Features

- **Multi-Model Architecture**: Ediformer (Dense Inception Transformer) with ensemble learning
- **Ensemble Learning**: Combines multiple seismic attributes for improved accuracy
- **Visualization Tools**: Comprehensive plotting and analysis utilities
- **Flexible Training**: Support for patch-based and full-volume training
- **Multiple Attributes**: Works with amplitude, frequency, phase, dip, RMS, and coherence data
- **Memory-Efficient Fusion**: Lightweight attention-based feature fusion

## 📋 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Ediformer-SeismicHorizonPicking.git
cd Ediformer-SeismicHorizonPicking

# Install dependencies
pip install torch torchvision numpy matplotlib scipy scikit-learn seaborn pandas
pip install mayavi  # For 3D visualization (optional)
pip install shap    # For feature importance analysis (optional)
```

### 2. Data Preparation
Place your seismic attribute files in a data directory structure:
```
data/
├── F3_seismic.npy                    # Original seismic amplitude
├── F3_crop_horizon_freq.npy          # Instantaneous frequency
├── F3_crop_horizon_phase.npy         # Instantaneous phase
├── F3_predict_MCDL_crossline.npy     # Dip attribute
├── F3_RMSAmp.npy                     # RMS amplitude
├── F3_coherence.npy                  # Coherence attribute
└── test_label_no_ohe.npy             # Ground truth labels
```

### 3. Training

#### Single Model Training
```bash
# Train individual Ediformer model
python Ediformer/train_ediformer.py \
    --data_path /path/to/data \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-3
```

#### Ensemble Training
```bash
# Train ensemble with multiple attributes
python Ediformer/ensembler.py \
    --is_training True \
    --attr_dirs "/path/to/seismic,/path/to/dip" \
    --num_epochs 20 \
    --mm_ckpt_path ./checkpoints/meta_models \
    --fm_ckpt_path ./checkpoints/fusion_model
```

### 4. Testing and Inference
```bash
# Test ensemble model
python Ediformer/ensembler.py \
    --is_testing True \
    --mm_ckpt_path ./checkpoints/meta_models \
    --fm_ckpt_path ./checkpoints/fusion_model \
    --output_path ./results
```

### 5. Visualization
```bash
# Visualize seismic attributes
python utils/seisvis.py --data_path /path/to/data

# Generate attribute correlation pairplots
python utils/seispair.py --data_path /path/to/data

# 3D volume visualization
python utils/seis3dvis.py --volume_path /path/to/volume.npy

# SHAP feature importance analysis
python utils/attrshap.py --model_path ./checkpoints --data_path /path/to/data
```

## 📁 Project Structure

```
Ediformer-SeismicHorizonPicking/
├── Ediformer/
│   ├── models/
│   │   ├── diformer_patch_attn.py    # Main Ediformer architecture
│   │   ├── memfusion.py              # Memory-efficient fusion model
│   │   └── __init__.py
│   ├── ensembler.py                  # Advanced ensemble learning
│   ├── train_ediformer.py            # Single model training
│   └── config.py                     # Configuration settings
├── utils/
│   ├── seisvis.py                    # Seismic attribute visualization
│   ├── seispair.py                   # Attribute pairplot generator
│   ├── seis3dvis.py                  # 3D volume visualization
│   ├── attrshap.py                   # SHAP importance analysis
│   ├── unpatch.py                    # Patch-to-volume converter
│   ├── datafactory.py                # Data processing utilities
│   ├── tools.py                      # Training utilities
│   └── __init__.py
├── checkpoints/                      # Model checkpoints
├── results/                          # Output predictions
├── data/                            # Seismic data files
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT License
```

## 🏗️ Model Architecture

### 1. Ediformer (Dense Inception Transformer)
- **Purpose**: Attention-based seismic horizon segmentation with dense connections
- **Input**: `[batch_size, 1, height, width]` - Seismic patches or full sections
- **Output**: `[batch_size, 7, height, width]` - 7-class horizon predictions
- **Architecture**: 
  - Dense CNN blocks for feature extraction
  - Mixer attention mechanisms for long-range dependencies
  - Skip connections for gradient flow
  - Multi-scale feature fusion

### 2. Memory-Efficient Fusion Model
- **Purpose**: Lightweight multi-attribute feature fusion
- **Input**: Concatenated features from multiple Ediformer models
- **Output**: Final ensemble predictions
- **Architecture**: Sparse U-Net with cross-attention mechanisms

### 3. Advanced Ensemble Learning
- **Purpose**: Meta-learning across multiple seismic attributes
- **Components**: 
  - Individual meta-models for each attribute
  - Fusion model for combining predictions
- **Training Strategy**: Two-stage training approach
  1. Train individual meta-models on specific attributes
  2. Train fusion model on combined features

## 🛠️ Key Components

### Core Models
- **Diformer**: Dense Inception Transformer with attention
- **MemFusion**: Memory-efficient U-Net fusion architecture
- **Ensemble**: Advanced multi-attribute ensemble learner

### Attention Mechanisms
- **High Mixer**: Convolutional attention for local features
- **Low Mixer**: Self-attention for global context
- **Mixer**: Combined attention mechanism

### Training Utilities
- **EarlyStopping**: Prevent overfitting with patience-based stopping
- **DataFactory**: Efficient data loading and preprocessing
- **Tools**: Training loops, metrics, and checkpointing

## 💡 Usage Examples

### Single Model Training
```python
from Ediformer.models.diformer_patch_attn import Diformer
import torch

# Initialize model
model = Diformer(
    dim=[72, 144, 288, 288],  # Feature dimensions for each block
    num_heads=[2, 4, 8, 8],   # Attention heads for each block
    feature_projection_dim=288
)

# Load data and train
input_tensor = torch.randn(16, 1, 16, 288)
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # [16, 7, 16, 288]
```

### Ensemble Training
```python
from Ediformer.ensembler import AdvancedEnsembleLearner

# Initialize ensemble
ensemble = AdvancedEnsembleLearner(
    dim=[72, 144, 288, 288],
    num_heads=[2, 4, 8, 8],
    num_classes=7,
    num_classifiers=2,  # Number of attributes
    height=16,
    width=288
)

# Train ensemble
ensemble.train_ensemble(
    attribute_dataloaders,
    validation_dataloaders,
    epochs=20
)
```

### Visualization
```python
from utils.seisvis import SeismicVisualizer
import numpy as np

# Load seismic data
seismic = np.load('data/F3_seismic.npy')
dip = np.load('data/F3_predict_MCDL_crossline.npy')

# Visualize
visualizer = SeismicVisualizer()
visualizer.plot_attributes_comparison({
    'Seismic': seismic[100],  # Select a slice
    'Dip': dip[100]
})
```

## 📊 Expected Performance

### Dataset Metrics
- **F3 Seismic Volume**: 650 × 951 × 462 samples
- **Horizon Classes**: 7 distinct geological horizons
- **Training Patches**: 16 × 288 sliding window approach
- **Validation Split**: 80/20 train/validation

### Model Performance
- **Single Attribute Accuracy**: ~85-90%
- **Ensemble Accuracy**: ~92-95%
- **Training Time**: ~2-4 hours on GPU
- **Inference Speed**: Real-time on modern GPUs

## 🔧 Configuration

### Model Parameters
```python
# Default configuration
CONFIG = {
    'dim': [72, 144, 288, 288],
    'num_heads': [2, 4, 8, 8],
    'num_classes': 7,
    'height': 16,
    'width': 288,
    'learning_rate': 1e-3,
    'batch_size': 16,
    'num_epochs': 100
}
```

### Data Paths
Update paths in configuration files or command line arguments:
```bash
--data_path /path/to/seismic/data
--checkpoint_path /path/to/save/models
--output_path /path/to/save/predictions
```

## 📚 Dependencies

### Core Requirements
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
matplotlib>=3.3.0
scipy>=1.6.0
scikit-learn>=0.24.0
```

### Visualization (Optional)
```
seaborn>=0.11.0
pandas>=1.2.0
mayavi>=4.7.0
plotly>=5.0.0
```

### Analysis (Optional)
```
shap>=0.39.0
tensorboard>=2.4.0
wandb>=0.12.0
```

## 🎯 Results and Applications

### Geological Applications
- **Horizon Interpretation**: Automated detection of geological boundaries
- **Structural Analysis**: Enhanced understanding of subsurface geology
- **Seismic Processing**: Quality control and interpretation assistance

### Technical Achievements
- **Multi-attribute Fusion**: Improved accuracy through ensemble learning
- **Attention Mechanisms**: Better handling of long-range dependencies
- **Memory Efficiency**: Scalable to large seismic volumes

## 📖 Citation

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

## 🔗 References

- **Paper**: [Seismic Attributes Aided Horizon Interpretation Using an Ensemble Dense Inception Transformer Network](https://doi.org/10.1109/TGRS.2024.3349687)
- **Google Drive**: [Paper PDF](https://drive.google.com/file/d/1HQQIZkRjQfuteo0zcAJ5NT2TxK3KuyfU/view?usp=drive_link)
- **DexiNed**: [Dense Extreme Inception Network for edge detection](https://arxiv.org/abs/2112.02250) *(Modified for horizon picking)*
- **Transformer Architecture**: [Attention mechanisms for computer vision](https://arxiv.org/abs/2205.12956) *(Dense blocks + mixer attention)*
- **Ensemble Learning**: [Multi-attribute fusion approaches](https://ieeexplore.ieee.org/document/9709337) *(Ediformer + Sparse U-Net fusion)*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaborations, please contact: **Email**: [jinlong.huo99@gmail.com]
