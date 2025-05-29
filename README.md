# Seismic Horizon Picking with Transformer-Based Models

A deep learning framework for automated seismic horizon detection using transformer-based architectures and ensemble learning methods.

## Features

- **Multi-Model Architecture**: DexiNed, Diformer, and Memory-Efficient Fusion models
- **Ensemble Learning**: Combines multiple seismic attributes for improved accuracy
- **Visualization Tools**: Comprehensive plotting and analysis utilities
- **Flexible Training**: Support for patch-based and full-volume training
- **Multiple Attributes**: Works with amplitude, frequency, phase, dip, RMS, and coherence data

## Quick Start

### 1. Installation
```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn
```

### 2. Data Preparation
Place your seismic attribute files in a data directory:
- `F3_seismic.npy` - Original seismic amplitude
- `F3_crop_horizon_freq.npy` - Instantaneous frequency
- `F3_crop_horizon_phase.npy` - Instantaneous phase
- `F3_predict_MCDL_crossline.npy` - Dip attribute
- `F3_RMSAmp.npy` - RMS amplitude
- `F3_coherence.npy` - Coherence attribute
- `test_label_no_ohe.npy` - Ground truth labels

### 3. Training
```bash
# Single model training
python training/DFormer_seis.py --num_epochs 100

# Ensemble training
python Diformer_final/ensembler.py --is_training True --num_epochs 20
```

### 4. Visualization
```bash
# Visualize seismic attributes
python utils/seisvis.py

# Generate pairplots
python utils/seispair.py

# 3D visualization
python utils/seis3dvis.py
```

## Project Structure

```
├── models/
│   ├── dexined.py          # DexiNed edge detection model
│   ├── diformer.py         # Transformer-based seismic model
│   ├── memfusion.py        # Memory-efficient fusion model
│   └── ensembler.py        # Multi-attribute ensemble model
├── utils/
│   ├── seisvis.py          # Seismic attribute visualization
│   ├── seispair.py         # Attribute pairplot generator
│   ├── seis3dvis.py        # 3D volume visualization
│   ├── attrshap.py         # SHAP importance analysis
│   ├── unpatch.py          # Patch-to-volume converter
│   ├── datafactory.py      # Data processing utilities
│   └── tools.py            # Training utilities
├── training/
│   ├── DFormer_patch_main.py    # Patch-based training
│   ├── DFormer_seis.py          # Seismic-specific training
│   └── train_seismic.py         # General training pipeline
└── Diformer_final/
    └── ensembler.py             # Main ensemble orchestrator
```

## Models

### 1. DexiNed (Dense Extreme Inception Network)
- **Purpose**: Multi-scale edge detection for seismic horizons
- **Input**: `[batch_size, 1, height, width]` - Single-channel seismic data
- **Output**: 7 edge maps at different scales + fused ensemble result
- **Architecture**: Dense blocks with skip connections

### 2. Diformer (Transformer-based Model)
- **Purpose**: Attention-based seismic horizon segmentation
- **Input**: `[batch_size, 1, 16, 288]` - Typical patch dimensions
- **Output**: `[batch_size, 7, 16, 288]` - 7-class horizon predictions
- **Architecture**: Dense CNN blocks + transformer attention mechanisms

### 3. MemFusion (Memory-Efficient Fusion)
- **Purpose**: Efficient multi-model feature fusion
- **Input**: Combined features from multiple models
- **Output**: Final segmentation predictions
- **Architecture**: Lightweight attention with sparse cross-attention

### 4. Ensemble Model
- **Purpose**: Meta-learning across multiple seismic attributes
- **Components**: Individual meta-models + fusion model
- **Attributes**: Seismic, frequency, phase, dip, RMS, coherence
- **Training**: Two-stage training (meta-models → fusion model)

## Utilities

### Visualization Tools
- **seisvis.py**: Generate publication-ready attribute visualizations
- **seispair.py**: Create pairplot matrices for attribute relationships
- **seis3dvis.py**: Interactive 3D volume visualization with Mayavi
- **attrshap.py**: SHAP-based feature importance analysis

### Data Processing
- **unpatch.py**: Convert model predictions back to full volumes
- **datafactory.py**: Data loading and preprocessing utilities

## Usage Examples

### Single Model Inference
```python
from models.diformer import Diformer

model = Diformer(dim=[72, 36, 36, 36], num_heads=2)
model.load_state_dict(torch.load('checkpoint.pth'))
output = model(input_tensor)
```

### Ensemble Inference
```python
# Training ensemble
python Diformer_final/ensembler.py --is_training True \
    --attr_dirs "path/to/seismic,path/to/dip" \
    --num_epochs 20

# Testing ensemble
python Diformer_final/ensembler.py --is_testing True
```

### Visualization
```python
# Attribute visualization
python utils/seisvis.py

# SHAP analysis
python utils/attrshap.py
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- Mayavi (for 3D visualization)
- SHAP (for feature importance)

## Dataset

The framework is designed for the F3 seismic dataset but can be adapted for other seismic volumes. Expected data format:
- **Seismic volume**: 3D numpy arrays `[depth, inline, crossline]`
- **Labels**: Integer arrays with horizon class indices (0-6)
- **Attributes**: Computed seismic attributes in same spatial dimensions

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

## Paper Link
[Seismic Attributes Aided Horizon Interpretation Using an Ensemble Dense Inception Transformer Network](https://doi.org/10.1109/TGRS.2024.3349687) or >[Google Drive](https://drive.google.com/file/d/1HQQIZkRjQfuteo0zcAJ5NT2TxK3KuyfU/view?usp=drive_link)

- **DexiNed**: [Dense Extreme Inception Network for edge detection](https://arxiv.org/abs/2112.02250) ```(The model I modified for horizon picking)```
- **Diformer**: [Transformer-based model with attention mechanisms](https://arxiv.org/abs/2205.12956) ```(I replace the dense blocks with mixer attention)```
- **Ensemble Learning**: [Multi-attribute fusion for improved accuracy](https://ieeexplore.ieee.org/document/9709337) ```(Ensemble base model diformer with a Sparse unet as fusion model)```
- 
## License
This project is licensed under the MIT License - see the LICENSE file for details.




