# Seismic Attribute Pairplot Generator
# 
# Quick Start
# 1. Prepare your data:
#    - Place .npy files in one folder
#    - Required files: F3_seismic.npy, F3_crop_horizon_freq.npy, F3_predict_MCDL_crossline.npy, 
#      F3_crop_horizon_phase.npy, F3_RMSAmp.npy, test_label_no_ohe.npy
#
# 2. Run the script:
#    ```bash
#    python utils/seispair.py
#    ```
#    Or modify config parameters in main() function
#
# Output
# - High-quality pairplot visualizations showing relationships between seismic attributes
# - One pairplot per selected trace location
# - Multi-class visualization with color-coded horizon labels
# All outputs saved as .png files in './output/figures/' directory.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import multiprocessing as mp

def create_visualization(data, file_name):
    """
    Create pairplot with larger font sizes and no axis labels
    
    Parameters:
    data (pandas.DataFrame): Input dataframe
    file_name (str): Base name for the output file
    """
    
    # Set larger font sizes globally
    plt.rcParams.update({
        'font.size': 20,             # Base font size
        'axes.titlesize': 20,        # Plot title font size
        'axes.labelsize': 0,         # Set axis label size to 0 to hide labels
        'xtick.labelsize': 20,       # x-axis tick label size
        'ytick.labelsize': 20,       # y-axis tick label size
        'legend.fontsize': 20,       # Legend font size
        'figure.titlesize': 20       # Figure title size
    })
    
    plot_data = data
    label_column = 'label'
    palette = sns.color_palette()
    
    # Set style
    sns.set_style("ticks")
    
    # Create pairplot with larger size
    pairplot = sns.pairplot(plot_data, 
                           hue=label_column,
                           diag_kind="kde",
                           plot_kws={'alpha': 0.6, 's': 30},  # Increased marker size
                           diag_kws={'alpha': 0.6},
                           palette=palette,
                           height=3.5,  # Larger subplot size
                           aspect=1.2)  # Wider aspect ratio
    
    # Remove all axis labels but keep tick values
    for ax in pairplot.axes.flatten():
        # Increase tick label sizes
        ax.tick_params(labelsize=20)
        # Remove axis labels (set to empty string)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Adjust legend font size - use proper syntax
    pairplot._legend.set_title('Label')
    title_text = pairplot._legend.get_title()
    title_text.set_fontsize(20)  # Correct way to set legend title font size
    
    # Set legend text font size
    for text in pairplot._legend.texts:
        text.set_fontsize(20)
    
    # Create output directory
    output_dir = './output'
    figures_dir = os.path.join(output_dir, 'figures')
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    
    # Save pairplot with higher DPI for better quality
    pairplot_path = os.path.join(figures_dir, f'Horizon_{file_name}_pairplot.png')
    pairplot.savefig(pairplot_path, dpi=500, bbox_inches='tight')
    print(f"Saved pairplot to {pairplot_path}")
    plt.close()
    
    return pairplot
def process_single_trace(args: Tuple) -> pd.DataFrame:
    """Helper function to process a single trace."""
    position_idx, il, xl, traces_volume, scalers, attr_names = args
    trace_dict = {}
    
    for j, attr in enumerate(attr_names):
        trace_data = traces_volume[j][:, position_idx]
        if scalers is not None:
            trace_data = scalers[attr].transform(trace_data.reshape(-1, 1)).ravel()
        trace_dict[f'{attr}'] = trace_data
        
    return pd.DataFrame(trace_dict)

def get_optimal_cpu_count(max_percent=80):
    """Get optimal number of CPU cores to use"""
    n_cpus = mp.cpu_count()
    return max(1, int(n_cpus * max_percent / 100))

def prepare_seismic_data(data_dir: str, n_traces: int = 5, attr_names: List[str] = None, normalize: bool = True, seed: int = 42) -> Tuple[List[pd.DataFrame], List[Tuple[int, int]]]:
    """
    Load, preprocess, and prepare seismic data for pairplot visualization.
    
    Args:
        data_dir: Directory containing seismic data files
        n_traces: Number of traces to select
        attr_names: List of attribute names to process
        normalize: Whether to normalize the data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of prepared DataFrames, list of positions)
    """
    # Set default attributes if none provided
    if attr_names is None:
        attr_names = ['seismic', 'freq', 'dip', 'phase', 'rms']
    
    # Load data files
    file_paths = {
        name: f'F3_{name}.npy' if name != 'labels' else 'test_label_no_ohe.npy'
        for name in attr_names + ['labels']
    }
    file_paths.update({
        'freq': 'F3_crop_horizon_freq.npy',
        'dip': 'F3_predict_MCDL_crossline.npy',
        'phase': 'F3_crop_horizon_phase.npy',
        'rms': 'F3_RMSAmp.npy',
        'complex': 'F3_complex_trace.npy',
        'azc': 'F3_Average_zero_crossing.npy'
    })
    
    # Load volumes
    volumes = []
    for attr in attr_names:
        data = np.load(os.path.join(data_dir, file_paths[attr]))
        volumes.append(data)
    labels = np.load(os.path.join(data_dir, file_paths['labels']))
    
    # Preprocess volumes
    processed_volumes = []
    for i, volume in enumerate(volumes):
        if i == 0:  # seismic volume
            volume = np.squeeze(volume).reshape(-1, 951, 288)
        elif attr_names[i] == 'dip':  # dip volume
            volume = np.swapaxes(volume, -1, 1)
        else:
            volume = volume.reshape(-1, 951, 288)
        volume = volume[:600, :, :]
        processed_volumes.append(volume)
    
    processed_labels = labels.reshape(-1, 951, 288)[:600, :, :]
    
    # Select random traces
    np.random.seed(seed)
    n_inline, n_crossline, n_samples = processed_volumes[0].shape
    flat_indices = np.random.choice(n_inline * n_crossline, n_traces, replace=False)
    inline_pos = flat_indices // n_crossline
    crossline_pos = flat_indices % n_crossline
    
    # Extract traces
    selected_traces = []
    for volume in processed_volumes:
        traces = np.array([volume[il, xl, :] for il, xl in zip(inline_pos, crossline_pos)]).T
        selected_traces.append(traces)
    
    selected_labels = np.array([processed_labels[il, xl, :] 
                              for il, xl in zip(inline_pos, crossline_pos)]).T
    positions = list(zip(inline_pos, crossline_pos))
    
    # Prepare data with parallel processing
    n_jobs = get_optimal_cpu_count(max_percent=80)
    
    # Initialize scalers if needed
    scalers = None
    if normalize:
        scalers = {
            attr: StandardScaler().fit(traces.reshape(-1, 1))
            for attr, traces in zip(attr_names, selected_traces)
        }
    
    # Prepare parallel processing arguments
    process_args = [
        (i, il, xl, selected_traces, scalers, attr_names)
        for i, (il, xl) in enumerate(positions)
    ]
    
    # Process traces in parallel
    with mp.Pool(n_jobs) as pool:
        df_list = list(pool.imap(process_single_trace, process_args))
    
    # Add labels
    for i, df in enumerate(df_list):
        df['label'] = selected_labels[:, i]
    
    return df_list, positions

def main():
    """Main function to run pairplot visualization"""
    
    # Configuration
    config = {
        'data_dir': '/home/dell/disk1/Jinlong/Horizontal-data',
        'attr_names': ['seismic', 'freq', 'dip', 'phase', 'rms'],
        'n_traces': 30,  # Adjust as needed
        'normalize': True,
        'seed': 42
    }
    
    # Prepare data
    print(f"Loading seismic data for {config['n_traces']} traces...")
    df_list, positions = prepare_seismic_data(
        data_dir=config['data_dir'],
        n_traces=config['n_traces'],
        attr_names=config['attr_names'],
        normalize=config['normalize'],
        seed=config['seed']
    )
    print(f"Data preparation complete! Loaded {len(df_list)} traces.")
    
    # Create pairplots for each trace
    print("Creating pairplots...")
    for i, (il, xl) in enumerate(positions):
        print(f"Processing trace {i+1}/{len(positions)}: Inline_{il}_Crossline_{xl}")
        file_name = f'Inline_{il}_Crossline_{xl}'
        create_visualization(df_list[i], file_name)
    
    print("All pairplots created successfully!")

if __name__ == "__main__":
    main()