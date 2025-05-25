# Seismic Data Distribution Analysis Tool
# 
# Quick Start
# 1. Prepare your data:
#    - Load your seismic volume (.npy format)
#    - Only requires the original seismic data file (e.g., F3_seismic.npy)
#
# 2. Run the script:
#    ```bash
#    python utils/seisstat.py
#    ```
#
# Output
# - Distribution plots: KDE visualizations comparing original vs normalized data
# - Statistical metrics: Mean, standard deviation, data ranges
# - Multiple analyses: Full volume and slice-based distribution comparisons
# All outputs saved as .png files in the 'seismic_kde_analysis' directory.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def plot_kde_comparison(original_data, normalized_data, title, save_path=None):
    """
    Plot KDE comparison using two separate subplots for better scale visualization,
    focusing on the main distribution range.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Increase font sizes
    AXIS_FONTSIZE = 14  # Larger font for axis labels
    TITLE_FONTSIZE = 16  # Larger font for subplot titles
    MAIN_TITLE_FONTSIZE = 18  # Larger font for main title
    TICK_FONTSIZE = 12  # Larger font for tick labels
    STATS_FONTSIZE = 12  # Font size for stats text
    
    # Calculate percentiles to focus on main distribution (removing extreme outliers)
    # For original data
    orig_flattened = original_data.flatten()
    orig_lower = np.percentile(orig_flattened, 0.5)
    orig_upper = np.percentile(orig_flattened, 99.5)
    orig_mean = np.mean(orig_flattened)
    orig_std = np.std(orig_flattened)
    
    # For normalized data
    norm_flattened = normalized_data.flatten()
    norm_lower = np.percentile(norm_flattened, 0.5)
    norm_upper = np.percentile(norm_flattened, 99.5)
    norm_mean = np.mean(norm_flattened)
    norm_std = np.std(norm_flattened)
    
    # Plot original data
    sns.kdeplot(data=orig_flattened, ax=ax1, color='blue', fill=True)
    ax1.set_title('Original Data Distribution', fontsize=TITLE_FONTSIZE, pad=10)
    ax1.set_xlabel('Value', fontsize=AXIS_FONTSIZE)
    ax1.set_ylabel('Density', fontsize=AXIS_FONTSIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax1.grid(True, alpha=0.3)
    
    # Set x-limits to focus on main distribution
    x_margin = 0.5 * orig_std  # Add some margin
    ax1.set_xlim([orig_lower - x_margin, orig_upper + x_margin])
    
    # Add stats to original plot
    stats_text = f'μ={orig_mean:.2f}, σ={orig_std:.2f}\n'
    stats_text += f'Range: [{np.min(orig_flattened):.2f}, {np.max(orig_flattened):.2f}]'
    stats_text += f'\nDisplay range: [{orig_lower:.2f}, {orig_upper:.2f}]'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=STATS_FONTSIZE)
    
    # Plot normalized data
    sns.kdeplot(data=norm_flattened, ax=ax2, color='red', fill=True)
    ax2.set_title('Normalized Data Distribution', fontsize=TITLE_FONTSIZE, pad=10)
    ax2.set_xlabel('Value', fontsize=AXIS_FONTSIZE)
    ax2.set_ylabel('Density', fontsize=AXIS_FONTSIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax2.grid(True, alpha=0.3)
    
    # Set x-limits to focus on main distribution
    x_margin = 0.5 * norm_std  # Add some margin
    ax2.set_xlim([norm_lower - x_margin, norm_upper + x_margin])
    
    # Add stats to normalized plot
    stats_text = f'μ={norm_mean:.2f}, σ={norm_std:.2f}\n'
    stats_text += f'Range: [{np.min(norm_flattened):.2f}, {np.max(norm_flattened):.2f}]'
    stats_text += f'\nDisplay range: [{norm_lower:.2f}, {norm_upper:.2f}]'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=STATS_FONTSIZE)
    
    # Improve main title positioning and size
    plt.suptitle(title, fontsize=MAIN_TITLE_FONTSIZE, y=0.98)
    plt.tight_layout()
    
    # Adjust spacing to ensure title is well displayed
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def normalize_data(data, method='standard'):
    """Normalize the data using specified method."""
    if method == 'standard':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")

def analyze_seismic_distributions_with_normalization(data, save_dir, norm_method='standard'):
    """Analyze distribution of original and normalized data."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Normalize full volume
    normalized_data = normalize_data(data, method=norm_method)
    
    # Print basic statistics
    print("\nOriginal Data Statistics:")
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Range: [{np.min(data):.4f}, {np.max(data):.4f}]")
    
    print("\nNormalized Data Statistics:")
    print(f"Mean: {np.mean(normalized_data):.4f}")
    print(f"Std: {np.std(normalized_data):.4f}")
    print(f"Range: [{np.min(normalized_data):.4f}, {np.max(normalized_data):.4f}]")
    
    # Analyze full volume
    print("\nAnalyzing full volume distributions...")
    plot_kde_comparison(
        data, 
        normalized_data,
        f'Full Volume Distribution (Original vs {norm_method.capitalize()} Normalized)',
        os.path.join(save_dir, f'full_volume_kde_comparison_{norm_method}.png')
    )
    
    # Analyze middle slices
    slices = {
        'inline': (data.shape[1]//2, 1, 'Inline'),
        'xline': (data.shape[2]//2, 2, 'Crossline'),
        'time': (data.shape[0]//2, 0, 'Time')
    }
    
    for name, (idx, axis, label) in slices.items():
        print(f"\nAnalyzing {name} slice distributions...")
        slice_data = np.take(data, idx, axis=axis)
        slice_norm = np.take(normalized_data, idx, axis=axis)
        
        plot_kde_comparison(
            slice_data,
            slice_norm,
            f'Distribution of {label} Slice (Original vs {norm_method.capitalize()} Normalized)',
            os.path.join(save_dir, f'{name}_slice_kde_comparison_{norm_method}.png')
        )

# Example usage
if __name__ == "__main__":
    # Load seismic data
    seismic_data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
    seismic_data = seismic_data.reshape(-1, 951, 288)
    seismic_data = seismic_data[::5,::5,::5]
    
    save_dir = 'seismic_kde_analysis'
    
    # Analyze with standard normalization
    analyze_seismic_distributions_with_normalization(seismic_data, save_dir, 'standard')