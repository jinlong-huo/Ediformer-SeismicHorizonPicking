import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def plot_grouped_distributions(data_dict, title, is_normalized=False, save_path=None):
    """
    Plot multiple distributions on the same axis.
    data_dict: Dictionary with keys as names and values as data arrays
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'purple']
    for (name, data), color in zip(data_dict.items(), colors):
        sns.kdeplot(data=data.flatten(), ax=ax, color=color, fill=True, alpha=0.2, label=name)
    
    ax.set_title(title, pad=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add stats to plot
    stats_text = ""
    for name, data in data_dict.items():
        stats_text += f'{name}: μ={np.mean(data):.2f}, σ={np.std(data):.2f}\n'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
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
    
    # Get middle slices
    slices = {
        'inline': (data.shape[1]//2, 1, 'Inline'),
        'xline': (data.shape[2]//2, 2, 'Crossline'),
        'time': (data.shape[0]//2, 0, 'Time')
    }
    
    # Prepare data dictionaries
    original_data_dict = {'Full': data}
    normalized_data_dict = {'Full': normalize_data(data, method=norm_method)}
    
    # Add slices to dictionaries
    for name, (idx, axis, label) in slices.items():
        slice_data = np.take(data, idx, axis=axis)
        original_data_dict[label] = slice_data
        normalized_data_dict[label] = normalize_data(slice_data, method=norm_method)
    
    # Print basic statistics
    print("\nOriginal Data Statistics:")
    for name, d in original_data_dict.items():
        print(f"\n{name}:")
        print(f"Mean: {np.mean(d):.4f}")
        print(f"Std: {np.std(d):.4f}")
        print(f"Range: [{np.min(d):.4f}, {np.max(d):.4f}]")
    
    print("\nNormalized Data Statistics:")
    for name, d in normalized_data_dict.items():
        print(f"\n{name}:")
        print(f"Mean: {np.mean(d):.4f}")
        print(f"Std: {np.std(d):.4f}")
        print(f"Range: [{np.min(d):.4f}, {np.max(d):.4f}]")
    
    # Plot original distributions
    plot_grouped_distributions(
        original_data_dict,
        'Original Data Distributions',
        is_normalized=False,
        save_path=os.path.join(save_dir, f'original_distributions_{norm_method}.png')
    )
    
    # Plot normalized distributions
    plot_grouped_distributions(
        normalized_data_dict,
        f'{norm_method.capitalize()} Normalized Data Distributions',
        is_normalized=True,
        save_path=os.path.join(save_dir, f'normalized_distributions_{norm_method}.png')
    )

# Example usage
if __name__ == "__main__":
    # Load seismic data
    seismic_data = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
    seismic_data = seismic_data.reshape(-1, 951, 288)
    seismic_data = seismic_data[::10,::10,::10]
    
    save_dir = 'seismic_kde_analysis'
    
    # Analyze with standard normalization
    analyze_seismic_distributions_with_normalization(seismic_data, save_dir, 'standard')