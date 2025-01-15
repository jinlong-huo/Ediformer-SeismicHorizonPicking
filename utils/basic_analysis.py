import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def plot_kde_comparison(original_data, normalized_data, title, save_path=None):
    """
    Plot KDE comparison using two separate subplots for better scale visualization.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original data
    sns.kdeplot(data=original_data.flatten(), ax=ax1, color='blue', fill=True)
    ax1.set_title('Original Data Distribution', pad=10)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    
    # Add stats to original plot
    stats_text = f'μ={np.mean(original_data):.2f}, σ={np.std(original_data):.2f}\n'
    stats_text += f'Range: [{np.min(original_data):.2f}, {np.max(original_data):.2f}]'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    # Plot normalized data
    sns.kdeplot(data=normalized_data.flatten(), ax=ax2, color='red', fill=True)
    ax2.set_title('Normalized Data Distribution', pad=10)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.grid(True, alpha=0.3)
    
    # Add stats to normalized plot
    stats_text = f'μ={np.mean(normalized_data):.2f}, σ={np.std(normalized_data):.2f}\n'
    stats_text += f'Range: [{np.min(normalized_data):.2f}, {np.max(normalized_data):.2f}]'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.suptitle(title, fontsize=14, y=1.02)
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