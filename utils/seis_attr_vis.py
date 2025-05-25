# Seismic Attribute Visualization Creates publication-ready visualizations of seismic attributes from the F3 dataset.

# Seismic Attribute Visualization Tool
# 
# Quick Start
# 1. Prepare your data:
#    - Place all `.npy` files in one folder
#    - Required files: `F3_seismic.npy`, `F3_crop_horizon_freq.npy`, `F3_predict_MCDL_crossline.npy`, 
#      `F3_crop_horizon_phase.npy`, `F3_RMSAmp.npy`, `F3_coherence.npy`
#
# 2. Run the script:
#    ```bash
#    python utils/seisvis.py
#    ```
#
# Output
# - Labeled figures: With titles and colorbars
# - Clean figures: No labels (for presentations)
# - Combined overview: All attributes in one image
# All outputs saved as `.png` files in the specified directory.


import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_seismic_attributes_enhanced(data_dir, time_slice=None, mode='timeslice', save_dir='attribute_plots_enhanced'):
    """
    Create enhanced seismic attribute visualizations using time slices with geophysically 
    appropriate colormaps for better geological feature visualization.
    
    Args:
        data_dir (str): Directory containing seismic data files
        time_slice (int, optional): Time sample for horizontal slice (0-287 for F3)
        mode (str): 'timeslice', 'inline', or 'crossline' 
        save_dir (str): Directory to save the plots
    """
    
    # Enhanced font settings for better visibility
    plt.rcParams['font.family'] = 'Arial'  # Clean, professional font
    plt.rcParams['font.weight'] = 'bold'   # Bold text for better visibility
    
    # Much larger font sizes for PowerPoint presentation
    TITLE_FONTSIZE = 28
    TICK_FONTSIZE = 24
    COLORBAR_FONTSIZE = 22
    LABEL_FONTSIZE = 26
    
    # List of attributes to visualize
    attr_names = ['seismic', 'freq', 'dip', 'phase', 'rms', 'coherence']
    
    # Descriptive labels for each attribute
    attr_labels = {
        'seismic': 'Seismic Amplitude',
        'freq': 'Instantaneous Frequency',
        'dip': 'Dip Attribute',
        'phase': 'Instantaneous Phase',
        'rms': 'RMS Amplitude',
        'coherence': 'Coherence Attribute'
    }
    
    # File paths dictionary for each attribute
    file_paths = {
        'seismic': 'F3_seismic.npy',
        'freq': 'F3_crop_horizon_freq.npy',
        'dip': 'F3_predict_MCDL_crossline.npy',
        'phase': 'F3_crop_horizon_phase.npy',
        'rms': 'F3_RMSAmp.npy',
        'coherence': 'F3_coherence.npy'
    }
    
    # Geophysically appropriate colormaps - commonly used in industry
    colormaps = {
        'seismic': 'seismic',            # Standard red-blue for seismic amplitude
        'freq': 'jet',                  # Rainbow colormap - classic for frequency
        'dip': 'turbo',                 # Modern rainbow replacement for directional data
        'phase': 'twilight',            # Cyclic colormap perfect for phase (-π to π)
        'rms': 'hot',                   # Black-red-yellow-white for energy
        'coherence': 'gray'             # Standard grayscale for coherence (black=discontinuous)
    }
    
    # Colormap value ranges for better visualization
    vmin_vmax = {
        'seismic': 'auto',              # Use data range
        'freq': 'auto',
        'dip': 'auto', 
        'phase': (-np.pi, np.pi),       # Phase always -π to π
        'rms': 'auto',
        'coherence': (0, 1)             # Coherence always 0 to 1
    }
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load volumes
    volumes = {}
    for attr in attr_names:
        try:
            data = np.load(os.path.join(data_dir, file_paths[attr]))
            
            # Handle different data shapes and orientations
            if attr == 'seismic':
                data = np.squeeze(data).reshape(-1, 951, 288)
            elif attr == 'dip':
                data = np.swapaxes(data, -1, 1)
            else:
                data = data.reshape(-1, 951, 288)
                
            # Trim to consistent size (first 600 inlines)
            data = data[:600, :, :]
            volumes[attr] = data
            print(f"✓ Loaded {attr} with shape {data.shape}")
        except Exception as e:
            print(f"✗ Could not load {attr}: {e}")
    
    # Get data dimensions
    n_inline, n_crossline, n_samples = next(iter(volumes.values())).shape
    print(f"\nData dimensions: {n_inline} inlines × {n_crossline} crosslines × {n_samples} time samples")
    
    # Select time slice (middle portion usually has good geology)
    if time_slice is None:
        if mode == 'timeslice':
            time_slice = n_samples // 2  # Middle time slice (~850ms for F3)
        elif mode == 'inline':
            time_slice = np.random.randint(50, n_inline-50)
        else:  # crossline
            time_slice = np.random.randint(50, n_crossline-50)
    
    # Convert time sample to milliseconds for F3 (4ms sampling)
    time_ms = time_slice * 4 if mode == 'timeslice' else None
    
    if mode == 'timeslice':
        print(f"Selected time slice at sample {time_slice} (~{time_ms}ms)\n")
    else:
        print(f"Selected {mode} at position {time_slice}\n")
    
    # Create enhanced plots for each attribute
    for attr, volume in volumes.items():
        
        # ===== MAIN FIGURE WITH COLORBAR =====
        fig, ax = plt.subplots(figsize=(16, 10))  # Larger figure size
        
        # Extract the appropriate slice
        if mode == 'timeslice':
            slice_data = volume[:, :, time_slice]  # Horizontal slice at constant time
            extent = [300, 1250, 100, 600]  # [left, right, bottom, top] - crossline vs inline
            ax.set_xlabel('Crossline', fontsize=LABEL_FONTSIZE, fontweight='bold')
            ax.set_ylabel('Inline', fontsize=LABEL_FONTSIZE, fontweight='bold')
            slice_title = f"Time Slice at {time_ms}ms"
        elif mode == 'inline':
            slice_data = volume[time_slice, :, :]
            extent = [300, 1250, 1700, 0]  # [left, right, bottom, top] for F3 coordinates
            ax.set_xlabel('Crossline', fontsize=LABEL_FONTSIZE, fontweight='bold')
            ax.set_ylabel('Time (ms)', fontsize=LABEL_FONTSIZE, fontweight='bold')
            slice_title = f"Inline {time_slice}"
        else:  # crossline
            slice_data = volume[:, time_slice, :]
            extent = [100, 600, 1700, 0]  # [left, right, bottom, top] for F3 coordinates  
            ax.set_xlabel('Inline', fontsize=LABEL_FONTSIZE, fontweight='bold')
            ax.set_ylabel('Time (ms)', fontsize=LABEL_FONTSIZE, fontweight='bold')
            slice_title = f"Crossline {time_slice}"
        
        # Set colormap value range
        if vmin_vmax[attr] == 'auto':
            vmin, vmax = np.percentile(slice_data, [2, 98])  # Robust range (avoid outliers)
        else:
            vmin, vmax = vmin_vmax[attr]
        
        # Create the image with proper colormap and range
        im = ax.imshow(slice_data.T if mode != 'timeslice' else slice_data, 
                      aspect='auto', cmap=colormaps[attr], extent=extent,
                      vmin=vmin, vmax=vmax, origin='upper' if mode == 'timeslice' else 'upper')
        
        # Enhanced tick formatting
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, width=2, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE-2, width=1, length=4)
        
        # Add title
        title = f"{attr_labels[attr]} - {slice_title}"
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=20)
        
        # Enhanced colorbar with proper labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
        cbar.ax.tick_params(labelsize=COLORBAR_FONTSIZE, width=2, length=6)
        
        # Attribute-specific colorbar labels
        colorbar_labels = {
            'phase': 'Phase (radians)',
            'freq': 'Frequency (Hz)',
            'rms': 'RMS Amplitude',
            'dip': 'Dip Angle (°)',
            'coherence': 'Coherence',
            'seismic': 'Amplitude'
        }
        
        cbar.set_label(colorbar_labels[attr], fontsize=COLORBAR_FONTSIZE, 
                      fontweight='bold', rotation=270, labelpad=30)
        
        # Special tick formatting for phase
        if attr == 'phase':
            cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        # Improve layout
        plt.tight_layout()
        
        # Save high-resolution figure  
        save_path = os.path.join(save_dir, f"{attr}_{mode}_{time_slice}_enhanced.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Saved enhanced plot: {save_path}")
        plt.close()
        
        # ===== CLEAN VERSION WITHOUT LABELS (for PPT assembly) =====
        fig_clean, ax_clean = plt.subplots(figsize=(14, 8))
        
        # Same image without labels
        im_clean = ax_clean.imshow(slice_data.T if mode != 'timeslice' else slice_data, 
                                 aspect='auto', cmap=colormaps[attr], extent=extent,
                                 vmin=vmin, vmax=vmax, origin='upper' if mode == 'timeslice' else 'upper')
        
        # Minimal styling - just the image
        ax_clean.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, width=2)
        ax_clean.set_xlabel('', fontsize=0)  # Remove labels for clean version
        ax_clean.set_ylabel('', fontsize=0)
        
        plt.tight_layout()
        
        # Save clean version
        clean_path = os.path.join(save_dir, f"{attr}_{mode}_{time_slice}_clean.png")
        plt.savefig(clean_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Saved clean version: {clean_path}")
        plt.close()
    
    # ===== CREATE A COMBINED OVERVIEW FIGURE =====
    create_combined_overview(volumes, mode, time_slice, time_ms, save_dir, colormaps, attr_labels, vmin_vmax)
    
    # Reset matplotlib settings
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"\n🎉 All enhanced attribute plots saved to: {save_dir}")
    print("Files created:")
    print("  - *_enhanced.png: Full plots with labels and colorbars")
    print("  - *_clean.png: Clean versions for PowerPoint assembly")
    print("  - combined_overview.png: All attributes in one figure")

def create_combined_overview(volumes, mode, time_slice, time_ms, save_dir, colormaps, attr_labels, vmin_vmax):
    """Create a combined overview figure with all attributes"""
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, (attr, volume) in enumerate(volumes.items()):
        # Extract appropriate slice
        if mode == 'timeslice':
            slice_data = volume[:, :, time_slice]
            extent = [300, 1250, 100, 600]
            slice_title = f"Time Slice at {time_ms}ms"
        elif mode == 'inline':
            slice_data = volume[time_slice, :, :]
            extent = [300, 1250, 1700, 0]
            slice_title = f"Inline {time_slice}"
        else:
            slice_data = volume[:, time_slice, :]
            extent = [100, 600, 1700, 0]
            slice_title = f"Crossline {time_slice}"
        
        # Set value range
        if vmin_vmax[attr] == 'auto':
            vmin, vmax = np.percentile(slice_data, [2, 98])
        else:
            vmin, vmax = vmin_vmax[attr]
        
        # Plot
        im = axes[i].imshow(slice_data.T if mode != 'timeslice' else slice_data, 
                           aspect='auto', cmap=colormaps[attr], extent=extent,
                           vmin=vmin, vmax=vmax, origin='upper' if mode == 'timeslice' else 'upper')
        
        axes[i].set_title(f"({chr(97+i)}) {attr_labels[attr]}", fontsize=20, fontweight='bold')
        axes[i].tick_params(labelsize=14)
        
        # Add colorbar to each subplot
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.ax.tick_params(labelsize=12)
        
        # Special formatting for phase
        if attr == 'phase':
            cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    plt.suptitle(f'Seismic Multi-Attribute Analysis - {slice_title}', 
                 fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    combined_path = os.path.join(save_dir, f"combined_overview_{mode}_{time_slice}.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved combined overview: {combined_path}")
    plt.close()

# Example usage with time slices and enhanced geophysical colormaps:
if __name__ == "__main__":
    # Time slice visualization (recommended for most attributes)
    visualize_seismic_attributes_enhanced(
        data_dir='/home/dell/disk1/Jinlong/Horizontal-data',
        time_slice=200,      # ~800ms time slice, or None for automatic middle slice
        mode='timeslice',    # 'timeslice' shows geological features best
        save_dir='enhanced_timeslice_plots'
    )
    
    # You can also create vertical slices if needed:
    # visualize_seismic_attributes_enhanced(
    #     data_dir='/home/dell/disk1/Jinlong/Horizontal-data', 
    #     time_slice=150,
    #     mode='inline',
    #     save_dir='enhanced_inline_plots'
    # )