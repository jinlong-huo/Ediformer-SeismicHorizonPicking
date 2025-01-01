import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

"""

This code is for displaying label distribution

"""

def plot_facies_distribution(filepath):
    # Load the facies volume data
    facies_volume = np.load(filepath)

    # Calculate unique classes and their counts
    unique_classes, counts = np.unique(facies_volume, return_counts=True)

    # Total number of samples
    total_samples = facies_volume.size

    # Calculate percentages
    percentages = (counts / total_samples) * 100

    # Create a bar plot of the distribution
    plt.figure(figsize=(12, 6))
    plt.bar(unique_classes, percentages)
    plt.title('Distribution of Facies Classes')
    plt.xlabel('Facies Class')
    plt.ylabel('Percentage (%)')

    # Add percentage labels on top of each bar
    for i, v in enumerate(percentages):
        plt.text(unique_classes[i], v + 0.5, f'{v:.1f}%', ha='center')

    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Print the distribution details
    print("Facies Class Distribution:")
    for class_id, count, percentage in zip(unique_classes, counts, percentages):
        print(f"Class {class_id}: {count:,} samples ({percentage:.1f}%)")

    # Save the plot as an image file
    plt.savefig('facies_cls_distribution.png')

    # Display the plot
    plt.show()



def select_random_traces(seismic_volume, label_volume, n_traces=5, seed=42):
    """
    Randomly select traces from a 3D seismic volume
    
    Parameters:
    seismic_volume (numpy.ndarray): list of 3D array with shape (time/depth, inline, crossline)
    label_volume (numpy.ndarray): 3D array with shape (time/depth, inline, crossline)
    n_traces (int): Number of traces to select
    seed (int): Random seed for reproducibility
    
    Returns:
    tuple: Selected traces list, labels, and their positions (inline, crossline)
    """
    np.random.seed(seed)
    selected_traces_volume = []
    # Get volume dimensions
    n_inline, n_crossline, n_samples = seismic_volume[0].shape
    
    # Generate random positions
    total_traces = n_inline * n_crossline
    flat_indices = np.random.choice(total_traces, n_traces, replace=False)
    
    # Convert to inline/crossline positions
    inline_pos = flat_indices // n_crossline
    crossline_pos = flat_indices % n_crossline
    
    # Extract traces
    for i in range(len(seismic_volume)):
        selected_traces = np.array([
            seismic_volume[i][il, xl, :] 
            for il, xl in zip(inline_pos, crossline_pos)
        ]).T  # Shape: (n_samples, n_traces)
        
        selected_traces_volume.append(selected_traces)
        
    selected_label_traces = np.array([
        label_volume[il, xl, :] 
        for il, xl in zip(inline_pos, crossline_pos)
    ]).T 
    
    return selected_traces_volume, selected_label_traces, list(zip(inline_pos, crossline_pos))


def prepare_trace_data(traces_volume, labels, positions, attr_name):
    """
    Prepare data for each trace position
    
    Parameters:
    traces_volume (numpy.ndarray): Selected traces for different time samples
    labels (numpy.ndarray): Labels for each trace
    positions (list): List of (inline, crossline) positions
    
    Returns:
    list: List of DataFrames, one for each trace position
    """
    df_list = []
    
    
    for i, (il, xl) in enumerate(positions):
        
        trace_dict = {
            f'{attr_name[j]}_Inline_{il}_Crossline_{xl}':  traces_volume[j][:, i] 
            for j in range(len(traces_volume))
        }
        
        df = pd.DataFrame(trace_dict)
        df['label'] = labels[:, i]  
        
        df_list.append(df)
    
    return df_list


def create_visualization(data, file_name):
    """
    Create pairplot and correlation matrix for given features and label
    
    Parameters:
    data (pandas.DataFrame): Input dataframe
    features (list): List of feature column names
    label_column (str): Name of the label column
    """
    
    plot_data = data
    label_column = 'label'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.set_style("whitegrid")
    pairplot = sns.pairplot(plot_data, 
                           hue=label_column,
                           diag_kind="kde",
                           plot_kws={'alpha': 0.6},
                           diag_kws={'alpha': 0.6})
    
    correlation_matrix = plot_data.corr()
    
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                center=0,
                ax=ax2)
    
    ax2.set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    output_dir = './output'
    figures_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    
    filepath = os.path.join(figures_dir, f'Horizon_{file_name}_correlation_matrix.png')
    plt.savefig(filepath)
    
    return pairplot, fig


if __name__ == "__main__":
    

    np.random.seed(42)
    n_samples = 100
    attr_name = ['seismic', 'freq', 'dip', 'phase', 'rms']
    
    seismic_volume_1 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
    seismic_volume_2 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy')
    seismic_volume_3 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_predict_MCDL_crossline.npy')
    seismic_volume_4 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_phase.npy')
    seismic_volume_5 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_RMSAmp.npy')
    seismic_labels = np.load('/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy')
    
    # Make sure all the data are in 3D views mcdl has 600 not 601 inline slices
    seismic_volume_1 = np.squeeze(seismic_volume_1).reshape(-1, 951, 288)
    seismic_volume_2 = seismic_volume_2.reshape(-1, 951, 288)
    seismic_volume_3 = np.swapaxes(seismic_volume_3, -1, 1)
    seismic_volume_4 = seismic_volume_4.reshape(-1, 951, 288)
    seismic_volume_5 = seismic_volume_5.reshape(-1, 951, 288)
    seismic_labels = seismic_labels.reshape(-1, 951, 288)
    
    n_traces = 5
    seed = 42
    seismic_volume = [seismic_volume_1,seismic_volume_2, seismic_volume_3, seismic_volume_4, seismic_volume_5]
    selected_traces_volume, selected_labels, positions = select_random_traces(seismic_volume, seismic_labels, n_traces, seed)
    
    df_list = prepare_trace_data(selected_traces_volume, selected_labels, positions, attr_name)
    
    for i, (il, xl) in enumerate(positions):
        file_name = f'Inline_{il}_Crossline_{xl}'
        pairplot, correlation_fig = create_visualization(df_list[i], file_name)
     
        
    # plot_facies_distribution(seismic_labels)