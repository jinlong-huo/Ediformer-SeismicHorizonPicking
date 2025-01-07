import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    # facies_palette = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
    #    '#1B4F72','#2E86C1', '#AED6F1']
    # palette = sns.husl_palette(s=.4)
    palette = sns.color_palette()
    
    sns.set_style("ticks")
    # fig.set_facecolor('white') 
    pairplot = sns.pairplot(plot_data, 
                           hue=label_column,
                           diag_kind="kde",
                           plot_kws={'alpha': 0.6},
                           diag_kws={'alpha': 0.6},
                           palette=palette)
    
    output_dir = './output'
    figures_dir = os.path.join(output_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    
    # Save pairplot
    pairplot_path = os.path.join(figures_dir, f'Horizon_{file_name}_pairplot.png')
    pairplot.savefig(pairplot_path)
    
    return pairplot


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

def prepare_trace_data(traces_volume, labels, positions, attr_name, normalize=True):
    """
    Prepare data for each trace position with optional normalization
    
    Parameters:
    traces_volume (numpy.ndarray): Selected traces for different time samples
    labels (numpy.ndarray): Labels for each trace
    positions (list): List of (inline, crossline) positions
    attr_name (list): List of attribute names
    normalize (bool): Whether to normalize the data (default: True)
    
    Returns:
    list: List of DataFrames, one for each trace position
    dict: Scaler objects for each attribute (if normalization is applied)
    """
    from sklearn.preprocessing import StandardScaler
    
    df_list = []
    scalers = {}  # Dictionary to store scalers for each attribute
    
    # Initialize scalers for each attribute if normalization is enabled
    if normalize:
        for j, attr in enumerate(attr_name):
            scalers[attr] = StandardScaler()
            # Fit scaler on all data for this attribute
            all_traces = traces_volume[j].reshape(-1, 1)
            scalers[attr].fit(all_traces)
    
    for i, (il, xl) in enumerate(positions):
        trace_dict = {}
        
        for j, attr in enumerate(attr_name):
            trace_data = traces_volume[j][:, i]
            
            if normalize:
                # Reshape for scaler and transform
                trace_data = trace_data.reshape(-1, 1)
                trace_data = scalers[attr].transform(trace_data).ravel()
            
            trace_dict[f'{attr}_Inline_{il}_Crossline_{xl}'] = trace_data
        
        df = pd.DataFrame(trace_dict)
        df['label'] = labels[:, i]
        df_list.append(df)
    
    if normalize:
        return df_list, scalers
    return df_list



class ShapAnalyzer:
    def __init__(self, output_dir: str = 'shap_results'):
        """Initialize SHAP analysis pipeline."""
        self.output_dir = output_dir
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create directory for SHAP results if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def prepare_data(self, df_list: List[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and combine data from multiple DataFrames."""
        X_list, y_list = [], []
        
        for df in df_list:
            feature_cols = [col for col in df.columns if col != 'label']
            X_current = df[feature_cols].copy()
            X_current.columns = [f'feature_{i}' for i in range(len(feature_cols))]
            y_current = df['label']
            
            X_list.append(X_current)
            y_list.append(y_current)
        
        return pd.concat(X_list, axis=0, ignore_index=True), pd.concat(y_list, axis=0, ignore_index=True)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Train Random Forest model and split data."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X_train, y_train, X_test, y_test
    
    def generate_shap_values(self, model: RandomForestClassifier, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate SHAP values using TreeExplainer."""
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X_test)
    
    def plot_class_shap_analysis(self, shap_values: List[np.ndarray], X_test: pd.DataFrame, class_idx: int):
        """Generate and save SHAP plots for a specific class."""
        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[:,:, class_idx], X_test, show=False)
        plt.title(f'SHAP Summary Plot - Class {class_idx}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/summary_beeswarm_class_{class_idx}.png')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[:,:, class_idx], X_test, plot_type='bar', show=False)
        plt.title(f'Feature Importance Plot - Class {class_idx}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance_bar_class_{class_idx}.png')
        plt.close()
         
        # Heat map
        plt.figure(figsize=(15, 10))
        shap.plots.heatmap(shap.Explanation(
            values=shap_values[:,:, class_idx],
            base_values=np.zeros(len(X_test)),
            data=X_test,
            feature_names=X_test.columns
        ), show=False)
        plt.title(f'SHAP Heat Map - Class {class_idx}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/heatmap_class_{class_idx}.png')
        plt.close()
        
        # Waterfall plot
        # Find a representative example for this class
        class_shap = np.abs(shap_values[:,:, class_idx])
        # Select example with highest sum of absolute SHAP values
        example_idx = np.argmax(np.sum(class_shap, axis=1))
        
        # Create Explanation object for the waterfall plot
        explanation = shap.Explanation(
            values=shap_values[:,:, class_idx][example_idx],
            base_values=np.zeros(1),
            data=X_test.iloc[example_idx],
            feature_names=X_test.columns
        )
        
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Class {class_idx} (Example {example_idx})')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/waterfall_plot_class_{class_idx}.png')
        plt.close()
        
        # Decision plot
        plt.figure(figsize=(12, 8))
        shap.decision_plot(0, shap_values[:,:, class_idx], X_test, show=False)
        plt.title(f'SHAP Decision Plot - Class {class_idx}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/decision_plot_class_{class_idx}.png')
        plt.close()
    
    def save_feature_importance(self, shap_values: List[np.ndarray], X: pd.DataFrame, n_classes):
        """Save feature importance analysis to CSV files."""
        
        # Per-class importance
        for i in range(n_classes):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values[i]).mean(axis=1)
            }).sort_values('importance', ascending=False)
            feature_importance.to_csv(f'{self.output_dir}/feature_importance_class_{i}.csv', index=False)
        
        # Global importance
        global_importance = np.zeros(X.shape[1])
        for i in range(n_classes):
            global_importance += np.abs(shap_values[i]).mean(axis=1)
        
        global_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': global_importance / n_classes
        }).sort_values('importance', ascending=False)
        global_importance_df.to_csv(f'{self.output_dir}/global_feature_importance.csv', index=False)
        
        return global_importance_df
    
    def run_analysis(self, df_list: List[pd.DataFrame]) -> Dict:
        """Run complete SHAP analysis pipeline."""
        # Prepare data
        X, y = self.prepare_data(df_list)
        
        # Train model
        model, X_train, y_train, X_test, y_test = self.train_model(X, y)
        
        # Generate SHAP values
        shap_values = self.generate_shap_values(model, X_test)
        
        # Generate plots for each class
        n_classes = len(np.unique(y))
        for i in range(n_classes):
            self.plot_class_shap_analysis(shap_values, X_test, i)
        
        # Save feature importance analysis
        global_importance_df = self.save_feature_importance(shap_values, X, n_classes)
        
        return {
            'model': model,
            'shap_values': shap_values,
            'X_test': X_test,
            'y_test': y_test,
            'feature_importance': global_importance_df
        }


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 100
    attr_name = ['seismic', 'freq', 'dip', 'phase', 'rms']
    
    # Load data
    seismic_volume_1 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_seismic.npy')
    seismic_volume_2 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_freq.npy')
    seismic_volume_3 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_predict_MCDL_crossline.npy')
    seismic_volume_4 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_crop_horizon_phase.npy')
    seismic_volume_5 = np.load('/home/dell/disk1/Jinlong/Horizontal-data/F3_RMSAmp.npy')
    
    seismic_labels = np.load('/home/dell/disk1/Jinlong/Horizontal-data/test_label_no_ohe.npy')
    
    # Reshape data
    seismic_volume_1 = np.squeeze(seismic_volume_1).reshape(-1, 951, 288)
    seismic_volume_2 = seismic_volume_2.reshape(-1, 951, 288)
    seismic_volume_3 = np.swapaxes(seismic_volume_3, -1, 1)
    seismic_volume_4 = seismic_volume_4.reshape(-1, 951, 288)
    seismic_volume_5 = seismic_volume_5.reshape(-1, 951, 288)
    seismic_labels = seismic_labels.reshape(-1, 951, 288)
    
    n_traces = 5
    seed = 42
    seismic_volume = [seismic_volume_1, seismic_volume_2, seismic_volume_3, 
                      seismic_volume_4, seismic_volume_5]
    
    # Select random traces
    selected_traces_volume, selected_labels, positions = select_random_traces(
        seismic_volume, seismic_labels, n_traces, seed
    )
    
    # Prepare data with normalization
    df_list, scalers = prepare_trace_data(
        selected_traces_volume, selected_labels, positions, attr_name, normalize=True
    )
    
    # Create pairplot
    for i, (il, xl) in enumerate(positions):
        file_name = f'Inline_{il}_Crossline_{xl}'
        pairplot = create_visualization(df_list[i], file_name)
    
    # SHAP analysis    
    analyzer = ShapAnalyzer()
    results = analyzer.run_analysis(df_list)  # df_list from your data preparation
    
    # Print top features globally
    print("\nGlobal top 5 important features:")
    print(results['feature_importance'].head())
    
