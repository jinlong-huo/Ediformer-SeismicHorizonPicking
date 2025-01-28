import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.check_cpu_info import get_optimal_cpu_count

"""
 
This code is for displaying label distribution

"""


class ShapAnalyzer:
    def __init__(self, output_dir: str = 'shap_results', attr_name: List[str] = None, batch_size: int = 1000):
        """Initialize SHAP analysis pipeline."""
        self.output_dir = output_dir
        self.attr_name = attr_name
        self.batch_size = batch_size
        self.evaluation_history = []
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_class_shap_analysis(self, shap_values: List[np.ndarray], X_test: pd.DataFrame, class_idx: int):
        """Generate and save essential SHAP plots for a specific class.
        
        Args:
            shap_values: SHAP values for all classes
            X_test: Test data features
            class_idx: Index of the class to analyze
        """
        # Only keep essential plots that provide unique insights
        
        # 1. Feature Importance Bar Plot (Most informative overview)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[:,:, class_idx], X_test, plot_type='bar', show=False)
        plt.title(f'Feature Importance Plot - Class {class_idx}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance_bar_class_{class_idx}.png')
        plt.close()
        
        # 2. Summary Beeswarm Plot (Distribution of SHAP values)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[:,:, class_idx], X_test, show=False)
        plt.title(f'SHAP Summary Plot - Class {class_idx}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/summary_beeswarm_class_{class_idx}.png')
        plt.close()
        
        # 3. Waterfall plot for the most representative example
        class_shap = np.abs(shap_values[:,:, class_idx])
        example_idx = np.argmax(np.sum(class_shap, axis=1))
        
        explanation = shap.Explanation(
            values=shap_values[:,:, class_idx][example_idx],
            base_values=np.zeros(1),
            data=X_test.iloc[example_idx],
            feature_names=X_test.columns
        )
        
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Class {class_idx} (Most Representative Example)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/waterfall_plot_class_{class_idx}.png')
        plt.close()
        
        # Optional: Decision plot with subsampled data for large datasets
        if len(X_test) > 1000:
            # Subsample data for decision plot
            subsample_size = 1000
            indices = np.random.choice(len(X_test), subsample_size, replace=False)
            X_subset = X_test.iloc[indices]
            shap_subset = shap_values[:,:, class_idx][indices]
            
            plt.figure(figsize=(12, 8))
            shap.decision_plot(0, shap_subset, X_subset, show=False, ignore_warnings=True)
            plt.title(f'SHAP Decision Plot - Class {class_idx} (Subsampled Data)')
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
           
    def train_model(self, X: pd.DataFrame, y: pd.Series, model=None) -> Tuple[BaseEstimator, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Train model with optimized parameters
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model: Pre-configured model instance (if None, uses default RandomForest)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model is None:
            # Default model if none provided
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=10,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
        
        # Create a copy of the model to avoid modifying the original
        model = clone(model)
        model.fit(X_train, y_train)
        return model, X_train, y_train, X_test, y_test

    def generate_shap_values(self, model: RandomForestClassifier, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate SHAP values using parallel processing"""
        all_shap_values = []
        n_jobs = os.cpu_count() - 1
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(self._process_batch, X_test.iloc[i:i + self.batch_size], model): i 
                for i in range(0, len(X_test), self.batch_size)
            }
            
            for future in as_completed(futures):
                batch_shap = future.result()
                all_shap_values.append((futures[future], batch_shap))
        
        all_shap_values.sort(key=lambda x: x[0])
        all_shap_values = [x[1] for x in all_shap_values]
        
        return (
            [np.concatenate([batch[i] for batch in all_shap_values]) 
             for i in range(len(all_shap_values[0]))]
            if isinstance(all_shap_values[0], list)
            else np.concatenate(all_shap_values)
        )

    @staticmethod
    def _process_batch(batch_data, model):
        """Process a single batch for SHAP calculation"""
        return shap.TreeExplainer(model).shap_values(batch_data)

    def progressive_evaluation(self, X: pd.DataFrame, y: pd.Series, ranked_attrs: List[str], cv=5) -> Dict:
        """Evaluate attribute combinations progressively"""
        results = []
        current_attrs = []
        base_classifier = DecisionTreeClassifier(max_depth=8, random_state=42)
            
        for attr in ranked_attrs:
            current_attrs.append(attr)
            X_current = X[current_attrs]
            
            cv_scores = cross_val_score(
                base_classifier,
                X_current,
                y,
                cv=cv,
                scoring='f1_macro'
            )
            
            result = {
                'attributes': current_attrs.copy(),
                'n_attributes': len(current_attrs),
                'f1_score_mean': cv_scores.mean(),
                'f1_score_std': cv_scores.std(),
            }
            results.append(result)
            self.evaluation_history.append(result)
            
        gains = self._calculate_gains(results)
        contributions = self._calculate_attribute_contributions(results)
        # self._plot_comprehensive_results(results, contributions, gains)
        
        return {
            'results': results,
            'gains': gains,
            'contributions': contributions,
            'ranked_attrs': ranked_attrs
        }

    def _calculate_gains(self, results: List[Dict]) -> List[float]:
        """Calculate performance gains between consecutive attribute additions"""
        prev_f1 = 0
        return [result['f1_score_mean'] - prev_f1 for result in results]

    def _calculate_attribute_contributions(self, results: List[Dict]) -> List[Dict]:
        """Calculate attribute contributions to overall performance"""
        contributions = []
        prev_f1 = 0
        
        for result in results:
            current_f1 = result['f1_score_mean']
            contributions.append({
                'attribute': result['attributes'][-1],
                'marginal_gain': current_f1 - prev_f1,
                'cumulative_f1': current_f1
            })
            prev_f1 = current_f1
            
        return contributions

    def optimize_feature_set(self, X: pd.DataFrame, y: pd.Series, ranked_attrs: List[str], 
                           gain_threshold: float = 0.05) -> Tuple[List[str], Dict]:
        """Run progressive evaluation and select optimal attributes"""
        eval_results = self.progressive_evaluation(X=X, y=y, ranked_attrs=ranked_attrs)
        selected_attrs = self._select_optimal_attributes(eval_results, gain_threshold)
        
        return selected_attrs, eval_results

    def _select_optimal_attributes(self, eval_results: Dict, gain_threshold: float = 0.05) -> List[str]:
        """Select optimal attributes based on gain threshold"""
        for i, gain in enumerate(eval_results['gains']):
            if gain < gain_threshold:
                return eval_results['ranked_attrs'][:i+1]
        return eval_results['ranked_attrs']

    def run_complete_analysis(self, df_list: List[pd.DataFrame], model=None, gain_threshold: float = 0.05) -> Dict:
        """Run complete SHAP analysis pipeline
        
        Args:
            df_list: List of DataFrames containing the data
            model: Model instance to use for analysis
            gain_threshold: Threshold for feature selection
        """
        X, y = prepare_combined_data(df_list, self.attr_name)
        
        # Optional dimensionality reduction for large feature sets
        if X.shape[1] > 100:
            selector = SelectPercentile(f_classif, percentile=50)
            X = pd.DataFrame(selector.fit_transform(X, y), columns=X.columns[selector.get_support()])
        
        # Initial model and SHAP analysis
        model, _, _, X_test, _ = self.train_model(X, y, model)
        shap_values = self.generate_shap_values(model, X_test)
        
        # Plot SHAP analysis for each class
        n_classes = len(np.unique(y))
        for i in range(n_classes):
            self.plot_class_shap_analysis(shap_values, X_test, i)
            
        # Calculate feature importance and optimize feature set
        n_classes = len(np.unique(y))
        global_importance_df = self.save_feature_importance(shap_values, X, n_classes)
        selected_attrs, eval_results = self.optimize_feature_set(
            X, y, global_importance_df['feature'].tolist(), gain_threshold
        )
        
        # Final analysis with optimized features
        optimized_df_list = [
            df[[col for col in df.columns if any(col.startswith(attr) for attr in selected_attrs) 
                or col == 'label']].copy() 
            for df in df_list
        ]
        
        X_opt, y_opt = prepare_combined_data(optimized_df_list, selected_attrs)
        model_opt, _, _, X_test_opt, y_test_opt = self.train_model(X_opt, y_opt, model)
        shap_values_opt = self.generate_shap_values(model_opt, X_test_opt)
        
        return {
            'initial_results': {
                'results': eval_results['results'],
                'gains': eval_results['gains'],
                'contributions': eval_results['contributions'],
                'model': model,
                'shap_values': shap_values,
                'feature_importance': global_importance_df
            },
            'optimized_results': {
                'selected_attributes': selected_attrs,
                'model': model_opt,
                'shap_values': shap_values_opt,
                'X_test': X_test_opt, 
                'y_test': y_test_opt
            }
        }
        
    def plot_comparative_results(self, all_results: Dict[str, Dict[int, Dict[str, Any]]], tau: float = 0.05):
        """Plot comparative analysis across different models and trace numbers using comprehensive visualization"""
        # Define MATLAB-style colors explicitly as a list
        colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
        trace_styles = ['-', '--', ':', '-.']
        
        # Create separate plots for each model
        for model_idx, (model_name, traces_results) in enumerate(all_results.items()):
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Plot 1: Performance vs Attributes (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            for j, (n_traces, result) in enumerate(traces_results.items()):
                initial_results = result['initial_results']['results']
                n_attrs = [r['n_attributes'] for r in initial_results]
                scores = [r['f1_score_mean'] for r in initial_results]
                std = [r['f1_score_std'] for r in initial_results]
                
                ax1.errorbar(n_attrs, scores, yerr=std, 
                            marker='o', label=f"{n_traces} traces",
                            color=colors[j % len(colors)], linestyle=trace_styles[j % len(trace_styles)],
                            capsize=5)
            
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_xlabel('Number of Attributes')
            ax1.set_ylabel('F1 Score')
            ax1.set_title(f'{model_name}: Performance vs Number of Attributes')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 2: Gain Ratio (top-right)
            ax2 = fig.add_subplot(gs[0, 1])
            for j, (n_traces, result) in enumerate(traces_results.items()):
                gains = result['initial_results']['gains']
                ax2.plot(range(1, len(gains)+1), gains, 
                        marker='o', label=f"{n_traces} traces",
                        color=colors[j % len(colors)], linestyle=trace_styles[j % len(trace_styles)])
            
            ax2.axhline(y=tau, color='#FF0000', linestyle='--', label=f'Threshold ({tau})')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_xlabel('Attribute Addition Step')
            ax2.set_ylabel('Gain Ratio')
            ax2.set_title(f'{model_name}: Gain Ratio per Attribute Addition')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 3: Marginal Gains (bottom-left)
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Get unique attributes for this model
            unique_attrs = sorted(set(
                c['attribute'] 
                for result in traces_results.values()
                for c in result['initial_results']['contributions']
            ))
            
            n_groups = len(unique_attrs)
            n_bars = len(traces_results)
            bar_width = 0.8 / n_bars
            
            for j, (n_traces, result) in enumerate(traces_results.items()):
                contributions = result['initial_results']['contributions']
                marginal_gains = [next((c['marginal_gain'] for c in contributions 
                                    if c['attribute'] == attr), 0) 
                                for attr in unique_attrs]
                
                x = np.arange(len(unique_attrs)) + j * bar_width
                bars = ax3.bar(x, marginal_gains, bar_width,
                            label=f"{n_traces} traces",
                            color=colors[j % len(colors)], alpha=0.8)
                
                # Color bars based on threshold
                for bar, gain in zip(bars, marginal_gains):
                    if gain < tau:
                        bar.set_alpha(0.3)
            
            ax3.axhline(y=tau, color='#FF0000', linestyle='--', label=f'Threshold ({tau})')
            ax3.set_xticks(np.arange(n_groups) + (n_bars-1) * bar_width / 2)
            ax3.set_xticklabels(unique_attrs, rotation=45, ha='right')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.set_xlabel('Attributes')
            ax3.set_ylabel('Marginal Gain')
            ax3.set_title(f'{model_name}: Marginal Gain per Attribute')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Plot 4: Cumulative Performance (bottom-right)
            ax4 = fig.add_subplot(gs[1, 1])
            for j, (n_traces, result) in enumerate(traces_results.items()):
                contributions = result['initial_results']['contributions']
                cumulative_f1 = [c['cumulative_f1'] for c in contributions]
                
                ax4.plot(range(1, len(cumulative_f1) + 1), cumulative_f1,
                        marker='o', label=f"{n_traces} traces",
                        color=colors[j % len(colors)], linestyle=trace_styles[j % len(trace_styles)])
            
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.set_xlabel('Number of Attributes')
            ax4.set_ylabel('Cumulative F1 Score')
            ax4.set_title(f'{model_name}: Cumulative Performance')
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust layout and save
            plt.savefig(f'{self.output_dir}/comparative_analysis_{model_name}.png',
                        dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()
    
def plot_facies_distribution(filepath):
    # Load the facies volume data
    facies_volume = np.load(filepath)

    # Calculate unique classes and their counts
    unique_classes, counts = np.unique(facies_volume, return_counts=True)

    # Total number of samples
    total_samples = facies_volume.size

    # Calculate percentagesf
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
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pairplot

def prepare_seismic_data(data_dir: str, n_traces: int = 50, attr_names: List[str] = None, normalize: bool = True, seed: int = 42) -> Tuple[List[pd.DataFrame], Dict]:
    """
    Comprehensive function to load, preprocess, and prepare seismic data for analysis.
    
    Args:
        data_dir: Directory containing seismic data files
        n_traces: Number of traces to select
        attr_names: List of attribute names to process
        normalize: Whether to normalize the data
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (list of prepared DataFrames, scalers dictionary if normalize=True)
    """
    # Set default attributes if none provided
    if attr_names is None:
        attr_names = ['seismic', 'freq', 'dip', 'phase', 'rms', 'complex', 'coherence', 'azc']
    
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
        elif i == 2:  # dip volume
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
    with Pool(n_jobs) as pool:
        df_list = list(pool.imap(process_single_trace, process_args))
    
    # Add labels
    for i, df in enumerate(df_list):
        df['label'] = selected_labels[:, i]
    
    return df_list, scalers if normalize else (df_list, None)

def process_single_trace(args: Tuple) -> pd.DataFrame:
    """Helper function to process a single trace."""
    position_idx, il, xl, traces_volume, scalers, attr_names = args
    trace_dict = {}
    
    for j, attr in enumerate(attr_names):
        trace_data = traces_volume[j][:, position_idx]
        if scalers is not None:
            trace_data = scalers[attr].transform(trace_data.reshape(-1, 1)).ravel()
        trace_dict[f'{attr}_Inline_{il}_Crossline_{xl}'] = trace_data
        
    return pd.DataFrame(trace_dict)

def prepare_combined_data(df_list: List[pd.DataFrame], attr_names: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """Combine multiple DataFrames into features and labels."""
    features = []
    labels = []
    
    for df in df_list:
        feature_cols = [col for col in df.columns if col != 'label']
        curr_features = df[feature_cols].copy()
        curr_features.columns = attr_names[:len(feature_cols)]
        features.append(curr_features)
        labels.append(df['label'])
    
    return pd.concat(features, axis=0, ignore_index=True), pd.concat(labels, axis=0, ignore_index=True)

def prepare_all_datasets(config: Dict):
    """Prepare datasets for all trace counts"""
    datasets = {}
    data_params = config['data_params']
    
    for n_traces in config['analysis_params']['n_traces']:
        print(f"Processing dataset with {n_traces} traces...")
        df_list, scalers = prepare_seismic_data(
            data_dir=data_params['data_dir'],
            n_traces=n_traces,
            attr_names=data_params['attr_names'],
            normalize=data_params['normalize'],
            seed=data_params['seed']
        )
        datasets[n_traces] = (df_list, scalers)
    
    return datasets

def main():
    # Set random seed for reproducibility
    config = {
    'data_params': {
        'data_dir': '/home/dell/disk1/Jinlong/Horizontal-data',
        'attr_names': ['seismic', 'freq', 'dip', 'phase', 'rms', 'complex', 'coherence', 'azc'],
        'normalize': True,
        'seed': 42
    },
    'analysis_params': {
        'n_traces': [10, 100, 500, 2000],
        'models': {
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'DecisionTree': DecisionTreeClassifier(max_depth=10),
            'GradientBoosting': HistGradientBoostingClassifier()
        }
    }
}
    
    # Prepare data
    print("Processing seismic data...")
    # df_list, scalers = prepare_seismic_data(config)
    all_datasets = prepare_all_datasets(config)
    print("Data preparation complete!")
    
    # Run SHAP analysis
    print("\nRunning SHAP analysis...")
    analyzer = ShapAnalyzer(
    output_dir='shap_results',
    attr_name=config['data_params']['attr_names'],
    batch_size=5000  # Adjust based on your system's memory
)
    # shap_results = analyzer.run_complete_analysis(df_list, gain_threshold=0.05) 
    all_results = {}
    for model_name, model in config['analysis_params']['models'].items():
        model_results = {}
        for n_traces in config['analysis_params']['n_traces']:
            df_list, _ = all_datasets[n_traces]
            results = analyzer.run_complete_analysis(
                df_list=df_list,
                model=model,
                gain_threshold=0.05
            )
            model_results[n_traces] = results
        all_results[model_name] = model_results
    
    # Plot comparative results
    analyzer.plot_comparative_results(all_results)
    # shap_results = analyzer.run_comparative_analysis(data_dir='shap_results', configurations=config)
    print("Analysis complete!")
    
    # Create pairplot
    # for i, (il, xl) in enumerate(positions):
    #     file_name = f'Inline_{il}_Crossline_{xl}'
    #     pairplot = create_visualization(df_list[i], file_name)
    
    return all_results


if __name__ == "__main__":
    shap_results = main()

