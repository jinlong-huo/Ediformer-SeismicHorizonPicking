import datetime
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from memory_profiler import profile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import warnings
from contextlib import contextmanager

from models.UNet import UNetClassifier
from utils.check_cpu_info import get_optimal_cpu_count, monitor_cpu_usage

"""

This code is for displaying label distribution

"""

@dataclass
class AttributeContribution:
    attribute: str
    marginal_gain: float
    cumulative_f1: float
    relative_importance: float

class ResourceMonitor:
    """Enhanced monitor for computational resources including GPU during execution"""
    
    def __init__(self, log_warnings: bool = True):
        self.start_time = None
        self.start_cpu_memory = None
        self.start_gpu_memory = None
        self.log_warnings = log_warnings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.using_gpu = torch.cuda.is_available()
        self.monitoring = False
        
        # Initialize monitoring history
        self.history = {
            'timestamps': [],
            'cpu_memory': [],
            'gpu_memory_allocated': [],
            'gpu_memory_cached': [],
            'gpu_utilization': []
        }

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """
        Get current GPU memory usage in MB
        Returns:
            Tuple[float, float]: (allocated memory, cached memory) in MB
        """
        if not self.using_gpu:
            return 0.0, 0.0
        
        try:
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            memory_cached = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            
            # Record in history
            if self.monitoring:
                self.history['timestamps'].append(time.time())
                self.history['gpu_memory_allocated'].append(memory_allocated)
                self.history['gpu_memory_cached'].append(memory_cached)
            
            return memory_allocated, memory_cached
        except Exception as e:
            if self.log_warnings:
                warnings.warn(f"Could not get GPU memory usage: {str(e)}")
            return 0.0, 0.0

    def get_gpu_utilization(self) -> float:
        """
        Get GPU utilization percentage, considering both compute and memory usage
        Returns:
            float: GPU utilization percentage
        """
        if not self.using_gpu:
            return 0.0
        
        try:
            # Get memory metrics
            memory_allocated = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_utilization = (memory_allocated / total_memory) * 100
            
            # Get compute utilization
            compute_utilization = float(torch.cuda.utilization())
            
            # If no memory is allocated, cap the utilization
            if memory_allocated == 0:
                final_utilization = 0.0
            else:
                # Use a weighted average of memory and compute utilization
                final_utilization = (memory_utilization + compute_utilization) / 2
            
            if self.monitoring:
                self.history['gpu_utilization'].append(final_utilization)
            
            return final_utilization
            
        except Exception as e:
            if self.log_warnings:
                warnings.warn(f"Could not get GPU utilization: {str(e)}")
            return 0.0

    def start(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.start_time = time.time()
        self.start_cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        if self.using_gpu:
            self.clear_gpu_memory()  # Clear cache before starting
            self.start_gpu_memory = self.get_gpu_memory_usage()
            torch.cuda.synchronize()  # Ensure GPU sync
            
        # Reset history
        for key in self.history:
            self.history[key] = []

    def stop(self) -> Dict:
        """
        Stop monitoring and return resource usage
        Returns:
            Dict: Resource usage statistics
        """
        if not self.monitoring:
            raise RuntimeError("Monitor wasn't started")
            
        if self.using_gpu:
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        resources = {
            'execution_time': end_time - self.start_time,
            'cpu_memory_usage': end_cpu_memory - self.start_cpu_memory,
            'peak_cpu_percent': psutil.cpu_percent(),
            'cpu_memory_percent': psutil.virtual_memory().percent
        }
        
        if self.using_gpu:
            end_gpu_memory = self.get_gpu_memory_usage()
            gpu_props = torch.cuda.get_device_properties(0)
            
            current_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            resources.update({
                'gpu_memory_allocated': current_memory_allocated,
                'gpu_memory_cached': end_gpu_memory[1] - self.start_gpu_memory[1],
                'gpu_total_memory': gpu_props.total_memory / (1024 * 1024),
                'gpu_memory_percent': (current_memory_allocated / (gpu_props.total_memory / (1024 * 1024))) * 100,
                'gpu_utilization': self.get_gpu_utilization(),
                'gpu_name': gpu_props.name,
                'gpu_max_memory_allocated': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'gpu_peak_memory_cached': torch.cuda.max_memory_reserved() / (1024 * 1024)
            })
        
        self.monitoring = False
        return resources

    def clear_gpu_memory(self):
        """Clear GPU memory cache and reset peak stats"""
        if self.using_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
    def get_memory_history(self) -> Dict:
        """Get historical memory usage data"""
        return self.history
    
    @contextmanager
    def monitor_section(self, section_name: str):
        """Context manager for monitoring specific code sections"""
        try:
            self.start()
            yield
        finally:
            resources = self.stop()
            if self.log_warnings:
                print(f"\nResource usage for section '{section_name}':")
                for key, value in resources.items():
                    if isinstance(value, (int, float)):
                        print(f"{key}: {value:.2f}")
                    else:
                        print(f"{key}: {value}")
                                 
class BaseShapAnalyzer:
    def __init__(self, output_dir: str = 'shap_results', attr_name: List[str] = None):
        self.output_dir = output_dir
        self.attr_name = attr_name
        self.monitor = ResourceMonitor()
        self.evaluation_history = []
        self.create_output_directory()
        
        
    def create_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
    # def _plot_evaluation_results(self, results, gains, tau=0.05):
    #     """
    #     Plot and save evaluation results separately with improved legend placement.
    #     """
    #     n_attrs = [r['n_attributes'] for r in results]
    #     scores = [r['f1_score_mean'] for r in results]
    #     std = [r['f1_score_std'] for r in results]

    #     # Plot 1: Performance vs Number of Attributes
    #     plt.figure(figsize=(10, 6))
    #     plt.errorbar(n_attrs, scores, yerr=std, marker='o', color='blue', capsize=5)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xlabel('Number of Attributes')
    #     plt.ylabel('F1 Score')
    #     plt.title('Performance vs Number of Attributes')
        
    #     # Adjust layout to prevent cutoff
    #     plt.tight_layout()
    #     plt.savefig(f'{self.output_dir}/performance_plot.png', 
    #                 dpi=300, bbox_inches='tight')
    #     plt.close()

    #     # Plot 2: Gain Ratio Visualization
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(1, len(gains)+1), gains, marker='o', color='green')
    #     plt.axhline(y=tau, color='r', linestyle='--', label=f'Threshold ({tau})')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xlabel('Attribute Addition Step')
    #     plt.ylabel('Gain Ratio')
    #     plt.title('Gain Ratio per Attribute Addition')
        
    #     # Place legend outside the plot
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    #     # Adjust layout to prevent cutoff
    #     plt.tight_layout()
    #     plt.savefig(f'{self.output_dir}/gain_ratio_plot.png', 
    #                 dpi=300, bbox_inches='tight')
    #     plt.close()

    #     self._create_combined_plot(results, gains, tau)
        
    # def _create_combined_plot(self, results, gains, tau=0.05):
    #     """Create a combined plot of all metrics with proper spacing."""
    #     plt.figure(figsize=(20, 5))
        
    #     # Add extra space at right for legends
    #     plt.subplots_adjust(right=0.85, wspace=0.4)
        
    #     n_attrs = [r['n_attributes'] for r in results]
        
    #     # Plot 1: Performance
    #     plt.subplot(121)
    #     scores = [r['f1_score_mean'] for r in results]
    #     std = [r['f1_score_std'] for r in results]
    #     plt.errorbar(n_attrs, scores, yerr=std, marker='o', color='blue', capsize=5)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xlabel('Number of Attributes')
    #     plt.ylabel('F1 Score')
    #     plt.title('Performance vs Attributes')
        
    #     # Plot 2: Gains
    #     plt.subplot(122)
    #     plt.plot(range(1, len(gains)+1), gains, marker='o', color='green')
    #     plt.axhline(y=tau, color='r', linestyle='--', label=f'Threshold ({tau})')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xlabel('Attribute Addition Step')
    #     plt.ylabel('Gain Ratio')
    #     plt.title('Gain Ratio')
    #     # Place legend inside with small font
    #     plt.legend(loc='upper right', fontsize='small')
        
        
    #     plt.savefig(f'{self.output_dir}/combined_evaluation_results.png', 
    #                 dpi=300, bbox_inches='tight', pad_inches=0.5)
    #     plt.close()
    
    def _plot_comprehensive_results(self, results: List[Dict], contributions: List[Dict], gains: List[float], tau: float = 0.05):
        """
        Create a comprehensive visualization with four subplots:
        1. Performance vs Attributes
        2. Gain Ratio
        3. Marginal Gain per Attribute
        4. Cumulative Performance
        
        Args:
            results: List of evaluation results
            contributions: List of attribute contributions
            gains: List of gain ratios
            tau: Threshold value for gain ratio
        """
        # Set up the figure with a 2x2 grid
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Extract common data
        n_attrs = [r['n_attributes'] for r in results]
        attrs = [c['attribute'] for c in contributions]
        
        # Plot 1: Performance vs Number of Attributes (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        scores = [r['f1_score_mean'] for r in results]
        std = [r['f1_score_std'] for r in results]
        ax1.errorbar(n_attrs, scores, yerr=std, marker='o', color='blue', capsize=5)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel('Number of Attributes')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Performance vs Number of Attributes')
        
        # Plot 2: Gain Ratio (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(range(1, len(gains)+1), gains, marker='o', color='green')
        ax2.axhline(y=tau, color='r', linestyle='--', label=f'Threshold ({tau})')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlabel('Attribute Addition Step')
        ax2.set_ylabel('Gain Ratio')
        ax2.set_title('Gain Ratio per Attribute Addition')
        ax2.legend()
        
        # Plot 3: Marginal Gains (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        marginal_gains = [c['marginal_gain'] for c in contributions]
        bars = ax3.bar(attrs, marginal_gains)
        # Color bars based on gain value
        for bar, gain in zip(bars, marginal_gains):
            bar.set_color('g' if gain >= tau else 'r')
        ax3.axhline(y=tau, color='r', linestyle='--', label=f'Threshold ({tau})')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.set_xlabel('Attributes')
        ax3.set_ylabel('Marginal Gain')
        ax3.set_title('Marginal Gain per Attribute')
        # Rotate attribute names for better readability
        ax3.tick_params(axis='x', rotation=45, labelsize=8)
        ax3.legend()
        
        # Plot 4: Cumulative Performance (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        cumulative_f1 = [c['cumulative_f1'] for c in contributions]
        ax4.plot(range(1, len(cumulative_f1) + 1), cumulative_f1, marker='o', color='purple')
        # Add attribute names at each point
        for i, (f1, attr) in enumerate(zip(cumulative_f1, attrs)):
            ax4.annotate(attr, (i + 1, f1), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.set_xlabel('Number of Attributes')
        ax4.set_ylabel('Cumulative F1 Score')
        ax4.set_title('Cumulative Performance')
        
        # Adjust layout and save
        plt.savefig(f'{self.output_dir}/comprehensive_analysis.png', 
                    dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
    # def _plot_comprehensive_results(self, results: List[Dict], contributions: List[Dict], tau=0.05):
    #     """Create comprehensive visualization including attribute contributions"""
    #     # Create figure with subplots
    #     plt.figure(figsize=(20, 10))
    #     plt.subplots_adjust(right=0.85, wspace=0.4)
        
    #     n_attrs = [r['n_attributes'] for r in results]
    #     attrs = [c['attribute'] for c in contributions]
        
    #     # Plot 1: Performance with error bars
    #     plt.subplot(131)
    #     scores = [r['f1_score_mean'] for r in results]
    #     std = [r['f1_score_std'] for r in results]
    #     plt.errorbar(n_attrs, scores, yerr=std, marker='o', color='blue', capsize=5)
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xlabel('Number of Attributes')
    #     plt.ylabel('F1 Score')
    #     plt.title('Performance vs Number of Attributes')
        
    #     # Plot 2: Marginal Gains
    #     plt.subplot(132)
    #     gains = [c['marginal_gain'] for c in contributions]
    #     plt.bar(attrs, gains, color=['g' if g >= 0 else 'r' for g in gains])
    #     plt.axhline(y=tau, color='r', linestyle='--', label=f'Threshold ({tau})')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xlabel('Attributes')
    #     plt.ylabel('Marginal Gain')
    #     plt.title('Marginal Gain per Attribute')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.legend()
      
    #     # Plot 4: Cumulative Performance
    #     plt.subplot(133)
    #     cumulative_f1 = [c['cumulative_f1'] for c in contributions]
    #     plt.plot(range(1, len(cumulative_f1) + 1), cumulative_f1, marker='o', color='green')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.xlabel('Number of Attributes')
    #     plt.ylabel('Cumulative F1 Score')
    #     plt.title('Cumulative Performance')
        
    #     # Save plots
    #     plt.tight_layout()
    #     plt.savefig(f'{self.output_dir}/comprehensive_analysis.png', 
    #                 dpi=300, bbox_inches='tight', pad_inches=0.5)
    #     plt.close()

    def progressive_evaluation(self, X: pd.DataFrame, y: pd.Series, ranked_attrs: List[str], 
                         base_classifier=DecisionTreeClassifier(max_depth=8, random_state=42), 
                         cv=5) -> Dict:
        """Evaluate attribute combinations progressively"""
        results = []
        current_attrs = []
        
        # Use PyTorch model if on GPU
        if self.device.type == 'cuda' and base_classifier is None:
            base_classifier = UNetClassifier()  # Instantiate the class
        elif base_classifier is None:
            base_classifier = DecisionTreeClassifier(max_depth=8, random_state=42)
            
        for attr in ranked_attrs:
            self.monitor.start()
            current_attrs.append(attr)
            
            X_current = X[current_attrs]
            
            if self.device.type == 'cuda':
                # GPU-based cross validation
                cv_scores = self._gpu_cross_val_score(
                    base_classifier,
                    X_current,
                    y,
                    cv=cv
                )
            else:
                # CPU-based cross validation
                cv_scores = cross_val_score(
                    base_classifier,
                    X_current,
                    y,
                    cv=cv,
                    scoring='f1_macro'
                )
            
            resources = self.monitor.stop()
            
            result = {
                'attributes': current_attrs.copy(),
                'n_attributes': len(current_attrs),
                'f1_score_mean': cv_scores.mean(),
                'f1_score_std': cv_scores.std(),
                'resources': resources
            }
            
            results.append(result)
            self.evaluation_history.append(result)
            
            # Clear GPU memory after each evaluation
            if self.device.type == 'cuda':
                self.monitor.clear_gpu_memory()
        
        gains = self._calculate_gains(results)
        # self._plot_evaluation_results(results, gains)
        
        # Calculate contributions and plot comprehensive results
        contributions = self._calculate_attribute_contributions(results)
        self._plot_comprehensive_results(results, contributions, gains)
        
        return {
            'results': results,
            'gains': gains,
            'contributions': contributions,
            'ranked_attrs': ranked_attrs
        }

    def _calculate_gains(self, results: List[Dict]) -> List[float]:
        """Calculate performance gains between consecutive attribute additions"""
        gains = []
        prev_f1 = 0
        
        for result in results:
            current_f1 = result['f1_score_mean']
            gain = current_f1 - prev_f1
            gains.append(gain)
            prev_f1 = current_f1
            
        return gains

    def _calculate_attribute_contributions(self, results: List[Dict]) -> List[Dict]:
        """Calculate the contribution of each attribute to the overall performance"""
        contributions = []
        prev_f1 = 0
        
        for result in results:
            current_f1 = result['f1_score_mean']
            marginal_gain = current_f1 - prev_f1
            
            contribution = {
                'attribute': result['attributes'][-1],
                'marginal_gain': marginal_gain,
                'cumulative_f1': current_f1
            }
            
            contributions.append(contribution)
            prev_f1 = current_f1
            
        return contributions        
    
    def _gpu_cross_val_score(self, model_class, X, y, cv):
        """Perform cross-validation on GPU"""
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            input_dim = X.shape[1]
            hidden_dim = min(64, input_dim * 2)
            num_classes = len(np.unique(y))
            
            model = model_class(input_dim, hidden_dim, num_classes).to(self.device)
            self.train_model(X_train, y_train)
            
            # Evaluate
            y_pred = self.predict(model, X_val)
            score = f1_score(y_val, y_pred, average='macro')
            scores.append(score)
            
        return np.array(scores)
    

    def select_optimal_attributes(self, eval_results: Dict, gain_threshold: float = 0.05) -> List[str]:
        """
        Select optimal attributes based on gain threshold
        """
        gains = eval_results['gains']
        ranked_attrs = eval_results['ranked_attrs']
        
        for i, gain in enumerate(gains):
            if gain < gain_threshold:
                return list(ranked_attrs[:i+1])
        
        return list(ranked_attrs)

    def optimize_feature_set(self, X: pd.DataFrame, y: pd.Series, ranked_attrs: List[str], 
                           gain_threshold: float = 0.05) -> Tuple[List[str], Dict]:
        """
        Run progressive evaluation and select optimal attributes
        """
        if self.device.type == 'cuda':
            classifier = UNetClassifier
        else:
            classifier = DecisionTreeClassifier(max_depth=8, random_state=42)
            
        eval_results = self.progressive_evaluation(
            X=X,
            y=y, 
            ranked_attrs=ranked_attrs,
            base_classifier=classifier
        )
        selected_attrs = self.select_optimal_attributes(eval_results, gain_threshold)
        
        # Print optimization results
        print("\nFeature Set Optimization Results:")
        print("-" * 40)
        print(f"Original features: {len(ranked_attrs)}")
        print(f"Optimized features: {len(selected_attrs)}")
        
        print("\nSelected attributes:")
        print("-" * 40)
        for attr in selected_attrs:
            print(f"- {attr}")
        
        # Print performance metrics for selected attribute set
        for result in eval_results['results']:
            if result['attributes'] == selected_attrs:
                print("\nPerformance Metrics:")
                print("-" * 40)
                print(f"F1 Score: {result['f1_score_mean']:.3f} ± {result['f1_score_std']:.3f}")
                
                resources = result['resources']
                # print("\nComputation Resources:")
                # print("-" * 40)
                # print(f"Time: {resources['execution_time']:.2f}s")
                # print(f"CPU Memory: {resources['cpu_memory_usage']:.2f}MB")
                
                # Print GPU metrics if available
                if self.device.type == 'cuda':
                    print("\nGPU Resources:")
                    print("-" * 40)
                    print(f"Allocated Memory: {resources['gpu_memory_allocated']:.2f}MB")
                    print(f"Cached Memory: {resources['gpu_memory_cached']:.2f}MB")
                    print(f"GPU Utilization: {resources['gpu_utilization']:.1f}%")
                    print(f"Total GPU Memory: {resources['gpu_total_memory']:.2f}MB\n")
                break
    
        return selected_attrs, eval_results
    


class ShapAnalyzer(BaseShapAnalyzer):
    def __init__(self, output_dir: str = 'shap_results', attr_name: List[str] = None, batch_size: int = 1000):
        super().__init__(output_dir, attr_name)
        self.output_dir = output_dir
        self.attr_name = attr_name
        self.batch_size = batch_size
        self.create_output_directory()
        self.device = torch.device('cpu')
        
        """
        Initialize SHAP analysis pipeline.
        
        Args:
            output_dir: Directory to save SHAP analysis results
            attr_name: List of feature names to use in the analysis
        """

        
    def create_output_directory(self):
        """Create directory for SHAP results if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _filter_columns_by_attributes(self, df: pd.DataFrame, selected_attrs: List[str]) -> List[str]:
        """
        Filter DataFrame columns based on selected attribute prefixes
        
        Args:
            df: DataFrame containing the data
            selected_attrs: List of base attribute names to filter by
            
        Returns:
            List of column names that start with any of the selected attributes
        """
        selected_columns = []
        for attr in selected_attrs:
            # Find all columns that start with this attribute name
            matching_cols = [col for col in df.columns if col.startswith(attr)]
            selected_columns.extend(matching_cols)
        
        # Always include the label column
        if 'label' in df.columns:
            selected_columns.append('label')
            
        return selected_columns
    
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
        # print(f'saved at {self.output_dir}')
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Train Random Forest model with optimized parallel processing"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Optimize RandomForest parameters for speed
        model = RandomForestClassifier(
            n_estimators=50,      # Reduced number of trees
            max_depth=10,         # Limited depth
            min_samples_leaf=10,  # Increased to reduce tree complexity
            max_features='sqrt',  # Reduced feature consideration
            n_jobs=-1,           # Use all CPU cores
            random_state=42
        )
        
        model.fit(X_train, y_train)
        return model, X_train, y_train, X_test, y_test

    @staticmethod
    def _process_batch(batch_data, model):
        """Static method to process a single batch"""
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(batch_data)

    def generate_shap_values(self, model: RandomForestClassifier, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate SHAP values using parallel batch processing with progress tracking"""
        
        
        # Calculate number of batches
        # n_batches = len(X_test) // self.batch_size + (1 if len(X_test) % self.batch_size != 0 else 0)
        
        all_shap_values = []
        
        # Create progress bar
        # pbar = tqdm(total=n_batches, desc="Calculating SHAP values", unit="batch")
        
        # Use ProcessPoolExecutor for parallel processing
        n_jobs = os.cpu_count() - 1  # Leave one CPU core free
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all batches
            future_to_batch = {}
            for i in range(0, len(X_test), self.batch_size):
                batch = X_test.iloc[i:i + self.batch_size]
                future = executor.submit(self._process_batch, batch, model)
                future_to_batch[future] = i
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_shap = future.result()
                all_shap_values.append((future_to_batch[future], batch_shap))
                # pbar.update(1)
        
        # pbar.close()
        
        # Sort results by batch index
        all_shap_values.sort(key=lambda x: x[0])
        all_shap_values = [x[1] for x in all_shap_values]
        
        # Combine results efficiently
        if isinstance(all_shap_values[0], list):
            # Multi-class case
            return [
                np.concatenate([batch[i] for batch in all_shap_values])
                for i in range(len(all_shap_values[0]))
            ]
        else:
            # Binary classification case
            return np.concatenate(all_shap_values)

    def run_complete_analysis(self, df_list: List[pd.DataFrame], gain_threshold: float = 0.05) -> Dict:
        """Optimized complete analysis pipeline"""
        # Combine data efficiently
        X, y = prepare_combined_data(df_list, self.attr_name)
        
        # Optional: Reduce dimensionality if needed
        if X.shape[1] > 100:  # If more than 100 features
            from sklearn.feature_selection import SelectPercentile, f_classif
            selector = SelectPercentile(f_classif, percentile=50)
            X = pd.DataFrame(selector.fit_transform(X, y), columns=X.columns[selector.get_support()])
        
        model, X_train, y_train, X_test, y_test = self.train_model(X, y)
        
        print("Calculating Init SHAP values...")
        shap_values = self.generate_shap_values(model, X_test)
        # plot init SHAP values
        n_classes = len(np.unique(y))
        for i in range(n_classes):
            self.plot_class_shap_analysis(shap_values, X_test, i)
        print("\nCalculating SHAP feature importance...")
        # Calculate feature importance efficiently
        n_classes = len(np.unique(y))
        global_importance_df = self.save_feature_importance(shap_values, X, n_classes)
        
        # Optimize feature set
        ranked_attrs = global_importance_df['feature'].tolist()
        selected_attrs, eval_results = self.optimize_feature_set(X, y, ranked_attrs, gain_threshold)
        
        # Create optimized dataset efficiently
        optimized_df_list = []
        for df in df_list:
            selected_columns = [col for col in df.columns if any(col.startswith(attr) for attr in selected_attrs) or col == 'label']
            optimized_df = df[selected_columns].copy()
            optimized_df_list.append(optimized_df)
        
        # Final analysis with optimized features
        X_opt, y_opt = prepare_combined_data(optimized_df_list, selected_attrs)
        model_opt, X_train_opt, y_train_opt, X_test_opt, y_test_opt = self.train_model(X_opt, y_opt)
        print('Calculating Optimized SHAP values for optimized model...')
        shap_values_opt = self.generate_shap_values(model_opt, X_test_opt)
        print('\n SHAP values analysis done')
        return {
            'initial_results': {
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

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        """
        Simple neural network classifier.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            num_classes: Number of output classes
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class TorchShapAnalyzer(BaseShapAnalyzer):
    def __init__(self, output_dir: str = 'torch_shap_results', attr_name: List[str] = None, batch_size: int = 1000):
        super().__init__(output_dir, attr_name)
        self.output_dir = output_dir
        self.attr_name = attr_name
        self.batch_size = batch_size
    
        """
        Initialize PyTorch-based SHAP analysis pipeline.
        
        Args:
            output_dir: Directory to save SHAP analysis results
            attr_name: List of feature names to use in the analysis
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.create_output_directory()
        
    def _filter_columns_by_attributes(self, df: pd.DataFrame, selected_attrs: List[str]) -> List[str]:
        """
        Filter DataFrame columns based on selected attribute prefixes
        
        Args:
            df: DataFrame containing the data
            selected_attrs: List of base attribute names to filter by
            
        Returns:
            List of column names that start with any of the selected attributes
        """
        selected_columns = []
        for attr in selected_attrs:
            # Find all columns that start with this attribute name
            matching_cols = [col for col in df.columns if col.startswith(attr)]
            selected_columns.extend(matching_cols)
        
        # Always include the label column
        if 'label' in df.columns:
            selected_columns.append('label')
            
        return selected_columns
       
    def create_output_directory(self):
        """Create directory for SHAP results if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
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
        
    def save_feature_importance(self, shap_values: List[np.ndarray], X: pd.DataFrame, n_classes: int) -> pd.DataFrame:
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[nn.Module, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Train PyTorch model with optimizations for SHAP analysis."""
        # Split data with validation set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train.values)
        
        # Create model
        input_dim = X.shape[1]
        hidden_dim = min(64, input_dim * 2)
        num_classes = len(np.unique(y))
        
        model = UNetClassifier(input_dim, hidden_dim, num_classes).to(self.device)
        model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)
        
        # Training setup
        scaler = torch.cuda.amp.GradScaler()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Create DataLoader
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=min(64, len(X_train) // 10),
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        )
        
        # Training configuration
        max_epochs = 6
        min_epochs = 5
        patience = 3
        convergence_threshold = 0.001
        best_loss = float('inf')
        best_model = None
        patience_counter = 0
        previous_loss = float('inf')
        
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            
            for batch_X, batch_y in train_loader:
                # Move batch to GPU and train
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                # Optimization step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Update metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                running_total += batch_y.size(0)
                running_correct += predicted.eq(batch_y).sum().item()
            
            # Epoch-end processing
            avg_epoch_loss = running_loss / len(train_loader)
            # accuracy = 100.0 * running_correct / running_total
            
            # Model checkpointing
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Convergence checking
            loss_improvement = (previous_loss - avg_epoch_loss) / previous_loss if previous_loss != float('inf') else float('inf')
            previous_loss = avg_epoch_loss
            
            # Early stopping checks
            if epoch >= min_epochs:
                if patience_counter >= patience:
                    # print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                if loss_improvement < convergence_threshold:
                    # print(f"Converged after {epoch + 1} epochs")
                    break
        
        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)
        
        return model, X_train, y_train, X_test, y_test
    
    @staticmethod
    def _process_batch(batch_data, model):
        """Static method to process a single batch"""
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(batch_data)

    def generate_shap_values(self, model: nn.Module, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate SHAP values using batch processing with CUDA safety measures"""
        # Set multiprocessing start method to 'spawn'
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        
        # Move model to CPU for SHAP calculations
        model = model.cpu()
        
        # Initialize KernelExplainer with subset of background data
        background = shap.kmeans(X_test.values, k=min(50, len(X_test)))
        explainer = shap.KernelExplainer(
            lambda x: self.model_predict(model, x),
            background,
            link="identity"
        )
        
        # Calculate batch size
        batch_size = min(100, len(X_test))  # Adjust based on available memory
        
        all_shap_values = []
        
        # Create progress bar
    
        for i in range(0, len(X_test), batch_size):
            # Process batch
            batch = X_test.iloc[i:i + batch_size].values
            batch_shap = explainer.shap_values(batch)
            
            # Store results
            if isinstance(batch_shap, list):
                # Multi-class case
                if not all_shap_values:
                    all_shap_values = [[] for _ in range(len(batch_shap))]
                for j, class_shap in enumerate(batch_shap):
                    all_shap_values[j].append(class_shap)
            else:
                # Binary classification case
                all_shap_values.append(batch_shap)
            
        # Combine results
        if isinstance(all_shap_values[0], list):
            # Multi-class case
            return [np.concatenate(class_shap) for class_shap in all_shap_values]
        else:
            # Binary classification case
            return np.concatenate(all_shap_values)
        
    def predict(self, model: nn.Module, X: pd.DataFrame) -> np.ndarray:
        """
        Make class predictions for model evaluation.
        Returns class labels instead of probabilities. For SHAP calculations.
        """
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
        
    def model_predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Wrapper function for model prediction to use with SHAP."""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)  # Keep on CPU
            outputs = model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
            return probas.numpy()  # No need for .cpu() since we're already on CPU

    def run_complete_analysis(self, df_list: List[pd.DataFrame], gain_threshold: float = 0.05) -> Dict:
        """Run complete SHAP analysis pipeline with CUDA safety measures"""
        # Initial setup and model training
        X, y = prepare_combined_data(df_list, self.attr_name)
       
        model, X_train, y_train, X_test, y_test = self.train_model(X, y)
        print("Calculating Init Torch SHAP values...")
        model = model.cpu()
        shap_values = self.generate_shap_values(model, X_test)
        
        # plot shap analysis
        n_classes = len(np.unique(y))
        for i in range(n_classes):
            self.plot_class_shap_analysis(shap_values, X_test, i)
        # Move model back to GPU if available
        model = model.to(self.device)
        
        print("\nCalculating Init Torch SHAP feature importance...")
        n_classes = len(np.unique(y))
        global_importance_df = self.save_feature_importance(shap_values, X, n_classes)
        ranked_attrs = global_importance_df['feature'].tolist()
        
        selected_attrs, eval_results = self.optimize_feature_set(X, y, ranked_attrs, gain_threshold)
        # Create optimized dataset
        optimized_df_list = []
        for df in df_list:
            selected_columns = [col for col in df.columns if any(col.startswith(attr) for attr in selected_attrs) or col == 'label']
            optimized_df = df[selected_columns].copy()
            optimized_df_list.append(optimized_df)
        
        # Final analysis with optimized features
        X_opt, y_opt = prepare_combined_data(optimized_df_list, selected_attrs)
        model_opt, X_train_opt, y_train_opt, X_test_opt, y_test_opt = self.train_model(X_opt, y_opt)
        
        # Move model to CPU for final SHAP calculations
        model_opt = model_opt.cpu()
        print('Calculating Optimized Torch SHAP values for optimized model...')
        shap_values_opt = self.generate_shap_values(model_opt, X_test_opt)
        print('\nTorch SHAP values analysis done')
        
        return {
            'initial_results': {
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
    # np.random.seed(seed) # set to reproduce same results
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

def process_single_trace(args):
    """
    Process a single trace for all attributes
    
    Args:
        args: Tuple containing (position_idx, il, xl, traces_volume, scalers, attr_name)
    
    Returns:
        DataFrame for single trace
    """
    position_idx, il, xl, traces_volume, scalers, attr_name = args
    
    trace_dict = {}
    for j, attr in enumerate(attr_name):
        trace_data = traces_volume[j][:, position_idx]
        
        if scalers is not None:
            trace_data = scalers[attr].transform(trace_data.reshape(-1, 1)).ravel()
            
        trace_dict[f'{attr}_Inline_{il}_Crossline_{xl}'] = trace_data
        
    return pd.DataFrame(trace_dict)


@monitor_cpu_usage
def prepare_trace_data_parallel(traces_volume, labels, positions, attr_name, normalize=True, n_jobs=None):
    """
    Prepare data for each trace position with parallel processing.
    
    Args:
        traces_volume: Input traces data
        labels: Classification labels
        positions: List of trace positions
        attr_name: List of attribute names
        normalize: Whether to normalize the data
        n_jobs: Number of CPU cores to use (None for optimal)
    
    Returns:
        tuple: (list of DataFrames, scalers) if normalize=True, else list of DataFrames
    """
    if n_jobs is None:
        n_jobs = get_optimal_cpu_count()
    
    # Initialize progress tracking
    # total_steps = len(positions)
    # progress = tqdm(total=total_steps, desc="Preparing trace data", unit="trace")
    
        # Fit scalers if normalization is enabled
    scalers = None
    if normalize:
        # progress.write("Fitting scalers...")
        scalers = {
            attr: StandardScaler().fit(traces_volume[j].reshape(-1, 1))
            for j, attr in enumerate(attr_name)
        }
    
    # Prepare and process traces in parallel
    process_args = [
        (i, il, xl, traces_volume, scalers, attr_name)
        for i, (il, xl) in enumerate(positions)
    ]
    
    with Pool(n_jobs) as pool:
        # Process traces and update progress
        df_list = []
        for df in pool.imap(process_single_trace, process_args):
            df_list.append(df)
            # progress.update()
    
    # Add labels to DataFrames (in-place)
    # progress.set_description("Adding labels")
    for i, df in enumerate(df_list):
        df['label'] = labels[:, i]
        # progress.update(0)  # Refresh display without incrementing
    
    return (df_list, scalers) if normalize else df_list
        
    # finally:
    #     progress.close()

def load_seismic_data(data_dir: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load seismic data and labels from specified directory.
    
    Args:
        data_dir: Base directory containing the data files
    
    Returns:
        Tuple of (list of seismic volumes, labels)
    """
    # Define file paths
    file_paths = {
        'seismic': 'F3_seismic.npy',
        'freq': 'F3_crop_horizon_freq.npy',
        'dip': 'F3_predict_MCDL_crossline.npy',
        'phase': 'F3_crop_horizon_phase.npy',
        'rms': 'F3_RMSAmp.npy',
        'complex': 'F3_complex_trace.npy',
        'coherence': 'F3_coherence.npy',
        'azc': 'F3_Average_zero_crossing.npy',
        'labels': 'test_label_no_ohe.npy'
    }
    
    # Load volumes
    volumes = []
    for key, filepath in file_paths.items():
        if key != 'labels':
            data = np.load(os.path.join(data_dir, filepath))
            volumes.append(data)
            
    # Load labels
    labels = np.load(os.path.join(data_dir, file_paths['labels']))
    
    return volumes, labels

def preprocess_volumes(volumes: List[np.ndarray], labels: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Preprocess seismic volumes and labels to ensure consistent shapes.
    
    Args:
        volumes: List of seismic volumes
        labels: Label volume
    
    Returns:
        Tuple of (preprocessed volumes, preprocessed labels)
    """
    processed_volumes = []
    
    # Process each volume
    for i, volume in enumerate(volumes):
        if i == 0:  # seismic volume
            volume = np.squeeze(volume).reshape(-1, 951, 288)
        elif i == 2:  # dip volume
            volume = np.swapaxes(volume, -1, 1)
        else:
            volume = volume.reshape(-1, 951, 288)
        volume = volume[:600, :,: ]
        processed_volumes.append(volume)
    
    # Process labels
    processed_labels = labels.reshape(-1, 951, 288)
    processed_labels = processed_labels[:600, :, :]
    
    return processed_volumes, processed_labels

def prepare_combined_data(df_list: List[pd.DataFrame], attr_name: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare and combine data from multiple DataFrames for evaluation.
    
    Args:
        df_list: List of DataFrames containing the data
        attr_name: List of attribute names to use
    
    Returns:
        Tuple of (combined features DataFrame, combined labels Series)
    """
    X_list, y_list = [], []
    
    for df in df_list:
        feature_cols = [col for col in df.columns if col != 'label']
        X_current = df[feature_cols].copy()
        X_current.columns = attr_name[:len(feature_cols)]
        y_current = df['label']
        
        X_list.append(X_current)
        y_list.append(y_current)
    
    return pd.concat(X_list, axis=0, ignore_index=True), pd.concat(y_list, axis=0, ignore_index=True)


def run_shap_analysis(df_list: List[pd.DataFrame], attr_name: List[str]) -> Dict:
    """
    Run SHAP analysis using both tree-based and neural network models.
    
    Args:
        df_list: List of DataFrames containing prepared data
        attr_name: List of attribute names
    
    Returns:
        Dictionary containing results from both analyses
    """
    # Tree-based SHAP analysis
    tree_analyzer = ShapAnalyzer(output_dir='shap_results', attr_name=attr_name)
    tree_results = tree_analyzer.run_analysis(df_list)
    
    # Neural network SHAP analysis
    torch_analyzer = TorchShapAnalyzer(output_dir='torch_shap_results', attr_name=attr_name)
    torch_results = torch_analyzer.run_analysis(df_list)
    
    return {
        'tree_results': tree_results,
        'torch_results': torch_results
    }



def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configuration
    data_dir = '/home/dell/disk1/Jinlong/Horizontal-data' 
    attr_name = ['seismic', 'freq', 'dip', 'phase', 'rms', 'complex', 'coherence', 'azc']
    # attr_name = ['seismic', 'freq', 'dip', 'phase']
    # attr_name = ['seismic', 'freq']
    n_traces = 100 # 20/s then 50000 causes 41 minutes
    seed = 42
    
    # Load and preprocess data
    volumes, labels = load_seismic_data(data_dir)
    processed_volumes, processed_labels = preprocess_volumes(volumes, labels)
    
    # Select random traces
    selected_traces_volume, selected_labels, positions = select_random_traces(
        processed_volumes, processed_labels, n_traces, seed
    )
    print(f"Selected {n_traces} random traces done!")
    # Prepare data with normalization
    # df_list, scalers = prepare_trace_data(
    #     selected_traces_volume, selected_labels, positions, attr_name, normalize=True
    # )
    n_jobs = get_optimal_cpu_count(max_percent=80)
    
    df_list, scalers = prepare_trace_data_parallel(
    selected_traces_volume, 
    selected_labels, 
    positions, 
    attr_name, 
    normalize=True,
    n_jobs=n_jobs
    )
    print('Data preparation done!\n')
 
    # Create pairplot
    # for i, (il, xl) in enumerate(positions):
    #     file_name = f'Inline_{il}_Crossline_{xl}'
    #     pairplot = create_visualization(df_list[i], file_name)
    
    # Run complete analysis pipeline
    print('Running SHAP analysis...')
    analyzer = ShapAnalyzer(
    output_dir='shap_results',
    attr_name=attr_name,
    batch_size=3000  # Adjust based on your system's memory
)
    shap_results = analyzer.run_complete_analysis(df_list, gain_threshold=0.05) 
    
    print('Running Torch SHAP analysis...\n')
    
    torch_analyzer = TorchShapAnalyzer(
        output_dir='torch_shap_results', 
        attr_name=attr_name,
        batch_size=3000 )
    
    torch_shap_results = torch_analyzer.run_complete_analysis(df_list, gain_threshold=0.05) 
    
    return shap_results, torch_shap_results
    


if __name__ == "__main__":
    shap_results, torch_shap_results = main()

