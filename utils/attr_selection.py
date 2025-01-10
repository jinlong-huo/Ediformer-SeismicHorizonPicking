import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score
import shap
import time
import psutil
import datetime
import json
from memory_profiler import profile

# Generate synthetic seismic attributes data
def generate_synthetic_data(n_samples=1000, n_attributes=10, n_classes=3):
    """
    Generate synthetic seismic attributes data
    """
    np.random.seed(42)
    
    # Generate base attributes (amplitude, frequency, phase)
    amplitude = np.random.normal(0, 1, n_samples)
    frequency = np.random.normal(30, 5, n_samples)
    phase = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Generate derived attributes
    attributes = {
        'amplitude': amplitude,
        'frequency': frequency,
        'phase': phase,
        'rms_amp': np.abs(amplitude) + np.random.normal(0, 0.1, n_samples),
        'inst_freq': frequency + np.random.normal(0, 1, n_samples),
        'coherence': np.random.uniform(0, 1, n_samples),
        'dip': np.random.normal(0, 0.2, n_samples),
        'curvature': np.random.normal(0, 1, n_samples),
        'sweetness': amplitude * frequency + np.random.normal(0, 0.5, n_samples),
        'envelope': np.abs(amplitude) + np.random.normal(0, 0.2, n_samples)
    }
    
    # Create feature matrix
    X = pd.DataFrame(attributes)
    
    # Generate labels (simplified horizon classification)
    # Make labels dependent on some attributes to create meaningful patterns
    probabilities = np.zeros((n_samples, n_classes))
    probabilities[:, 0] = np.exp(-(amplitude**2 + frequency**2))
    probabilities[:, 1] = np.exp(-(amplitude**2 + phase**2))
    probabilities[:, 2] = np.exp(-(frequency**2 + phase**2))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    y = np.array([np.random.choice(n_classes, p=p) for p in probabilities])
    
    return X, y

class ResourceMonitor:
    """Monitor computational resources during execution"""
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.resources = []
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def stop(self):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'execution_time': end_time - self.start_time,
            'memory_usage': end_memory - self.start_memory
        }

class AttributeEvaluator:
    def __init__(self, base_classifier=None, cv=5):
        self.base_classifier = base_classifier or DecisionTreeClassifier(max_depth=8, random_state=42)
        self.cv = cv
        self.monitor = ResourceMonitor()
        self.evaluation_history = []
        
    def calculate_shap_values(self, X, y):
        """Calculate SHAP values for features"""
        self.monitor.start()
        
        # Fit the model
        self.base_classifier.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.base_classifier)
        shap_values = explainer.shap_values(X)
        
        # For multiple classes, take the mean absolute SHAP value across classes
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        
        # Calculate mean absolute SHAP value for each feature
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        shap_importance = pd.Series(mean_shap_values, index=X.columns)
        
        resources = self.monitor.stop()
        
        return shap_importance, resources

    def progressive_evaluation(self, X, y, shap_ranked_attrs):
        """Evaluate attribute combinations progressively"""
        results = []
        current_attrs = []
        
        for attr in shap_ranked_attrs:
            self.monitor.start()
            current_attrs.append(attr)
            
            # Current feature set
            X_current = X[current_attrs]
            
            # Cross validation
            cv_scores = cross_val_score(
                self.base_classifier,
                X_current,
                y,
                cv=self.cv,
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
            
        return results

    def calculate_gains(self, results):
        """Calculate gain ratios between consecutive attribute combinations"""
        gains = []
        for i in range(1, len(results)):
            prev_score = results[i-1]['f1_score_mean']
            curr_score = results[i]['f1_score_mean']
            gain = (curr_score - prev_score) / prev_score
            gains.append(gain)
        return gains

    def plot_results(self, results, gains, tau=0.05):
        """Plot evaluation results"""
        plt.figure(figsize=(15, 5))
        
        # Plot scores
        plt.subplot(131)
        n_attrs = [r['n_attributes'] for r in results]
        scores = [r['f1_score_mean'] for r in results]
        std = [r['f1_score_std'] for r in results]
        
        plt.errorbar(n_attrs, scores, yerr=std, marker='o')
        plt.xlabel('Number of Attributes')
        plt.ylabel('F1 Score')
        plt.title('Performance vs Number of Attributes')
        
        # Plot gains
        plt.subplot(132)
        plt.plot(range(1, len(gains)+1), gains, marker='o')
        plt.axhline(y=tau, color='r', linestyle='--', label=f'Threshold ({tau})')
        plt.xlabel('Attribute Addition Step')
        plt.ylabel('Gain Ratio')
        plt.title('Gain Ratio per Attribute Addition')
        plt.legend()
        
        # Plot resource usage
        plt.subplot(133)
        times = [r['resources']['execution_time'] for r in results]
        memory = [r['resources']['memory_usage'] for r in results]
        
        plt.plot(n_attrs, times, marker='o', label='Time (s)')
        plt.plot(n_attrs, memory, marker='s', label='Memory (MB)')
        plt.xlabel('Number of Attributes')
        plt.ylabel('Resource Usage')
        plt.title('Computational Resources')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def save_results(self, filename_prefix):
        """Save evaluation results to JSON"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.evaluation_history, f, indent=4, default=str)
        
        return filename

def main():
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, n_attributes=10, n_classes=3)
    
    # Initialize evaluator
    evaluator = AttributeEvaluator()
    
    # Calculate SHAP values
    shap_importance, shap_resources = evaluator.calculate_shap_values(X, y)
    
    # Get ranked attributes
    ranked_attrs = shap_importance.sort_values(ascending=False).index
    
    # Perform progressive evaluation
    results = evaluator.progressive_evaluation(X, y, ranked_attrs)
    
    # Calculate gains
    gains = evaluator.calculate_gains(results)
    
    # Plot results
    evaluator.plot_results(results, gains)
    
    # Save results
    results_file = evaluator.save_results('attribute_evaluation')
    print(f"Results saved to {results_file}")
    
    # Print summary
    print("\nTop 5 attributes by SHAP importance:")
    print(shap_importance.sort_values(ascending=False).head())
    
    print("\nAttribute combination recommendations:")
    for i, result in enumerate(results):
        print(f"\nCombination {i+1}:")
        print(f"Attributes: {result['attributes']}")
        print(f"F1 Score: {result['f1_score_mean']:.3f} ± {result['f1_score_std']:.3f}")
        print(f"Time: {result['resources']['execution_time']:.2f}s")
        print(f"Memory: {result['resources']['memory_usage']:.2f}MB")

if __name__ == "__main__":
    main()