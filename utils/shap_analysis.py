import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim


os.makedirs('shap_results', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_synthetic_data(n_samples=1000, n_features=5, n_classes=5):
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(X, columns=feature_names)
    
    # Generate a multi-class target variable with some relationship to features
    y = np.random.choice(n_classes, size=n_samples)  # Random class assignment for simplicity
    
    return X, y

# Generate data
X, y = generate_synthetic_data(n_classes=5)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a model with multi-class classification (RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Generate and save SHAP plots for each class

for i in range(model.n_classes_):  # Iterate through each class
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, i], X_test)  # For each class i
    plt.tight_layout()
    plt.savefig(f'shap_results/summary_plot_class_{i}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, i], X_test, plot_type='bar')  # For each class i
    plt.tight_layout()
    plt.savefig(f'shap_results/feature_importance_plot_class_{i}.png')
    plt.close()

# Save SHAP values to CSV for each class
for i in range(model.n_classes_):
    shap_df = pd.DataFrame(shap_values[:, :, i], columns=X.columns)  # For each class i
    shap_df.to_csv(f'shap_results/shap_values_class_{i}.csv', index=False)

# Calculate and save feature importance summary for each class
for i in range(model.n_classes_):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values[i]).mean(axis=0)  # For each class i
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    feature_importance.to_csv(f'shap_results/feature_importance_class_{i}.csv', index=False)

# Save example predictions with SHAP contributions
example_predictions = pd.DataFrame()
example_predictions['true_label'] = y_test
example_predictions['predicted_class'] = model.predict(X_test)
example_predictions['predicted_prob'] = model.predict_proba(X_test).max(axis=1)  # Probability of the predicted class

for i, col in enumerate(X.columns):
    shap_df = pd.DataFrame(shap_values[:, :, i], columns=X.columns)  # Use class 1
    example_predictions[f'shap_{col}'] = shap_df[col]

    
example_predictions.to_csv('shap_results/example_predictions.csv', index=False)

print("SHAP analysis completed. Results saved in 'shap_results' directory:")
print("1. summary_plot_class_X.png - SHAP summary plot for each class")
print("2. feature_importance_plot_class_X.png - Feature importance plot for each class")
print("3. shap_values_class_X.csv - Raw SHAP values for each class")
print("4. feature_importance_class_X.csv - Feature importance summary for each class")
print("5. example_predictions.csv - Example predictions with SHAP contributions")
