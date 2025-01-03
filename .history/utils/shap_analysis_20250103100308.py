import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_synthetic_data(n_samples=1000, n_features=5):
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(X, columns=feature_names)
    
    # Generate target variable with some relationship to features
    y = (X['feature_0'] > 0).astype(int) & (X['feature_1'] > 0).astype(int)
    
    return X, y

# Generate data
X, y = generate_synthetic_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Create directory for saving results
import os
os.makedirs('shap_results', exist_ok=True)

# Generate and save various SHAP plots
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:,:, 1], X_test)  # Use class 1 for binary classification
plt.tight_layout()
plt.savefig('shap_results/summary_plot.png')
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:,:, 1], X_test, plot_type='bar')  # Use class 1
plt.tight_layout()
plt.savefig('shap_results/feature_importance_plot.png')
plt.close()

# Save SHAP values to CSV (using class 1 values for binary classification)
shap_df = pd.DataFrame(shap_values[:,:, 1], columns=X.columns)  # Use class 1
shap_df.to_csv('shap_results/shap_values.csv', index=False)

# Calculate and save feature importance summary
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(shap_values[:,:, 1]).mean(axis=0)  # Use class 1
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
feature_importance.to_csv('shap_results/feature_importance.csv', index=False)

# Save example predictions with SHAP contributions
example_predictions = pd.DataFrame()
example_predictions['true_label'] = y_test
example_predictions['predicted_prob'] = model.predict_proba(X_test)[:, 1]
for col in X.columns:
    example_predictions[f'shap_{col}'] = shap_df[col]
example_predictions.to_csv('shap_results/example_predictions.csv', index=False)

print("SHAP analysis completed. Results saved in 'shap_results' directory:")
print("1. summary_plot.png - Main SHAP summary plot")
print("2. feature_importance_plot.png - Feature importance bar plot")
print("3. shap_values.csv - Raw SHAP values")
print("4. feature_importance.csv - Feature importance summary")
print("5. example_predictions.csv - Example predictions with SHAP contributions")