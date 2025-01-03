import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Initialize SHAP Explainer (TreeExplainer is suitable for tree-based models)
explainer = shap.TreeExplainer(model)

# 4. Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# 5. Check the shape of the SHAP values
print(f"SHAP values for class 0 shape: {shap_values[0].shape}")
print(f"SHAP values for class 1 shape: {shap_values[1].shape}")

# Both should have shape (300, 10), matching the number of samples and features

# 6. Plot SHAP summary plot for class 1 (or class 0 depending on interest)
shap.summary_plot(shap_values[1], X_test)  # class 1, change to shap_values[0] for class 0

# 7. Store SHAP values for class 1 in a pandas DataFrame for further analysis
shap_values_df = pd.DataFrame(shap_values[1], columns=[f"Feature_{i}" for i in range(X_test.shape[1])])

# 8. Save SHAP results to a CSV file
shap_values_df.to_csv("shap_values_class_1.csv", index=False)

# 9. Show a SHAP dependence plot for a specific feature (e.g., Feature_0)
shap.dependence_plot("Feature_0", shap_values[1], X_test)

# Optionally, you can also save the dependence plot
plt.savefig("shap_dependence_plot_class_1.png")
