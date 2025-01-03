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

# 5. Plot SHAP summary plot
shap.summary_plot(shap_values[1], X_test)  # Index 1 corresponds to class 1

# 6. Store SHAP values in a pandas DataFrame for further analysis
shap_values_df = pd.DataFrame(shap_values[1], columns=[f"Feature_{i}" for i in range(X_test.shape[1])])

# 7. Save SHAP results to a CSV file
shap_values_df.to_csv("shap_values.csv", index=False)

# 8. Show a SHAP dependence plot for a specific feature (e.g., Feature_0)
shap.dependence_plot("Feature_0", shap_values[1], X_test)

# Optionally, you can also save the dependence plot
plt.savefig("shap_dependence_plot.png")
