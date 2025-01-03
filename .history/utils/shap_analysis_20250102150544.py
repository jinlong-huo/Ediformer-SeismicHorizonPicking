# import json

# import numpy as np
# import torch
# from torchvision import models
# import matplotlib.pyplot as plt
# import shap
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]


# def normalize(image):
#     if image.max() > 1:
#         image /= 255
#     image = (image - mean) / std
#     # in addition, roll the axis so that they suit pytorch
#     return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
# # load the model

# # model = models.vgg16(pretrained=False)  # Initialize without pretrained weights
# # Then load the local weights
# # state_dict = torch.load('/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/models/vgg16-397923af.pth')
# # model.load_state_dict(state_dict)
# # model.eval()

# # model_weights = '/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/models/vgg16-397923af.pth'
# model = models.vgg16(pretrained=True).eval()

# X, y = shap.datasets.imagenet50()
# X /= 255

# to_explain = X[[39, 41]]

# # load the ImageNet class names
# url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
# fname = shap.datasets.cache(url)
# # fname = '/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/models/imagenet_class_index.json'

# with open(fname) as f:
#     class_names = json.load(f)

# e = shap.GradientExplainer((model, model.features[7]), normalize(X))
# shap_values, indexes = e.shap_values(normalize(to_explain), ranked_outputs=2, nsamples=200)

# # get the names for the classes
# index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# # plot the explanations
# shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

# shap.image_plot(shap_values, to_explain, index_names)
# plt.savefig('shap_analysis.png')

import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pickle

# 1. Load dataset (for this example, using the Iris dataset)
data = load_iris()
X = data.data
y = data.target

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train a simple decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Use SHAP to explain the model's predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 5. Visualize the SHAP values for the first instance in the test set
shap.initjs()  # Initialize the SHAP JS visualization (optional, only for Jupyter or IPython)
shap.summary_plot(shap_values[0], X_test, feature_names=data.feature_names)

# 6. Save the SHAP values to a local file
shap_values_file = "shap_values.pkl"
with open(shap_values_file, 'wb') as f:
    pickle.dump(shap_values, f)

# 7. Optionally, you can save a plot to an image file
summary_plot_file = "shap_summary_plot.png"
plt.savefig(summary_plot_file)

# Print confirmation of saved files
print(f"SHAP values saved to {shap_values_file}")
print(f"SHAP summary plot saved to {summary_plot_file}")
