import json

import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt
import shap
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize(image):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
# load the model

# model = models.vgg16(pretrained=False)  # Initialize without pretrained weights
# Then load the local weights
# state_dict = torch.load('/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/models/vgg16-397923af.pth')
# model.load_state_dict(state_dict)
# model.eval()

# model_weights = '/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/models/vgg16-397923af.pth'
model = models.vgg16(pretrained=True).eval()

X, y = shap.datasets.imagenet50()
X /= 255

to_explain = X[[39, 41]]

# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
# fname = '/home/dell/disk1/Jinlong/Ediformer-SeismicHorizonPicking/models/imagenet_class_index.json'

with open(fname) as f:
    class_names = json.load(f)

e = shap.GradientExplainer((model, model.features[7]), normalize(X))
shap_values, indexes = e.shap_values(normalize(to_explain), ranked_outputs=2, nsamples=200)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]


shap.image_plot(shap_values, to_explain, index_names)
plt.savefig('shap_analysis.png')