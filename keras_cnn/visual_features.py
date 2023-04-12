import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# get the current working directory
cwd = os.getcwd()

# define the file name
file_name = "President_Obama.jpg"

# create the full file path
file_path = os.path.join(cwd, file_name)

# open the image
image = Image.open(file_path)
plt.imshow(image)

# load pretrained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=True)

# review model parameters and architecture
print(vgg_model.summary())

# load and preprocess the image
img = keras_image.load_img(file_path, target_size=(224, 224))
x = keras_image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# get model predictions
predictions = vgg_model.predict(x)

# Get the class index with the highest score
class_idx = np.argmax(predictions[0])

# load the class label index
with open('imagenet_class_index.json', 'r') as f:
    labels = json.load(f)

# Get the class label
label = labels[str(class_idx)]

print("Predicted Class: ", label)

# Extract the feature maps from the VGG16 model
feature_maps = [
    layer.output for layer in vgg_model.layers if 'conv' in layer.name]
feature_maps_model = tensorflow.keras.Model(
    inputs=vgg_model.inputs, outputs=feature_maps)

# Get the feature maps for the input image
maps = feature_maps_model.predict(x)

# visualize the feature maps
fig = plt.figure(figsize=(30, 60))
for i, fmap in enumerate(maps):
    for j in range(fmap.shape[-1]):
        ax = fig.add_subplot(
            len(maps), fmap.shape[-1], i * fmap.shape[-1] + j + 1)
        ax.matshow(fmap[0, :, :, j], cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Layer {i + 1}, Filter {j + 1}', fontsize=30)

# Save the figure
plt.savefig("feature_maps_keras.png", dpi=300, bbox_inches='tight')
plt.show()