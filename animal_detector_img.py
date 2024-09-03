import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model('animal_species_detector.h5')

# Define image dimensions
img_width, img_height = 224, 224

# Define the path to the image you want to predict
image_path = 'path_to_your_image.jpg'  # Replace with the path to your image

# Load and preprocess the image
img = image.load_img(image_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Rescale the image

# Predict the class of the image
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

# Get the class labels
class_labels = list(train_generator.class_indices.keys())

# Display the image along with the predicted class
plt.imshow(img)
plt.title(f'Predicted: {class_labels[predicted_class[0]]}')
plt.axis('off')
plt.show()
