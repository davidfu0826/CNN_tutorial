import numpy as np
from tensorflow.keras.utils import to_categorical

# Function for preparing data
def prepare_binary_dataset(images, labels, first_label, second_label):
  # Select only the relevant images for training
  first_indices = np.where(labels == first_label)
  second_indices = np.where(labels == second_label)
  indices = np.hstack([first_indices, second_indices])[0]

  # All images are flattened to 1D-arrays
  new_shape = (-1, 28*28) 
  X = np.reshape(images[indices], new_shape)

  y = labels[indices] == first_label

  return X, y

def prepare_ANN_dataset(images, labels):
  # Images are reshapen from (-1, 28, 28) to (-1, 28, 28, 1) in acc. with Keras API
  new_shape = (-1, 28*28)

  X = np.reshape(images, new_shape)
  y = to_categorical(labels)

  return X, y

def prepare_CNN_dataset(images, labels):
  # Images are reshapen from (-1, 28, 28) to (-1, 28, 28, 1) in acc. with Keras API
  new_shape = (-1, 28, 28, 1)

  X = np.reshape(images, new_shape)
  y = to_categorical(labels)

  return X, y