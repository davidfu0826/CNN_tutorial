from skimage import io, color
import matplotlib.pyplot as plt
from skimage import exposure
import pylab
import skimage.measure
import numpy as np

import pandas as pd
import seaborn as sns

# Dimensionality reduction
from sklearn.decomposition import PCA

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.utils import to_categorical

# Displaying images
def visualize_dataset(images, labels, label_to_article):
  fig, axs = plt.subplots(2, 5, figsize = (16, 7))
  for i in range(10):
    grid_index = (i//5, i%5)

    index = np.where(labels==i)[0][0]
    image = images[index]
    axs[grid_index].imshow(image/255., cmap=plt.cm.gray)
    title = f"Article:  {label_to_article[labels[index]]}\n" + \
            f"Label:  {i}"
    axs[grid_index].set_title(title)
    axs[grid_index].axis('off')
  plt.show()

# Loading image
def load_example_image(url):
  image = io.imread(url)    # Load the image
  image = color.rgb2gray(image)       # Convert the image to grayscale (1 channel)
  return image

def visualize_convolution(original_image, kernel):
  fig, axs = plt.subplots(1, 2, figsize = (14, 8))

  # Original image
  axs[0].imshow(original_image, cmap=plt.cm.gray)
  axs[0].set_title("Original image")
  axs[0].axis('off')

  # Convoluted image
  convolved_image = convolve2d(original_image, kernel)
  convolved_image = skimage.measure.block_reduce(convolved_image, (2,2), np.max) # Maxpooling
  convolved_image = exposure.equalize_adapthist(convolved_image/np.max(np.abs(convolved_image)), clip_limit=0.03)
  axs[1].imshow(convolved_image, cmap=plt.cm.gray)
  axs[1].set_title("Convolved image")
  axs[1].axis('off')

def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+kernel.shape[1],x:x+kernel.shape[0]]).sum()        
    return output

def scatter_plot(images, labels, label_to_article, title='PCA of Fasion-MNIST Dataset', nbr_samples=400):
  fig = plt.figure(figsize=(14, 10))
  fig.suptitle('PCA of Fasion-MNIST Dataset', fontsize=40)

  for i in range(10):
    # Select a subset of the images
    indices = np.where(train_labels == i)[0][:nbr_samples]

    # Display images in a 2D grid
    plt.scatter(pca_train_images[indices][:,0], pca_train_images[indices][:,1])
  plt.legend([label_to_article[i] for i in range(10)], prop={'size': 16});

def display_PCA(train_images, train_labels, nbr_points, label_to_article):
  # Dimensionality reduction 
  pca = PCA(n_components=2)
  pca_train_images = pca.fit_transform(np.reshape(train_images/255., [-1, 28*28]))

  fig = plt.figure(figsize=(14, 10))
  fig.suptitle('PCA of Fasion-MNIST Dataset', fontsize=40)

  for i in range(10):
    # Select a subset of the images
    indices = np.where(train_labels == i)[0][:nbr_points]

    # Display images in a 2D grid
    plt.scatter(pca_train_images[indices][:,0], pca_train_images[indices][:,1])
  plt.legend([label_to_article[i] for i in range(10)], prop={'size': 16});
  
def build_ANN(nbr_nodes=[32], dropout=True):
  # Keras Functional API

  # Input layer
  inputs = Input(shape=(28*28,))
 
  x = inputs
  # Fully-connected layers
  for nbr_node in nbr_nodes:
    x = Dense(nbr_node, activation="relu")(x)
    x = Dropout(0.3)(x)

  # Output Layer
  outputs = Dense(10, activation="softmax")(x)
  
  
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
  
  return model

def build_CNN(nbr_filters=[64, 64], kernel_shape=(3, 3), nbr_nodes=[32], dropout=True):
  # Keras Functional API

  # Input layer
  inputs = Input(shape=(28, 28, 1))

  # Convolutional base
  x = inputs
  for nbr_filter in nbr_filters:
    x = Conv2D(nbr_filter, kernel_shape, activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)

  # Fully-connected layers
  x = Flatten()(x)
  for nbr_node in nbr_nodes:
    x = Dense(nbr_node, activation="relu")(x)
    x = Dropout(0.3)(x)

  # Output Layer
  outputs = Dense(10, activation="softmax")(x)
  
  
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
  
  return model
  
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
  
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


def prepare_NN_dataset(images, labels):
  # Images are reshapen from (-1, 28, 28) to (-1, 28, 28, 1) in acc. with Keras API
  new_shape = (-1, 28*28)

  X = np.reshape(images, new_shape)
  y = to_categorical(labels)

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
