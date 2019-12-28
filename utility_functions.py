from skimage import io, color
import matplotlib.pyplot as plt
from skimage import exposure
import pylab
import skimage.measure
import numpy as np

# Put these functions in to
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
