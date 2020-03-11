from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

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