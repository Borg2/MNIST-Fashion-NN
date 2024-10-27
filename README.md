# **Fashion MNIST Clothing Classification with Keras**

This project implements a neural network built using Keras' Sequential API to classify clothing items in the Fashion MNIST dataset.

## **Dataset**

The Fashion MNIST dataset consists of 70,000 grayscale images (28x28 pixels) belonging to 10 different clothing categories:

* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

The dataset is split into 60,000 training images and 10,000 testing images.

## **Model Architecture**

The neural network architecture leverages a series of fully connected layers to process the flattened image data and predict clothing labels. Here's a breakdown of the model's layers:

* **Flatten Layer:** Flattens the 28x28 pixel images into a 1D vector of 784 elements.
* **Dense Layers:**
    * Two Dense layers with 256 neurons each, followed by ReLU activation for non-linearity.
    * Another Dense layer with 128 neurons and ReLU activation.
    * A final Dense layer with 64 neurons and ReLU activation.
    * Dropout layers with a rate of 0.2 are added after each Dense layer to prevent overfitting.
* **Output Layer:** Dense layer with 10 neurons and Softmax activation for multi-class classification (one neuron per clothing category).

## **Code Highlights**

The `scheduler` function implements a learning rate schedule that reduces the learning rate by a factor of 0.1 after 15 epochs. This helps to fine-tune the training process and potentially improve convergence.

The model is compiled with the Adam optimizer, categorical crossentropy loss function (suitable for multi-class classification), and accuracy metric.

## **Training**

The model is trained for 40 epochs using a batch size of 32 and early stopping with validation data (`(x_val, y_val)`) to prevent overfitting.

## **Improvements**

* Consider using convolutional layers instead of fully connected layers to capture spatial features in the images.
* Experiment with different hyperparameters (learning rate, number of layers, etc.) to potentially improve performance.
* Utilize techniques like data augmentation to increase training data diversity.



