import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class mini_mnist(keras.utils.Sequence):
        
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.load_mnist()
    
    def load_mnist(self):
        # Load the MNIST dataset
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # create a mini dataset of 1000 each
        mini_train = x_train[:1000]
        mini_train_labels = y_train[:1000]
        mini_val = x_train[1000:2000]
        mini_val_labels = y_train[1000:2000]
        mini_test = x_train[2000:3000]
        mini_test_labels = y_train[2000:3000]
        
        return mini_train, mini_train_labels, mini_val, mini_val_labels, mini_test, mini_test_labels

