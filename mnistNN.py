from tensorflow.examples.tutorials.mnist import input_data
from basicNN import train_neural_network

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

train_neural_network([train_x, train_y], [test_x, test_y])