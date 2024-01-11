import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from network import Network
from layers.fully_connected_layer import FullyConnectedLayer
from layers.activation_layer import ActivationLayer
from helpers import tanh, tanh_prime, mse, mse_prime

def solve_xor():
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    network = Network() \
      .add(FullyConnectedLayer(2, 3)) \
      .add(ActivationLayer(tanh, tanh_prime)) \
      .add(FullyConnectedLayer(3, 1)) \
      .add(ActivationLayer(tanh, tanh_prime))

    model = network.use(mse, mse_prime) \
      .fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    print(model.predict(x_train))

def solve_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = to_categorical(y_train)

    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = to_categorical(y_test)

    network = Network() \
      .add(FullyConnectedLayer(28*28, 100)) \
      .add(ActivationLayer(tanh, tanh_prime)) \
      .add(FullyConnectedLayer(100, 50)) \
      .add(ActivationLayer(tanh, tanh_prime)) \
      .add(FullyConnectedLayer(50, 10)) \
      .add(ActivationLayer(tanh, tanh_prime))

    model = network.use(mse, mse_prime) \
      .fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

    out = model.predict(x_test[0:3])
    print(f"predicted values: {out}")
    print(f"true values: {y_test[0:3]}")

if __name__ == '__main__':
    # solve_xor()
    solve_mnist()