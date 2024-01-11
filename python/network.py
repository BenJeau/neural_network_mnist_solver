from typing import Callable, Optional

from layers.layer import Layer

class Network:
    layers: list[Layer] = []
    loss: Optional[Callable] = None
    loss_prime: Optional[Callable] = None

    def add(self, layer: Layer) -> "Network":
        self.layers.append(layer)
        return self

    def use(self, loss: Callable, loss_prime: Callable) -> "Network":
        self.loss = loss
        self.loss_prime = loss_prime
        return self

    def apply_layers(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward_propagation(output)
        return output

    def predict(self, input_data):
        return [self.apply_layers(data) for data in input_data]

    def fit(self, x_train, y_train, epochs, learning_rate) -> "Network":
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

        return self
