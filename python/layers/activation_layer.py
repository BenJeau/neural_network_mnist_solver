from .layer import Layer

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input_data = input_data
        self.output_data = self.activation(self.input_data)
        return self.output_data

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input_data) * output_error
