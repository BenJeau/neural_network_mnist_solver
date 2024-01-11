from abc import ABC, abstractmethod

class Layer(ABC):
    input_data = None
    output_data = None

    @abstractmethod
    def forward_propagation(self, input_data):
        ...

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        ...
