use nalgebra::DMatrix;

use crate::network::layers::Layer;

type ActivationFunction = fn(DMatrix<f32>) -> DMatrix<f32>;
type ActivationFunctionPrime = fn(DMatrix<f32>) -> DMatrix<f32>;

pub struct ActivationLayer {
    input: DMatrix<f32>,
    output: DMatrix<f32>,
    activation: ActivationFunction,
    activation_prime: ActivationFunctionPrime,
}

impl ActivationLayer {
    pub fn new(
        activation: ActivationFunction,
        activation_prime: ActivationFunctionPrime,
    ) -> Box<ActivationLayer> {
        Box::new(Self {
            input: DMatrix::zeros(0, 0),
            output: DMatrix::zeros(0, 0),
            activation,
            activation_prime,
        })
    }
}

impl Layer for ActivationLayer {
    fn forward_propagation(&mut self, input_data: &DMatrix<f32>) -> DMatrix<f32> {
        self.input = input_data.clone();
        self.output = (self.activation)(self.input.clone());
        self.output.clone()
    }

    fn back_propagation(
        &mut self,
        output_error: &DMatrix<f32>,
        _learning_rate: f32,
    ) -> DMatrix<f32> {
        let result = (self.activation_prime)(self.input.clone());
        result.component_mul(output_error)
    }
}
