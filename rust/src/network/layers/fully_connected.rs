use nalgebra::DMatrix;

use crate::network::{layers::Layer, math::rand_matrix};

pub struct FullyConnectedLayer {
    input: DMatrix<f32>,
    output: DMatrix<f32>,
    weights: DMatrix<f32>,
    biases: DMatrix<f32>,
}

impl FullyConnectedLayer {
    pub fn new(input_size: usize, output_size: usize) -> Box<FullyConnectedLayer> {
        Box::new(Self {
            weights: rand_matrix(input_size, output_size),
            biases: rand_matrix(1, output_size),
            input: DMatrix::zeros(0, 0),
            output: DMatrix::zeros(0, 0),
        })
    }
}

impl Layer for FullyConnectedLayer {
    fn forward_propagation(&mut self, input_data: &DMatrix<f32>) -> DMatrix<f32> {
        self.input = input_data.clone();
        self.output = &self.input * &self.weights + &self.biases;
        self.output.clone()
    }

    fn back_propagation(
        &mut self,
        output_error: &DMatrix<f32>,
        learning_rate: f32,
    ) -> DMatrix<f32> {
        let input_error = output_error * self.weights.transpose();
        let weights_error = self.input.transpose() * output_error;

        self.weights -= learning_rate * weights_error;
        self.biases -= learning_rate * output_error;
        input_error
    }
}
