use nalgebra::DMatrix;

mod activation;
mod fully_connected;

pub use activation::ActivationLayer;
pub use fully_connected::FullyConnectedLayer;

pub trait Layer {
    fn forward_propagation(&mut self, input_data: &DMatrix<f32>) -> DMatrix<f32>;
    fn back_propagation(&mut self, output_error: &DMatrix<f32>, learning_rate: f32)
        -> DMatrix<f32>;
}
