use nalgebra::{DMatrix, DVector};

use crate::network::layers::Layer;

pub mod layers;
pub mod math;

type Loss = fn(&DMatrix<f32>, &DMatrix<f32>) -> f32;
type LossPrime = fn(&DMatrix<f32>, &DMatrix<f32>) -> DMatrix<f32>;

pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    loss: Loss,
    loss_prime: LossPrime,
}

impl Network {
    pub fn new(loss: Loss, loss_prime: LossPrime) -> Network {
        Network {
            layers: Vec::new(),
            loss,
            loss_prime,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    fn apply_forward_propagation(&mut self, input_data: &DMatrix<f32>) -> DMatrix<f32> {
        let mut output = input_data.clone();
        for layer in &mut self.layers {
            output = layer.forward_propagation(&output).clone();
        }
        output
    }

    fn apply_back_propagation(&mut self, input_data: &DMatrix<f32>, learning_rate: f32) {
        let mut output = input_data.clone();
        for layer in self.layers.iter_mut().rev() {
            output = layer.back_propagation(&output, learning_rate);
        }
    }

    pub fn predict(&mut self, input_data: &DVector<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
        input_data
            .iter()
            .map(|data| self.apply_forward_propagation(data))
            .collect()
    }

    pub fn fit(
        &mut self,
        data: &DVector<DMatrix<f32>>,
        labels: &DVector<DMatrix<f32>>,
        epochs: usize,
        learning_rate: f32,
    ) -> &mut Self {
        let samples = data.len();

        for epoch in 0..epochs {
            let mut err = 0.0;

            for j in 0..samples {
                let output = self.apply_forward_propagation(&data[j]);

                err += (self.loss)(&labels[j], &output);

                self.apply_back_propagation(&(self.loss_prime)(&labels[j], &output), learning_rate);
            }

            err /= samples as f32;
            println!("epoch {}/{}   error={}", epoch + 1, epochs, err);
        }

        self
    }
}
