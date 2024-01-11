use nalgebra::{DMatrix, DVector};

use crate::{
    dataset::mnist,
    network::{
        layers::{ActivationLayer, FullyConnectedLayer},
        Network,math
    },
    solver::SolverData,
};

use super::Solver;

pub struct MnistSolver<'a> {
    pub base_url: &'a str,
    pub train_size: usize,
    pub test_size: usize,
}

impl MnistSolver<'_> {
    pub fn new<'a>(base_url: &'a str, train_size: usize, test_size: usize) -> MnistSolver<'a> {
        MnistSolver {
            base_url,
            train_size,
            test_size,
        }
    }
}

impl Solver for MnistSolver<'_> {
    type Output = u8;

    fn solve(&self, epochs: usize, learning_rate: f32) -> SolverData<Self::Output> {
        let dataset = mnist::MnistDataset::from_url(self.base_url).unwrap();

        let train_data = mnist_image_data(&dataset.train.images.images[0..self.train_size]);
        let train_labels = mnist_to_categorical(&dataset.train.labels.0[0..self.train_size]);

        let test_data = mnist_image_data(&dataset.test.images.images[0..self.test_size]);
        let test_labels = &dataset.test.labels.0[0..self.test_size];

        let predictions = Network::new(math::mse, math::mse_prime)
            .add(FullyConnectedLayer::new(28 * 28, 100))
            .add(ActivationLayer::new(math::tanh, math::tanh_prime))
            .add(FullyConnectedLayer::new(100, 50))
            .add(ActivationLayer::new(math::tanh, math::tanh_prime))
            .add(FullyConnectedLayer::new(50, 10))
            .add(ActivationLayer::new(math::tanh, math::tanh_prime))
            .fit(&train_data, &train_labels, epochs, learning_rate)
            .predict(&test_data);

        let predictions = categorical_to_mist_labels(&predictions);

        let accuracy = predictions
            .iter()
            .zip(test_labels.iter())
            .filter(|(a, b)| a == b)
            .count() as f32
            / predictions.len() as f32;

        SolverData {
            accuracy,
            labels: test_labels.to_vec(),
            predictions,
        }
    }
}

fn mnist_image_data(image: &[Vec<u8>]) -> DVector<DMatrix<f32>> {
    DVector::from_vec(
        image
            .iter()
            .map(|image| {
                DMatrix::from_row_slice(
                    1,
                    28 * 28,
                    &image
                        .iter()
                        .map(|pixel| *pixel as f32 / 255.0)
                        .collect::<Vec<f32>>(),
                )
            })
            .collect(),
    )
}

fn mnist_to_categorical(labels: &[u8]) -> DVector<DMatrix<f32>> {
    DVector::from_vec(
        labels
            .iter()
            .map(|label| {
                let mut matrix = DMatrix::zeros(1, 10);
                matrix[(0, *label as usize)] = 1.0;
                matrix
            })
            .collect(),
    )
}

fn categorical_to_mist_labels(labels: &[DMatrix<f32>]) -> Vec<u8> {
    labels
        .iter()
        .map(|label| {
            label
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0 as u8
        })
        .collect()
}
