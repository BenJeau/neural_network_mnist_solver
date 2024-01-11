use nalgebra::{DMatrix, DVector};

use crate::{
    network::{
        layers::{ActivationLayer, FullyConnectedLayer},
        math, Network,
    },
    solver::{Solver, SolverData},
};

pub struct XorSolver;

impl Solver for XorSolver {
    type Output = u8;

    fn solve(&self, epochs: usize, learning_rate: f32) -> SolverData<Self::Output> {
        let raw_data = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let raw_labels = vec![0, 1, 1, 0];

        let data = DVector::from_vec(
            raw_data
                .iter()
                .map(|data| {
                    DMatrix::from_row_slice(
                        1,
                        2,
                        data.into_iter()
                            .map(|x| *x as f32)
                            .collect::<Vec<f32>>()
                            .as_slice(),
                    )
                })
                .collect(),
        );

        let labels = DVector::from_vec(
            raw_labels
                .iter()
                .map(|label| DMatrix::from_row_slice(1, 1, &[*label as f32]))
                .collect(),
        );

        let predictions = Network::new(math::mse, math::mse_prime)
            .add(FullyConnectedLayer::new(2, 3))
            .add(ActivationLayer::new(math::tanh, math::tanh_prime))
            .add(FullyConnectedLayer::new(3, 1))
            .add(ActivationLayer::new(math::tanh, math::tanh_prime))
            .fit(&data, &labels, epochs, learning_rate)
            .predict(&data);

        let predictions = categorical_to_labels(&predictions);

        let accuracy = predictions
            .iter()
            .zip(raw_labels.iter())
            .filter(|(a, b)| a == b)
            .count() as f32
            / predictions.len() as f32;

        SolverData {
            accuracy,
            predictions,
            labels: raw_labels,
        }
    }
}

fn categorical_to_labels(data: &Vec<DMatrix<f32>>) -> Vec<u8> {
    data.iter().map(|data| (data[(0, 0)] > 0.5) as u8).collect()
}
