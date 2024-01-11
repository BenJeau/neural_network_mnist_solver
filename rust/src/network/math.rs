use nalgebra::DMatrix;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;

pub fn rand_matrix(d0: usize, d1: usize) -> DMatrix<f32> {
    let mut rng = thread_rng();
    let range = Uniform::new(-0.5, 0.5);

    DMatrix::from_fn(d0, d1, |_, _| range.sample(&mut rng))
}

pub fn tanh(x: DMatrix<f32>) -> DMatrix<f32> {
    x.map(|x| x.tanh())
}

pub fn tanh_prime(x: DMatrix<f32>) -> DMatrix<f32> {
    x.map(|x| 1.0 - x.tanh().powi(2))
}

pub fn mse(y_true: &DMatrix<f32>, y_pred: &DMatrix<f32>) -> f32 {
    (y_true - y_pred).map(|x| x.powi(2)).mean()
}

pub fn mse_prime(y_true: &DMatrix<f32>, y_pred: &DMatrix<f32>) -> DMatrix<f32> {
    2.0 * (y_pred - y_true) / y_true.len() as f32
}
