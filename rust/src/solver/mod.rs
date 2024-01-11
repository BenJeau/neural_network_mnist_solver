mod mnist;
mod xor;

pub use mnist::MnistSolver;
pub use xor::XorSolver;

#[derive(Debug)]
pub struct SolverData<T> {
    accuracy: f32,
    predictions: Vec<T>,
    labels: Vec<T>,
}

pub trait Solver {
    type Output;

    fn solve(&self, epochs: usize, learning_rate: f32) -> SolverData<Self::Output>;
}
