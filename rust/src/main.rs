mod dataset;
mod error;
mod network;
mod solver;

use clap::{Parser, Subcommand};
use solver::Solver;

#[derive(Parser)]
struct Args {
    /// The type of solver to run
    #[clap(subcommand)]
    solver_type: SolverType,
}

#[derive(Subcommand)]
enum SolverType {
    /// Solves the MNIST dataset
    Mnist {
        /// The number of training samples to use
        #[clap(long, default_value = "1000")]
        train_size: usize,
        /// The number of testing samples to use
        #[clap(long, default_value = "3")]
        test_size: usize,
        /// The base URL to download the MNIST dataset from
        #[clap(short, long, default_value = dataset::mnist::BASE_URL)]
        base_url: String,
        /// The number of model iterations to run
        #[clap(short, long, default_value = "35")]
        epochs: usize,
        /// The rate at which the model learns
        #[clap(short, long, default_value = "0.1")]
        learning_rate: f32,
    },
    /// Solves the XOR dataset
    Xor {
        /// The number of model iterations to run
        #[clap(short, long, default_value = "1000")]
        epochs: usize,
        /// The rate at which the model learns
        #[clap(short, long, default_value = "0.1")]
        learning_rate: f32,
    },
}

impl Args {
    fn run_solver(&self) {
        match &self.solver_type {
            SolverType::Mnist {
                train_size,
                test_size,
                base_url,
                epochs,
                learning_rate,
            } => {
                let data = solver::MnistSolver::new(&base_url, *train_size, *test_size)
                    .solve(*epochs, *learning_rate);

                println!("data = {:?}", data);
            }
            SolverType::Xor {
                epochs,
                learning_rate,
            } => {
                let data = solver::XorSolver.solve(*epochs, *learning_rate);

                println!("data = {:?}", data);
            }
        }
    }
}

fn main() {
    Args::parse().run_solver()
}
