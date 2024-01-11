# Neural Network - MNIST and XOR Solver

A simple Python and Rust based Neural Network.

Inspired from https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65. I started understanding the concepts in Python and was curious how it would look in Rust - turns out it wasn't that bad!

## Training and execution of the models

### Python

For Python, simply run the `main.py` file and modify the entrypoint to solve MNIST or XOR alongside with the parameters and hyperparameters, e.g.:

```
cd python
python3 python/main.py
```

### Rust

For Rust, use the CLI provided to select the solver you want to use and provide the parameters and hyperparameters via CLI arguments according to the help command, e.g.:

```
cd rust
cargo run --release -- mnist
```

## Performance

Unsuprisingly, Python doesn't have great performance, but for the MNIST model training, both are similar probably for the following reasons:
  * the Python's implementation relies on numpy (which is heavily optimized)
  * both implementation is single threaded
  * both implementation are simple and naïve - they do not contain batching or other methods to speed up the training process

Both give similar end result/accuracy (since their implementation are similar), but took the `time` to train and execute on my M1 Max Mac:

| Language | XOR                                            | MNIST                                             |
| -------- | ---------------------------------------------- | ------------------------------------------------- |
| Rust     | `0.01s user 0.00s system 86% cpu 0.019 total`  | `28.43s user 0.06s system 93% cpu 30.605 total`   |
| Python   | `3.05s user 1.93s system 190% cpu 2.615 total` | `155.36s user 6.00s system 840% cpu 19.188 total` |
