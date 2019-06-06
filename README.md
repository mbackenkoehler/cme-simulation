# Stochastic Simulation of SRN models

This is an implementation of stochastic simulation techniques for models
of stochastic reaction networks.

## Dependencies & Compilation
To compile and run this code one needs to install
* the __nightly__ rust toolchain
* Python 3 and Sympy
With these dependencies in place `cargo build --release` should suffice to build
the code.

## Model specification language
Models such as a birth-death process can be modelled as follows.

```
parameters
  a = 10
  b = 0.1

species X

reactions
  0 -> X @ a;
  X -> 0 @ b * X;

init
  X = 0
```

More examples can be found in the `models` directory.

## Usage
The tools options are best documented via the `--help` flag.
