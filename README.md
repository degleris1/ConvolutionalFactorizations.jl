# ConvolutionalFactorizations.jl


## Installation

To install this package and add it to your primary Julia development environment, run:

```
$ git clone https://github.com/degleris1/ConvolutionalFactorizations.jl.git
$ cd ConvolutionalFactorizations.jl
$ julia
```

From the Julia REPL, press `]` to enter package mode, and run:

```julia
pkg> dev .
```

Then press `[delete]` to exit package mode.
Alternatively, if you're not a developer, all you need to do is open up the Julia REPL, press `]` to enter package mode, and run:

```julia
pkg> add https://github.com/degleris1/ConvolutionalFactorizations.jl.git
```

## Style Guide

- One line between functions
- Use the following to delimit different sections of a file:
```
###
# SECTION NAME
###
```
- 4 lines to separate a section



## TODO

**Minor**

- [ ] Performance optimize `MaskedLoss`.
- [ ] Performance optimize penalty functions and constraints.


- In the long term, it would be nice to have an `AbstractTensor` class with methods like `get_slice(W::AbstractTensor, lag::Int)` and `get_factor(W::AbstractTensor, k::Int)`. The reason this would be helpful is that different algorithms perform better with different memory layouts, so dispatching on the memory layout could greatly improve performance.