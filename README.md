# About

This is an experiment using Julia to do sequence-based NLP tasks.

## Status

This project is not in a ready state for use. Several tests are set up for forward and backward propagation, but the training loop is not established yet.

## Issues

There are performance problems on the GPU; the forward pass is GPU-optimized and takes ~150 ms, but the backward pass performed either as

```julia
Flux.train!(Loss, θ, [(X, Y)], opt)
```
or as
```julia
grads = Flux.tracker.gradient(() -> Loss(X, Y), θ)
Flux.Tracker.update!(opt, θ, grads)
```
takes >10 seconds to complete on the GPU, compared with ~1.8 seconds on the CPU. Help with this from someone more experienced with Julia, Flux and/or CuArrays would be appreciated.
