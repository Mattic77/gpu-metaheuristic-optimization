# GPU Metaheuristic Optimization

CUDA implementations of population-based metaheuristic optimization algorithms,
with a focus on running the main computation on NVIDIA GPUs.

## Contents

- `kernel.cu`, `kernel.cpp`, `kernel.h`, and `main.cpp`: a CUDA
  implementation of Particle Swarm Optimization (PSO).
- `cudaDE_I/`: a CUDA implementation of Differential Evolution (DE/rand/1/bin)
  with GPU-side population initialization, mutation, crossover, evaluation, and
  replacement.
- `CudaPSO.ipynb`: an experimental notebook for running the PSO implementation.
- `ANALYSE_PROJET.md`: a technical analysis of the project architecture and
  algorithms.
- `An_Improved_CUDA-Based-Implementation_of_Different.pdf`: the reference
  article used for the Differential Evolution implementation.

The objective functions currently include Levy, shifted Rastrigin, shifted
Rosenbrock, shifted Griewank, and shifted Sphere functions. The selected
objective function and algorithm parameters are defined in each algorithm's
`kernel.h`.

## Requirements

- NVIDIA GPU with a compatible CUDA toolkit
- CUDA compiler (`nvcc`)
- A C++ compiler supported by the installed CUDA version

## Building and running PSO

From the repository root:

```powershell
nvcc -o cuda_pso.exe main.cpp kernel.cpp kernel.cu
.\cuda_pso.exe
```

## Building and running Differential Evolution

From the repository root:

```powershell
nvcc -o cuda_de.exe cudaDE_I\main.cpp cudaDE_I\kernel.cpp cudaDE_I\kernel.cu
.\cuda_de.exe
```

The programs print the GPU execution time and the best objective value found.
Random initialization means that results can differ between executions.

## Project status

This repository is an academic and experimental project intended for studying
parallel metaheuristic optimization with CUDA. Performance and numerical
results may depend on the GPU, CUDA version, compiler, and selected parameters.
