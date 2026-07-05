# The numerical simulation of Two-Dimensional Decaying Magnetohydrodynamic Turbulence in Astrophysical Plasma

## About the project

This is two-dimentional plasma simulation project. 

The physical model describes two-dimensional decaying magnetohydrodynamic turbulence in astrophysical plasma. Such turbulence allows understanding the evolution of various astrophysical objects from the Sun and stars to planetary systems, galaxies, and galaxy clusters. 

For mathimatical description I use the equations of two-dimensional magnetohydrodynamics without compressibility which can be divided on the Navier–Stokes equations with the Lorentz force and the magnetic induction behavior equation. 
Mathimatical aplications can be found in my artical about such turbulence on a β-Plane.

For the numerical solution of the system of equations, I use a pseudospectral method using the 2/3 rule to eliminate aliasing, i.e. when using a grid in the coordinate space, the Fourier space grid will be limited to a square region. As an initial condition, I take a set of Fourier harmonics with random phases in a ring. The initial kinetic and magnetic energies are uniformly distributed over the Fourier harmonics.

The numerical model is implemented in the C++ programming language and used the following tools:
* CUDA technology and the CUDA C++ language dialect to speed up calculations;
* OpenGL API for computer graphics and visualization of plasma behavior
* YAML parser and emitter in C++ (`yaml-cpp`)  to set up configuration of physical coefficient, numerical simulation parameters, output parameters and OpenGL ilustration parameters.

## Getting Started

This is an example of how you may set up the project locally. To get a local copy up and running follow these simple example steps.

### Prerequisites
* CUDA Toolkit 12.6 or newer and Nvidia Drivers
* OpenGL API 
* CMake 3.27 or newer

### Installation
Navigate into the source directory and configure the project into the `build` folder:
```sh
cmake -S . -B build
```
By default the code is compiled for the GPU of the current machine. To target a specific GPU architecture pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` (for example `86`).

Build the project:
```sh
cmake --build build --config Release
```

### Usage
Navigate into the binary directory and run the simulation:
```sh
cd build\bin
.\simulation
```
Optionally pass the name of a configuration file from the `configs` folder (no full path needed):
```sh
.\simulation fluid1024.yaml
```

## Screen Recording of default simulation behavior

https://github.com/user-attachments/assets/f174cf1d-ffc8-4d73-970c-da356a865390

The video was sped up twice to reduce size of the file
