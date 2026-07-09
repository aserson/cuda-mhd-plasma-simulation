# The numerical simulation of Two-Dimensional Magnetohydrodynamic Turbulence in Astrophysical Plasma

## About the project

This is a two-dimensional plasma simulation project.

The physical model describes two-dimensional magnetohydrodynamic turbulence in astrophysical plasma, both decaying and sustained by an external energy forcing. Such turbulence allows understanding the evolution of various astrophysical objects from the Sun and stars to planetary systems, galaxies, and galaxy clusters.

For the mathematical description I use the equations of two-dimensional incompressible magnetohydrodynamics, which can be divided into the Navier–Stokes equations with the Lorentz force and the magnetic induction equation. The model optionally includes:
* the beta-plane approximation (the `beta` parameter): the planetary vorticity gradient adds the Rossby wave term to the vorticity equation, which models the large-scale rotation;
* an external energy pumping (the `KineticForcing` and `MagneticForcing` parameters): a random-phase forcing on the wavenumber ring `ForcingWN ± ForcingBand` sustains the turbulence against dissipation.

Mathematical applications can be found in my article about such turbulence on a β-Plane.

For the numerical solution of the system of equations, I use a pseudospectral method with the 2/3 rule to eliminate aliasing, i.e. when using a grid in the coordinate space, the Fourier space grid will be limited to a square region. As an initial condition, I take a set of Fourier harmonics with random phases in a ring. The initial kinetic and magnetic energies are uniformly distributed over the Fourier harmonics.

The numerical model is implemented in the C++ programming language and uses the following tools:
* CUDA technology and the CUDA C++ language dialect to speed up calculations;
* Qt 6 for the interactive user interface: the rendered field and the energy spectra on the left, all simulation settings on the right;
* OpenGL API for visualization in the fallback build without Qt;
* YAML parser and emitter in C++ (`yaml-cpp`) to set up configuration of physical coefficients, numerical simulation parameters, output parameters and illustration parameters.

## Getting Started

This is an example of how you may set up the project locally. To get a local copy up and running follow these simple example steps.

### Prerequisites
* CUDA Toolkit 12.6 or newer and Nvidia Drivers
* CMake 3.27 or newer
* Qt 6 (the Widgets module) — optional, for the interactive UI
* OpenGL API — used by the fallback build without Qt

### Installation
Navigate into the source directory and configure the project into the `build` folder:
```sh
cmake -S . -B build
```
By default the code is compiled for the GPU of the current machine. To target a specific GPU architecture pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` (for example `86`).

On Windows the build looks for Qt under `C:/Qt` automatically (pass `-DQt6_DIR=<path>/lib/cmake/Qt6` for a custom location) and deploys the Qt runtime next to the executable. Without Qt the plain OpenGL window is built instead.

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

With Qt the application opens a window with the simulation view on the left (the `Field` tab shows the chosen field, the `Spectra` tab shows the log-log kinetic and magnetic energy spectra) and the settings panel on the right. No configuration file is needed: the panel starts from the default parameters, `Load config...` fills it from a YAML file (the files in the `configs` folder work as presets), and `Start`/`Restart` launches a run with the current settings. `Pause` freezes the integration and `Stop` ends the run.

In the build without Qt the parameters are set by a configuration file from the `configs` folder, optionally passed by name (this mode also suits scripted headless runs with `ShowGraphics: false`):
```sh
.\simulation fluid1024.yaml
```

### Main configuration parameters

| Key | Meaning | Default |
| --- | --- | --- |
| `GridLength` | Grid size (a power of two) | `1024` |
| `Precision` | `double` or `float` | `double` |
| `Time` | End time of the simulation | `5` |
| `CFL`, `MaxTimeStep` | Time step control | `0.2`, `0.01` |
| `nu`, `eta` | Viscosity and magnetic diffusivity | `1e-4` |
| `beta` | Planetary vorticity gradient of the beta-plane approximation, `0` disables the rotation | `0` |
| `KineticEnergy`, `MagneticEnergy`, `AverageWN` | Initial condition: energies and the spectrum peak wavenumber | `0.5`, `0.5`, `10` |
| `KineticForcing`, `MagneticForcing` | Amplitudes of the external energy pumping, `0` disables the channel | `0` |
| `ForcingWN`, `ForcingBand` | The forced wavenumber ring `ForcingWN ± ForcingBand` | `10`, `1.5` |
| `OutputStep` | Time between the outputs (display frames and saved data) | `0.01` |
| `SaveData`, `SaveVorticity`, `SaveCurrent`, `SaveStream`, `SavePotential` | Writing the fields to the `outputs` folder; the first enabled field is displayed | `false`, `true`, `false`, `false`, `false` |
| `ColorMap` | A color map name from `res/colormaps` | `winter` |

## Screen Recording of the application

<!-- To embed the video with the GitHub player: edit README.md on github.com
     and drag res/mhd_plasma_example_github.mp4 into the editor right below
     this comment; GitHub replaces it with a user-attachments URL -->

The recording shows the Qt interface during a forced turbulence run: the
vorticity field, the energy spectra view and the settings panel.
