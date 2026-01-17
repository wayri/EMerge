![GitHub License](https://img.shields.io/github/license/FennisRobert/EMerge)
![GitHub Release](https://img.shields.io/github/v/release/FennisRobert/EMerge)
[![PySimHub](https://pysimhub.io/badge.svg)](https://pysimhub.io/projects/emerge)
![PyPI - Downloads](https://img.shields.io/pypi/dw/emerge)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17025518-blue)](https://doi.org/10.5281/zenodo.17025518)

## Introduction
![LinkedInCover](https://github.com/user-attachments/assets/d9c194c8-eb34-49e5-96a6-9dfdfc787563)

Hello everybody. Thanks for showing interest in my EM FEM library!

EMerge is a python based FEM EM library for the time harmonic helmholtz formulation. It is thus best suited for
Electromagnetic wave phenomenon. You can use it to simulate:
 - RF Filters
 - Signal propagation through PCBs
 - Antennas
 - Optycal systems
 - Arrays and periodic structures
 - Much more!

It is designed to be as easy to use and compatible as possible. It runs on all operating systems allthough some solvers
are a bit harder to make work on some systems than others.

EMerge is designed to have your entire simulation start and finish in the same Python script (or more if you want).
You require no awkward configuration files, JSON's, external software to do modelling etc. It allows you to do everything
in Python:
 * Geometry Creation/description
 * Material assignment
 * Meshing + mesh settings and adaptive mesh refinement
 * Boundary condition setup
 * Solving
 * Post processing and visualization

If you have questions, suggestions, bug reports or just want to hang out, feel free to join the discord!

**[Discord Invitation](https://discord.gg/VMftDCZcNz)**

## How to install

You can now install the basic version of emerge from PyPi!

```
pip install emerge
```

## Direct solvers
EMerge solves all systems with direct solvers only. Some are faster than others. Depending on the operating 
system and hardware you have, some might work better and/or are easier to install than others.

### MacOS (ARM)
#### Single threaded UMFPACK
On MacOS you can install it with the very fast UMFPACK through scikit-umfpack

```
brew install cmake swig suite-sparse #MacOS
sudo apt-get install libsuitesparse-dev #Linux
```
Then on MacOS do:
```
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export CFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"
```
Finally:
```
pip install emerge[umfpack]
```
#### Multi threaded MUMPS
To install the MUMPS solver on MacOS, download the installer directory from my website and follow the instructions:

https://www.emerge-software.com/resources

### Windows (x86)
Windows has easy access to the lightning fast PARDISO solver out of the box, no installation needed.
If you want to install the UMFPACK solver for distributed sweeps, please follow the instructions for the UMFPACK
installation manual on my website

https://www.emerge-software.com/resources

### GPU bsed CuDSS solver
If you have a new NVidia card you can try the first test implementation of the cuDSS solver. The dependencies can be installed through:
```
pip install emerge[cudss]
```
*Limitations: * Cupy is currently only supporting 32 bit integer address so large EM problems cannot be correctly solved currently. This is not something I can do anything about.*

## Required libraries

To run this FEM library you need the following libraries

 - numpy
 - scipy
 - gmsh
 - loguru
 - numba
 - matplotlib (for the matplotlib base display)
 - pyvista (for the PyVista base display)
 - mkl (x86 devices only)
 - emsutil
 - 
Optional:
 - scikit-umfpack
 - cudss
 - ezdxf

## Resources / Manual

You can find the latest versions of the manual on: **https://www.emerge-software.com/resources/**
