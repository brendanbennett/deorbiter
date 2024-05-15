# Satellite Deorbit Simulator and Predictor Tutorial

This package is designed to simulate the trajectory of a deorbiting satellite around Earth and predict its impact location using radar station measurements and an Extended Kalman Filter.The package consists of 3 primary modules:
The Simulator provides the functionality to allow for easy dynamic simulations of a deorbiting satellite.
The Observer allows the user to configure and simulate measurements from this simulation using radar stations positioned on Earth.
The Predictor uses an Extended Kalman Filter with these simulated measurements. It predicts the trajectory and final impact location of the satellite using the radar measurement data. 

The following document outlines the setup and usage guide for the package.

## Setup

### Installing Python

This package requires `Python=3.10`.

To check your Python version with a Windows OS:
Open command prompt by typing 'cmd' into the search bar.
Type 'python' and press enter.
This should display the Python version currently installed. 
If it is not recognised, you can download Python from this source:
https://www.python.org/downloads/

### Creating a New Conda Environment (RECOMMENDED)

Before installing this package it is recommended to create a new Conda environment (see "Installing Conda" below). Using a new environment before installing the required packages will ensure there are no conflicts with your current packages installed.

Open your command prompt and enter:
'''conda create --name new_environment_name python=3.10'''
It is recommended to name the environment "deorbit" or similar.

Activate the new environment in the prompt by entering:
'''conda activate new_environment_name'''
and replace 'new_environment_name' with your previously chosen environment name, in this example it is "deorbit".
The prompt should now change from:
'''(base) C:\Users\User>'''
to:
'''(deorbit) C:\Users\User>'''
Which shows that the "deorbit" environment is activated.

If you encounter any issues, make sure to check your installation and path configuration of Conda. Alternatively, you can try entering the command into the Anaconda prompt by searching "Anaconda prompt" if you installed Anaconda Distribution.


### Installing Conda (RECOMMENDED)

If you do not have Conda installed it can be downloaded from: 
https://conda.io/projects/conda/en/latest/user-guide/install/windows.html
Choose either Miniconda for a lightweight version or Anaconda Distribution.


### Installing the Deorbit Package

pip installing the Deorbit package allows the package to be imported into your Python files, enter:
'''python -m pip install git+https://github.com/ES98B-Mir-project-23/mir-orbiter.git@main#egg=mir-satellite-deorbiter'''
Into your command prompt or Anaconda prompt.

### Installing the Deorbit Package for Development Purposes (OPTIONAL)

To download the code for development purposes and have access to the code, you can clone the repository from GitHub to a local directory with the command line prompt:
'''git clone https://github.com/ES98B-Mir-project-23/mir-orbiter.git'''

Alternatively, the repository can be accessed on https://github.com/ES98B-Mir-project-23

The prompt should now show something similar to:
'''(deorbit) C:\Users\User\mir-orbiter>'
With the name of your current environment in brackets. See above for activating Conda environments. 
If it displays:
'''(deorbit) C:\Users\User>'''
Type:
'''cd mir-orbiter'''
to navigate to the mir-orbiter path

To install the required package dependencies, enter:
'''python -m pip install -e .[dev]'''
This will install the package in editable mode, allowing the package to be modified and changes to be applied immediately in the local environment. 

After either of the above, the package will be available as `deorbit` in the python environment and is ready to use. 

## Usage Guide

When creating a new Python file, import the package together with Numpy and Matplotlib for complete functionality:

'''
import deorbit
import numpy as np
import matplotlib.pyplot as plt
'''

### Simulator


