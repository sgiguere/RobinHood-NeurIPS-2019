# RobinHood

Python code implementing algorithms that provide high-probability safety guarantees for solving offline bandit problems.

# Installation

This code has been tested on Ubuntu.

First, install Python 3.x, Numpy (1.16+), and Cython (0.29+).

The remaining dependencies can be installed by executing the following command from the Python directory of : 

	pip install -r requirements.txt

# Usage

The experiments featured in the paper can be executed by running the provided batch file from the Python directory, as follows:

     ./experiments/scripts/bandit_experiments.bat
     
Once the experiments complete, the figures for the paper can be generated using the command, 

     python -m experiments.scripts.bandit_figures
     
Once completed, the new figures will be saved to `Python/figures/neurips/*` by default.

# License

Code for RobinHood is released under the MIT license, with the exception of the code for POEM (located in `Python/baselines/POEM`) and the code for FairMachineLearning (located in `Python/baselines/fairml.py`), which are released under their licences assigned by their respective authors.
