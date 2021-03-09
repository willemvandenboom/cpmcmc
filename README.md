# cpmcmc

Repository with the code used for the paper "Unbiased approximation of posteriors via coupled particle Markov chain Monte Carlo" by Willem van den Boom, Ajay Jasra, Maria De Iorio, Alexandros Beskos and Johan G. Eriksson (to be posted on arXiv soon).


## Description of files

* [`cpmcmc.py`](cpmcmc.py) is a Python module that provides an implementation of coupled particle Markov chain Monte Carlo. The other Python scripts import it.

* [`mix_of_Gaussians.py`](mix_of_Gaussians.py) produces the figure for the simulation study with the mixture of Gaussians.

* [`horseshoe.py`](horseshoe.py) produces the figure for the simulation study on horseshoe regression. It imports the C++ code in [`horseshoe.h`](horseshoe.h).

* [`GGM.py`](GGM.py) provides the unbiased inference for the Gaussian graphical model as described in the paper. It considers a different dataset than the paper as the data from the paper are not public. It uses the C++ code in [`GGM.cpp`](GGM.cpp) via the header file [`GGM.h`](GGM.h).

* [`environment.yml`](environment.yml) details the conda environment used for the paper. It can be used to [recreate the environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). Dependencies of the C++ scripts are detailed preceding the respective include directives.