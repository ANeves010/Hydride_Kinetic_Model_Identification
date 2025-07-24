# Hydride_Kinetic_Model_Identification
This repository contains the data related to a publication named "Kinetic-Model Identification in Metal-Hydride Reactions Using Neural Network Autoencoder Surrogate Models" authored by Neves et al. submitted to Energy and AI (DOI to be added after publication). 
In this work, the kinetic properties of metal hydride AB2 alloy are investigated in different temperature and pressure conditions to determine the kinetic models which best describe their behaviour. The benchmark method (reduced-time method) is compared with a novel approach that applies neural networks that were trained without supervision to make the classification of each kinetic curve (time series). 
Twelve neural networks are trained with simulated data, coming from twelve different kinetic models (equations), using a range of kinetic constant (k) values. These networks are tasked with the reconstruction of the experimental data and the coefficient of determination (R^2) of all networks are compared to determine the best-ranked model for each combination of temperature and pressure. The input for the reconstruction is the time series (reacted fraction vs time) and the output contains the reconstructed curve and the coefficient of determination calculated between the reconstruction and the experimental data. The reacted fraction is calculated by the amount of absorbed (or desorbed) hydrogen divided by the maximum reached capacity during absorption (or desorption). 

This repository contains the raw data, a summary of results and the programming code.

Credits - Maintainer and Data Analysis: Dr.-Ing. André Neves; Data Collection: Dr. Jan Warfsmann; Coding: Dr. Willi Großmann

First version updated on: 24.07.2025 // Version v1.0
Latest Changes: 24.07.2025 // Current Version v1.0

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
