# Supplementary code #

Repository containing code for generating results presented in "Accurate
numerical simulation of electrodiffusion and osmotic water movement in brain
tissue". The sub-directories contain implementations of the different numerical schemes:

* model_full_BE  
    Contains implementation of a DG0/CG1 scheme with Backward Euler (BE) PDE time stepping
    for the full model

* model_zeroflow_BDF2  
    Contains implementation of a DG0/CG1 scheme with BDF2 PDE time stepping
    for the model in the zero flow limit

* model_zeroflow_BE  
    Contains implementation of a DG0/CG1 scheme with Backward Euler (BE) PDE time stepping
    for the model in the zero flow limit

* model_zeroflow_CG2  
    Contains implementation of a DG1/CG2 scheme with BDF2 PDE time stepping
    for the model in the zero flow limit

* model_zeroflow_CN
    Contains implementation of a DG0/CG1 scheme with Cranck Nicholsen (CN) PDE time stepping
    for the model in the zero flow limit

### Dependencies and usage ###

See individual README.md files in each sub-directory.

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
