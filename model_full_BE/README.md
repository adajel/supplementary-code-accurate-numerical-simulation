# Full model - BE time stepping scheme #

This directory contains an implementation of a numerical scheme for solving the
full Mori model with three compartments (neurons, glial and ECS) and
Na, K, Cl. CSD is stimulated by excitatory fluxes. Numerical scheme: BE for
time stepping, ESDIRK4 for ODE time stepping and a Godenov splitting scheme.

### Dependencies ###

To get the environment needed (all dependencies etc.) to run the code, download
the docker container by running for the MMS test:

    docker run -t -v $(pwd):/home/fenics -i quay.io/fenicsproject/stable:2017.1.0

or for the convergence test and CSD simulations:

    docker run -t -v $(pwd):/home/fenics -i quay.io/fenicsproject/stable

Note that plotting requires LaTex.

### Usage ###

Each numerical experiment can be run by the run_*.py files, i.e.:

    python run_CSD_simulation.py

    python run_MMS_test_1D.py

### Files ###

* *run_CSD_simulation.py*  
    Run convergence test with a CSD wave using Godenov splitting

    - Output: plots of all state variables in time and space, fluid velocities
        and intracellular pressures.

* *run_mms_test_1D.py*  
    Run method of manufactured solutions (MMS) test with passive membrane
    mechanism on a 1D mesh (i.e. only PDEs, no ODEs)

    - Output: tables with errors (L2 and H1 when applicable) for the state variables

* *solver_full_BE.py*  
    Contains class for a FEM solver for the mori model.  Numerical scheme: BE
    for time stepping, ESDIRK4 for ODE time stepping (can be altered in
    solve_BE.py) and a Godenov
    splitting scheme (solve_system_godenov()).

* *mms_1D.py*  
    Contains MMS problem used in run_mms_test_1D.py.

* *problem_base.py*  
    Contains class for problem base, specifying model parameters, initial
    conditions, membrane model (ODEs and/or algebraic expressions).

* *problems.py*  
    Contains class for problems, specifying triggering mechanism (excitatory
    fluxes).

* *plotter.py*  
    Contains class for plotting.

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
