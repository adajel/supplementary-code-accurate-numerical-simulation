# BE time stepping scheme #

This directory contains an implementation of a numerical scheme for solving the
Mori model in the zero flow limit with two compartments (neurons and ECS) and
Na, K, Cl. CSD is stimulated by excitatory fluxes. Numerical scheme: BE for
time stepping, ESDIRK4 for ODE time stepping (can be altered in solve_BE.py)
and a Strang (solve_system_strange() in solve_BE.py) or Godenov splitting
scheme (solve_system_godenov() in solve_BE.py)

### Dependencies ###

To get the environment needed (all dependencies etc.) to run the code, download
the docker container by running for the MMS test:

    docker run -t -v $(pwd):/home/fenics -i quay.io/fenicsproject/stable:2017.1.0

or for the convergence test and CSD simulations:

    docker run -t -v $(pwd):/home/fenics -i quay.io/fenicsproject/stable

Note that plotting requires LaTex.

### Usage ###

Each numerical experiments can be run by the run_*.py files:

    python run_convergence_test_godunov.py

    python run_convergence_test_strang.py

    python run_CSD_simulation.py

    python run_MMM_test_1D.py

    python run_MMM_test_1D_ODE.py

### Files ###

* *run_convergence_test_godunov.py*  
    Run convergence test with a CSD wave using Godenov splitting

    - Edit file to specify path for results (data) and figures.

    - Output: generates table with values for wave speed (table_wavespeed.txt), width
        of wave (table_wavewidth.txt), duration of wave (table_duration.txt) and
        plots of neuronal potential (*.png).

* *run_convergence_test_strang.py*  
    Run convergence test with a CSD wave using Strang splitting

    - Edit file to specify path for results (data) and figures.

    - Output: generates table with values for wave speed (table_wavespeed.txt), width
        of wave (table_wavewidth.txt), duration of wave (table_duration.txt) and
        plots of neuronal potential (*.png).

* *run_CSD_simulation.py*
    Run simulation of CSD

    - Output: generates time and space plots for the unknowns

* *run_mm_test_1D.py*
    Run method of manufactured solutions (MMS) test with passive membrane
    mechanism on a 1D mesh (i.e. no ODEs)

    - Output: tables with errors (L2 and H1 when applicable) for the unknowns

* *run_mm_test_1D_ODE.py*  
    Run method of manufactured solutions (MMS) test for coupled PDEs and ODEs
    on a 1D mesh.

* *mms_1D.py*
    Contains MMS problem used in run_mm_test_1D.py.


* *mms_1D_ODE.py*  
    Contains MMS problem used in run_mm_test_1D_ODE.py.

* *solver_BE.py*  
    Contains class for a FEM solver for the mori model.  Numerical scheme: BE
    for time stepping, ESDIRK4 for ODE time stepping (can be altered in
    solve_BE.py) and a Strang (solve_system_strange()) or Godenov
    splitting scheme (solve_system_godenov()).

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
