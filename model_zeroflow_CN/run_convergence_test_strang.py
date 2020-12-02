from dolfin import *

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import re

# import problem, solver and plotter
from problems import Problem
from solver_CN import Solver
from plotter import Plotter

if __name__ == '__main__':

    # path to store results
    directory = "results/convergence_test/RK4_STRANG/"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # time stop
    Tstop = 50
    # length of domain (m)
    L = 0.01

    # space resolutions (m)
    N_values = [1000, 2000, 4000, 8000, 16000, 32000]
    dt_values = [7.8125e-4, 3.90625e-4, 1.953125e-4]

    for i in range(len(N_values)):
        # create mesh
        N = N_values[i]              # number of cells
        mesh = IntervalMesh(N, 0, L) # mesh
        boundary_point = "near(x[0], %g)" % L

        for j in range(len(dt_values)):

            # time variables
            t_PDE = Constant(0.0)           # time constant PDE solver
            t_ODE = Constant(0.0)           # time constant ODE solver
            problem = Problem(mesh, boundary_point, t_PDE, t_ODE)

            dt = dt_values[j]
            dx = mesh.hmin()

            # check that directory for results (data) exists, if not create
            path_data = directory + 'data_%d%d/' % (i,j)

            # solve system
            S = Solver(problem, dt, Tstop)
            S.solve_system_strang(path_results=path_data)

    # save results
    params = {'directory':directory,
              'Tstop':Tstop,
              'L':L,
              'N_values':N_values,
              'dt_values':dt_values,
              'var':'phi_N'}

    P = Plotter(problem)
    P.make_convergence_results(params)
