import os
import sys
import re

import numpy as np
from dolfin import *
import matplotlib.pyplot as plt

# import problem, solver, plotter
from problems import Problem
from solver_CG2 import Solver
from plotter import Plotter

def run_convergence_test(params):
    # get parameter values
    directory = params["directory"]
    dt_values = params["dt_values"]
    N_values = params["N_values"]
    Tstop = params["Tstop"]
    L = params["L"]

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(len(N_values)):

        # create mesh
        N = N_values[i]              # number of cells
        mesh = IntervalMesh(N, 0, L) # mesh
        boundary_point = "near(x[0], %g)" % L

        for j in range(len(dt_values)):

            # temporal and spatial resolution
            dt = dt_values[j] # time step
            dx = mesh.hmin()  # mesh size

            t_PDE = Constant(0.0) # time constant
            t_ODE = Constant(0.0) # time constant
            problem = Problem(mesh, boundary_point, t_PDE, t_ODE)

            # solve system
            path_data = directory + 'data_%d%d/' % (i,j)

            S = Solver(problem, dt, Tstop)
            S.solve_system_strang(path_results=path_data)

    return

def plot(params):
    # get values
    N_values = params["N_values"]
    N = N_values[0] # number of cells
    L = params["L"] # length of domain (m)

    # create dummy problem
    mesh = IntervalMesh(N, 0, L) # mesh
    boundary_point = "near(x[0], %g)" % L
    t_PDE = Constant(0.0) # time constant
    t_ODE = Constant(0.0) # time constant
    problem = Problem(mesh, boundary_point, t_PDE, t_ODE)

    P = Plotter(problem)
    P.make_convergence_tables(params)

    return

if __name__ == '__main__':

    directory = "results/convergence_test/ESDIRK4_STRANG/"

    dt_values = [1.25e-2, 6.25e-3, 3.125e-3, 1.5625e-3, 7.8125e-4, 3.90625e-4, 1.953125e-4]
    N_values = [1000, 2000, 4000, 8000, 16000, 32000]

    Tstop = 50  # time stop (s)
    L = 0.01    # length of domain (m)

    # save results
    params = {'directory':directory,
              'Tstop':Tstop,
              'L':L,
              'N_values':N_values,
              'dt_values':dt_values}

    run_convergence_test(params)
    plot(params)
