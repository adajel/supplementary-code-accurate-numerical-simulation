from dolfin import *

import os
import glob
import sys
import numpy as np

# set path to solver
from solver_full_BE import Solver
from problems import Problem
from plotter import Plotter

if __name__ == '__main__':
    # mesh
    N = 8000                            # mesh size
    L = 0.01                            # m
    mesh = IntervalMesh(N, 0, L)        # create mesh
    boundary_point = "near(x[0], 0.01)" # point on boundary

    # time variables
    dt_value = 3.125e-3                 # time step (s)
    Tstop = 50                          # end time (s)

    t_ODE = Constant(0.0)               # time constant
    t_PDE = Constant(0.0)               # time constant
    problem = Problem(mesh, boundary_point, t_PDE, t_ODE)          # problem

    # check that directory for results (data) exists, if not create
    path_data = 'results/data/CSD/'
    """
    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    # solve system
    S = Solver(problem, dt_value, Tstop)
    S.solve_system_godenov(path_results=path_data)
    """

    # if directory for figures exists remove old, if not create
    path_figs = 'results/figures/CSD/'
    if not os.path.isdir(path_figs):
        os.makedirs(path_figs)

    # create plotter object for visualizing results
    P = Plotter(problem, path_data)

    """
    # initiate calculation of wave speed
    P.init_wavespeed()

    for n in range(Tstop):
        P.make_tmp_frames(path_figs, int(n))  # save plots for debugging/testing
        P.get_wavespeed(int(n))               # calculate wave speed

    # save wave speed
    P.save_wavespeed(path_figs)
    # plot pressure and space plot
    P.plot_pressure(path_figs, Tstop)
    """
    #P.make_spaceplot(path_figs, Tstop)
    P.print_max_min(path_figs, Tstop)
