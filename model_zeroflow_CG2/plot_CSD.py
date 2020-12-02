import os
import sys
import glob

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# import problem and plotter
from plotter import Plotter
from problems import Problem

if __name__ == '__main__':

    # setup directory for saving data
    path_data = "results/convergence_test/ESDIRK4_STRANG/data_32/"

    # setup directory for saving figures
    path_figs = 'results/figures/CSD/'

    # check that directory for results (figures) exists, if not create
    if not os.path.isdir(path_figs):
        os.makedirs(path_figs)

    # create mesh
    N = 8000                              # mesh size
    L = 0.01                              # m
    mesh = IntervalMesh(N, 0, L)          # create mesh
    boundary_point = "near(x[0], %g)" % L # point on boundary
    # time variables
    dt_value = 3.125e-3
    Tstop = 50                             # end time (s)

    # create problem
    t_PDE = Constant(0.0) # time constant
    t_ODE = Constant(0.0) # time constant
    problem = Problem(mesh, boundary_point, t_PDE, t_ODE) # problem

    # create plotter object for visualizing results
    P = Plotter(problem, path_data)

    for n in np.arange(Tstop):
        # make snapshot plot at each second
        P.make_tmp_frames(path_figs, int(n))

    # save time and space plots
    P.make_timeplot(path_figs, Tstop)
    P.make_spaceplot(path_data, path_figs, Tstop)
