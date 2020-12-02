from __future__ import print_function

import subprocess
import string
import os
import re

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from dolfin import *
import numpy as np

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

# define colors
red = '#ed3e15'
orange = '#fca106'
green = '#97d51a'
yellow = '#fdbf08'

# set colors
colormap = cm.viridis
mus = [1,2,3,4,5,6]
colorparams = mus
colormap = cm.viridis
normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

c2 = colormap(normalize(mus[0]))
c1 = colormap(normalize(mus[1]))
c0 = colormap(normalize(mus[2]))
c3 = colormap(normalize(mus[3]))
c4 = colormap(normalize(mus[4]))
c5 = colormap(normalize(mus[5]))

fs = 0.9
lw = 2.5

class Plotter():

    def __init__(self, problem, path_data=None):
        self.problem = problem
        N_ions = self.problem.N_ions
        N_comparts = self.problem.N_comparts
        self.N_unknows = N_comparts*(2 + N_ions) - 1

        # initialize mesh and data file
        if path_data is not None:
            self.set_mesh_and_datafile(path_data)

        return

    def set_mesh_and_datafile(self, path_data):
        # file containing data
        h5_fname = path_data + 'PDE/results.h5'

        # create mesh and read data file
        self.mesh = Mesh()
        self.hdf5 = HDF5File(MPI.comm_world, h5_fname, 'r')
        self.hdf5.read(self.mesh, '/mesh', False)
        # convert coordinates from m to mm
        self.mesh.coordinates()[:] *= 1e3

        return

    def read_from_file(self, n, i, scale=1.):
        """ get snapshot of solution w[i] at time = n seconds """
        N_comparts = self.problem.N_comparts
        N_unknows = self.N_unknows

        DG0 = FiniteElement('DG', self.mesh.ufl_cell(), 1)
        CG1 = FiniteElement('CG', self.mesh.ufl_cell(), 2)
        e = [DG0]*(N_comparts - 1) + [CG1]*(N_unknows - (N_comparts - 1))
        W = FunctionSpace(self.mesh, MixedElement(e))
        u = Function(W)

        if i < (N_comparts - 1):
            V_DG0 = FunctionSpace(self.mesh, DG0)
            f = Function(V_DG0)
        else:
            V_CG1 = FunctionSpace(self.mesh, CG1)
            f = Function(V_CG1)

        self.hdf5.read(u, "/solution/vector_" + str(n))
        assign(f, u.split()[i])

        f.vector()[:] = scale*f.vector().get_local()

        return f

    def project_to_function_space(self, u):
        """ project u onto function space """

        CG1 = FiniteElement('CG', self.mesh.ufl_cell(), 2)
        V = FunctionSpace(self.mesh, CG1)
        f = project(u, V)

        return f

    def init_duration(self):
        # duration (s)
        self.duration = 0
        return

    def get_duration(self, n):
        """ save wave speed at given time n """
        #g = self.read_from_file(n, 7, scale=1.0e3)
        g = self.read_from_file(n, 4)
        # evaluate g at x=1.0 mm
        g_ = g(1.0)

        # add one second to duration if wave is present (i.e K_E > 10 mM)
        if g_ > 10:
            self.duration += 1

        return

    def init_wavespeed(self):
        """ initiate calculation of wave speed """
        # for saving pairs of space and time coordinates when wave passes
        self.wavefront_space_time = []
        # get coordinates (mm)
        self.coordinates = self.mesh.coordinates()
        return

    def get_wavespeed(self, n):
        """ save wave speed at given time n """
        # get neuron potential
        phi_N = self.read_from_file(n, 7, scale=1.0e3)

        # get values of phi N
        phi_N_values = phi_N.compute_vertex_values()

        # get max value of potential (phi) at time n
        index_max = max(range(len(phi_N_values)), key=phi_N_values.__getitem__)

        v_max = phi_N_values[index_max]          # max value of phi N (mV)
        p_max = self.coordinates[index_max][0]   # point of max value (mm)
        t_max = n                                # time for max value (s)

        # check that wave has passed
        # (assumption: wave has passed if neuron potential phi  > -20 mV)
        if v_max > -20:
            # save point in space and time of wave front
            # (assumption: wave front is where phi_N has greatest value)
            self.wavefront_space_time.append([p_max, t_max])
        return

    def save_wavespeed(self, res_path):
        p_max_prev = 0  # previous value of p_max
        t_max_prev = 0  # previous value of t_max
        speeds = []     # for saving speeds

        for i, pair in enumerate(self.wavefront_space_time):
            # get values
            (p_max, t_max) = pair

            if i > 0:
                if p_max_prev == p_max:
                    # wave front has not moved to next point -> continue
                    continue
                else:
                    # wave front has moved to next point -> calculate speed
                    # distance between wave front (mm)
                    dx = p_max - p_max_prev
                    # time between wave front (convert from second to min)
                    dt = (t_max - t_max_prev)/60.0

                    # calculate and append speed
                    speeds.append(dx/dt)
                    # update p_max_prev and t_max_prev
                    p_max_prev = p_max
                    t_max_prev = t_max

        # plot speeds
        if (len(speeds) > 0):
            # calculate average speed
            avg = sum(speeds)/len(speeds)
            # plot speeds
            plt.figure()
            plt.plot(speeds, 'ro')
            plt.ylabel(r'mm/min')
            plt.xlabel('intervals')
            plt.title('Wave speed, avg = %.2f mm/min' % avg)
            # save figure
            fname_res = res_path + 'wavespeed.png'
            plt.savefig(fname_res)
            plt.close()

        return speeds

    def make_tmp_frames(self, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # get data
        alpha_N = self.read_from_file(n, 0)
        Na_N = self.read_from_file(n, 1)
        Na_E = self.read_from_file(n, 2)
        K_N = self.read_from_file(n, 3)
        K_E = self.read_from_file(n, 4)
        Cl_N = self.read_from_file(n, 5)
        Cl_E = self.read_from_file(n, 6)
        phi_N = self.read_from_file(n, 7, scale=1.0e3)
        phi_E = self.read_from_file(n, 8, scale=1.0e3)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N
        alpha_E = self.project_to_function_space(u_alpha_E)

        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_E_init = 1.0 - alpha_N_init
        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        # create plot
        fig = plt.figure(figsize=(17, 10))
        ax = plt.gca()

        # subplot number 1 - extracellular potential KNP-EMI
        ax1 = fig.add_subplot(2,3,1, xlim=[0,10], ylim=[0, 150])
        plt.title(r'Neuron ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_N, '--', linewidth=lw)
        plot(K_N, '--', linewidth=lw)
        plot(Cl_N, '--', linewidth=lw)

        ax3 = fig.add_subplot(2,3,2, xlim=[0,10], ylim=[0, 150])
        plt.title(r'ECS ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_E, '--', linewidth=lw)
        plot(K_E, '--', linewidth=lw)
        plot(Cl_E, '--', linewidth=lw)
        plt.legend([r'Na$^+$', r'K$^+$', r'Cl$^-$'], loc='center right')

        # subplot number 2 - extracellular potentail EMI
        ax2 = fig.add_subplot(2,3,4, xlim=[0,10], ylim=[-100, 20])
        plt.title(r'potentials')
        plt.ylabel(r'mV')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plot(phi_E, linewidth=lw)
        plot(phi_N, linewidth=lw)

        ax3 = fig.add_subplot(2,3,5, xlim=[0,10], ylim=[-50, 20])
        plt.title(r'change volume fractions (\%)')
        plt.ylabel(r'\%')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plot(alpha_E_diff, linewidth=lw)
        plot(alpha_N_diff, linewidth=lw)
        plt.legend([r'ECS', r'neuron'], loc='center right')

        # make pretty
        ax.axis('off')
        plt.tight_layout()

        # save figure to file
        fname_res = path_figs + '_tmp_%d' % n
        plt.savefig(fname_res + '.png', format='png')
        plt.close()

        return

    def make_convergence_tables(self, params):
        """ generate tables with wave characteristics during refinement in
            space and time """

        # get parameters
        directory = params['directory']
        Tstop = params['Tstop']
        L = params['L']
        N_values = params['N_values']
        dt_values = params['dt_values']

        # path for tables and plots
        dir_res = directory + 'tables_and_figures/'
        # check if directory exists, if not create
        if not os.path.exists(dir_res):
            os.makedirs(dir_res)

        # create files for results
        title_f4 = dir_res + "table_wavespeed.txt"
        title_f5 = dir_res + "table_duration.txt"
        title_f6 = dir_res + "table_wavewidth.txt"

        # open files
        f4 = open(title_f4, 'w+')
        f5 = open(title_f5, 'w+')
        f6 = open(title_f6, 'w+')

        # write header to file
        f4.write('$N \Delta t$')
        f5.write('$N \Delta t$')
        f6.write('$N \Delta t$')

        for dt in dt_values:
            # write header to file (dt values)
            f4.write('& %g ' % round((dt*1000),3))
            f5.write('& %g ' % round((dt*1000),3))
            f6.write('& %g ' % round((dt*1000),3))
 
        # write header to file (diff)
        f4.write('& $\Delta$')
        f5.write('& $\Delta$')
        f6.write('& $\Delta$')

        # write newline to file
        f4.write('\\\\')
        f5.write('\\\\')
        f6.write('\\\\')

        # write line to file
        f4.write('\midrule ')
        f5.write('\midrule ')
        f6.write('\midrule ')

        # lists for saving characteristics
        L2_norm_Nlist =  []
        dtdx_Nlist =  []
        inf_norm_Nlist =  []
        wave_speed_Nlist =  []
        duration_Nlist =  []
        wavewidth_Nlist =  []

        for i in range(len(N_values)):
            # number of cells
            N = N_values[i]

            # write header to file (N, mesh resolution)
            f4.write('%g &' % (N))
            f5.write('%g &' % (N))
            f6.write('%g &' % (N))

            for j in range(len(dt_values)):

                # set path to (data) and create mesh for current spatial resolution
                self.set_mesh_and_datafile(directory + 'data_%d%d/' % (i,j))

                # spatial and temporal resolution
                dt = dt_values[j]
                dx = self.mesh.hmin()

                # frame number (at end time)
                n = int(Tstop)

                # get neuron potential
                g = self.read_from_file(n, 7, scale=1.0e3)

                # current time and space resolution (for naming results file)
                dt_value = r'%.6f' % dt
                dt_str = re.sub(r'[/.!$%^&*()]', '',  dt_value)

                # get values of g
                g_vec = g.compute_vertex_values()
                # get max value (i.e. wave front)
                index_max = max(range(len(g_vec)), key=g_vec.__getitem__)

                # calculate width of wave (from ECS potassium concentration)
                # read file
                g_EK = self.read_from_file(n, 4)
                # get values of g
                g_vec_EK = g_EK.compute_vertex_values()
                # get min and max x s.t. [k]_e > 10
                fil_lst = [x*dx for x,y in enumerate(g_vec_EK) if y > 10]

                wavewidth = max(fil_lst) - min(fil_lst)

                # get wave speed
                self.init_wavespeed()
                self.init_duration()

                # calculate wave speed
                for k in range(n):
                    self.get_wavespeed(k)
                    self.get_duration(k)
                speeds = self.save_wavespeed(dir_res)
                # get mean wave speed and round to 3 decimals
                wave_speed = round(np.mean(speeds[3:]), 3)

                # estimated wave speed
                est = 5.1

                # green - wave speed = +/- 5 % of estimated wave speed
                g_low = est - est/100*5.0
                g_upp = est + est/100*5.0

                # orange - wave speed = +/- 10 % of estimated wave speed
                o_low = est - est/100*15.0
                o_upp = est + est/100*15.0

                # generate plot
                plt.figure()
                # colour code plots
                if (g_low <= wave_speed <= g_upp):
                    plot(g, linewidth=10.0, color=green)
                elif (o_low <= wave_speed <= o_upp):
                    #plot(g, linewidth=10.0, color=orange)
                    plot(g, linewidth=10.0, color=yellow)
                else:
                    plot(g, linewidth=10.0, color=red)
                # make plot pretty
                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                plt.tick_params(axis='y', which='both', left=False, top=False, labelleft=False)
                # save plot
                plt.savefig(dir_res + '_N%d_' % N + 'dt' + dt_str + '.png')
                plt.close()

                # write results to file
                f4.write('%.3f &' % wave_speed)
                f5.write('%.3f &' % self.duration)
                f6.write('%.3f &' % wavewidth)

                if i == (len(N_values) - 1):
                    wave_speed_Nlist.append(wave_speed)
                    duration_Nlist.append(self.duration)
                    wavewidth_Nlist.append(wavewidth)

            if i > 0:
                # write newline to file
                f4.write('%.3f' % (wave_speed_prev - wave_speed))
                f5.write('%.3f' % (duration_prev - self.duration))
                f6.write('%.3f' % (wavewidth_prev - wavewidth))

            # update previous values
            wave_speed_prev =  wave_speed
            duration_prev =  self.duration
            wavewidth_prev =  wavewidth

            # write newline to file
            f4.write('\\\\')
            f5.write('\\\\')
            f6.write('\\\\')

        # write line to file
        f4.write('\midrule ')
        f5.write('\midrule ')
        f6.write('\midrule ')

        # write delta to file
        f4.write('$\Delta$ & &')
        f5.write('$\Delta$ & &')
        f6.write('$\Delta$ & &')

        for i in range(len(dt_values[:-1])):
            # write delta to file
            f4.write('%.3f &' % abs(wave_speed_Nlist[i] - wave_speed_Nlist[i+1]))
            f5.write('%.3f &' % abs(duration_Nlist[i] - duration_Nlist[i+1]))
            f6.write('%.3f &' % abs(wavewidth_Nlist[i] - wavewidth_Nlist[i+1]))

        # write newline to file
        f4.write('\\\\')
        f5.write('\\\\')
        f6.write('\\\\')

        # close files
        f4.close()
        f5.close()
        f6.close()

        return

    def make_timeplot(self, path_figs, Tstop):

        # point at which to calculate duration
        point = 2.0

        # list of function values at point
        alpha_Ns = []; alpha_Es = []
        Na_Es = []; K_Es = []; Cl_Es = []
        Na_Ns = []; K_Ns = []; Cl_Ns = []
        phi_Ns = []; phi_Es = []

        for n in range(Tstop):
            # get functions
            alpha_N = self.read_from_file(n, 0)
            Na_N = self.read_from_file(n, 1)
            Na_E = self.read_from_file(n, 2)
            K_N = self.read_from_file(n, 3)
            K_E = self.read_from_file(n, 4)
            Cl_N = self.read_from_file(n, 5)
            Cl_E = self.read_from_file(n, 6)
            phi_N = self.read_from_file(n, 7, scale=1.0e3)
            phi_E = self.read_from_file(n, 8, scale=1.0e3)

            # calculate extracellular and initial volume fractions
            u_alpha_E = 1.0 - alpha_N
            alpha_E = self.project_to_function_space(u_alpha_E)
            alpha_N_init = float(self.problem.alpha_N_init)
            alpha_E_init = 1.0 - alpha_N_init
            # calculate charge in volume fractions (alpha) in %
            u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
            u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
            alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
            alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

            # evaluate functions at point and append value to list
            alpha_Ns.append(alpha_N_diff(point))
            alpha_Es.append(alpha_E_diff(point))
            Na_Es.append(Na_E(point))
            K_Es.append(K_E(point))
            Cl_Es.append(Cl_E(point))
            Na_Ns.append(Na_N(point))
            K_Ns.append(K_N(point))
            Cl_Ns.append(Cl_N(point))
            phi_Ns.append(phi_N(point))
            phi_Es.append(phi_E(point))

        # range of x values
        xlim = [0.0, Tstop]

        # create plot
        fig = plt.figure(figsize=(18.0*fs, 10*fs))
        ax = plt.gca()

        # extracellular concentrations
        ax1 = fig.add_subplot(3,1,1, xlim=xlim, ylim=[0.0, 150])
        plt.title(r'ECS ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r's')
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        #plt.xticks([0, 10, 20, 30, 40, 50])
        plt.plot(Na_Es, color=c0, label=r'Na$^+$', linewidth=lw)
        plt.plot(K_Es, color=c1,  label=r'K$^+$',linewidth=lw)
        plt.plot(Cl_Es, color=c2, label=r'Cl$^-$',linewidth=lw)

        # potentials
        ax2 = fig.add_subplot(3,1,2, xlim=xlim, ylim=[-100, 20])
        plt.title(r'potentials')
        plt.ylabel(r'mV')
        plt.xlabel(r's')
        plt.yticks([-90, -70, -50, -30, -10, 10])
        #plt.xticks([0, 10, 20, 30, 40, 50])
        plt.plot(phi_Es, color=c3, label=r'ECS', linewidth=lw)
        plt.plot(phi_Ns, color=c4, label=r'neuron',linewidth=lw)

        # volume fractions
        ax3 = fig.add_subplot(3,1,3, xlim=xlim, ylim=[-50, 20])
        plt.title(r'\% change volume fractions')
        plt.ylabel(r'\%')
        plt.xlabel(r's')
        plt.yticks([-40, -30, -20, -10, 0, 10])
        #plt.xticks([0, 10, 20, 30, 40, 50])
        plt.plot(alpha_Es, color=c3, linewidth=lw)
        plt.plot(alpha_Ns, color=c4, linewidth=lw)

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.25)

        # add legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.9))

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']
        for num, ax in enumerate([ax1, ax2, ax3]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'timeplot'
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()

        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        return

    def make_spaceplot(self, path_data, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """
        # path to file
        fname = path_data + 'PDE/' + 'results.h5'

        alpha_N = self.read_from_file(n, 0)
        Na_N = self.read_from_file(n, 1)
        Na_E = self.read_from_file(n, 2)
        K_N = self.read_from_file(n, 3)
        K_E = self.read_from_file(n, 4)
        Cl_N = self.read_from_file(n, 5)
        Cl_E = self.read_from_file(n, 6)
        phi_N = self.read_from_file(n, 7, scale=1.0e3)
        phi_E = self.read_from_file(n, 8, scale=1.0e3)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N
        alpha_E = self.project_to_function_space(u_alpha_E)

        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_E_init = 1.0 - alpha_N_init
        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        # plotting parameters
        xlim = [0.0, 10.0] # range of x values
        lw_ = 3.0

        # create plot
        fig = plt.figure(figsize=(3.5*fs, 14*fs))
        ax = plt.gca()

        # subplot number 2 - extracellular concentrations
        ax1 = fig.add_subplot(3,1,1, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'$[k]_R$ (mM)')
        plt.xticks([0, 2.5, 5, 7.5 , 10])
        plt.yticks([20, 60, 100, 140])
        plot(Na_E, color=c0, label=r'Na$^+$', linewidth=lw_)
        plot(K_E, color=c1, label=r'K$^+$',linewidth=lw_)
        plot(Cl_E, color=c2, label=r'Cl$^-$',linewidth=lw_)

        # subplot number 3 - potentials
        ax2 = fig.add_subplot(3,1,2, xlim=xlim, ylim=[-80, 20])
        plt.ylabel(r'$\phi_r$ (mV)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-70, -50, -30, -10, 10])
        plot(phi_E, color=c3, label=r'ECS', linewidth=lw_)
        plot(phi_N, color=c4, label=r'neuron', linewidth=lw_)

        # subplot number 4 - volume fractions
        ax3 = fig.add_subplot(3,1,3, xlim=xlim, ylim=[-50, 20])
        plt.ylabel(r'\% change $\alpha_r$')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-45, -30,  -15, 0, 15])
        plot(alpha_E_diff, color=c3, linewidth=lw_)
        plot(alpha_N_diff, color=c4, linewidth=lw_)

        # make pretty
        ax.axis('off')

        # add numbering for the subplots (A, B, C etc)
        fig.suptitle('t=10', fontsize=19)
        letters = [r'\textbf{A}', r'\textbf{D}', r'\textbf{G}']

        plt.subplots_adjust(hspace=0.25)

        for num, ax in enumerate([ax1, ax2, ax3]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'spaceplot_n%d' % n
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()
        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        return
