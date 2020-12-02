import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import numpy as np
from dolfin import *

import string
import os

# set font & text parameters
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'

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

class Plotter():

    def __init__(self, problem, path_data):
        self.problem = problem
        N_ions = self.problem.N_ions
        N_comparts = self.problem.N_comparts
        self.N_unknows = N_comparts*(2 + N_ions)

        # file containing data
        self.h5_fname = path_data + 'results.h5'

        # create mesh
        self.mesh = Mesh()
        hdf5 = HDF5File(MPI.comm_world, self.h5_fname, 'r')
        hdf5.read(self.mesh, '/mesh', False)
        self.mesh.coordinates()[:] *= 1e3

        return

    def read_from_file(self, n, i, scale=1.):
        """ get snapshot of solution w[i] at time = n seconds """
        hdf5 = HDF5File(MPI.comm_world, self.h5_fname, 'r')

        N_comparts = self.problem.N_comparts
        N_unknows = self.N_unknows

        CG1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        e = [CG1]*(N_comparts - 1) + [CG1]*(N_unknows - (N_comparts - 1))
        W = FunctionSpace(self.mesh, MixedElement(e))
        u = Function(W)

        V_CG1 = FunctionSpace(self.mesh, CG1)
        f = Function(V_CG1)

        hdf5.read(u, "/solution/vector_" + str(n))
        assign(f, u.split()[i])

        f.vector()[:] = scale*f.vector().get_local()
        return f

    def project_to_function_space(self, u):
        """ project u onto function space """

        P1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)
        V = FunctionSpace(self.mesh, P1)
        f = project(u, V)

        return f

    def make_spaceplot(self, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # calculate velocities
        kappa = self.problem.params['kappa']
        a = self.problem.params['a']
        z = self.problem.params['z']
        temperature = self.problem.params['temperature']
        R = self.problem.params['R']
        F = self.problem.params['F']

        # get data
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        Na_N = self.read_from_file(n, 2)
        Na_G = self.read_from_file(n, 3)
        Na_E = self.read_from_file(n, 4)
        K_N = self.read_from_file(n, 5)
        K_G = self.read_from_file(n, 6)
        K_E = self.read_from_file(n, 7)
        Cl_N = self.read_from_file(n, 8)
        Cl_G = self.read_from_file(n, 9)
        Cl_E = self.read_from_file(n, 10)
        phi_N = self.read_from_file(n, 11, scale=1.0e3)
        phi_G = self.read_from_file(n, 12, scale=1.0e3)
        phi_E = self.read_from_file(n, 13, scale=1.0e3)
        p_E = self.read_from_file(n, 14, scale=1.0e-3)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)

        # get initial volume fractions
        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_G_init = float(self.problem.alpha_G_init)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        # oncotic pressure neuron (convert to kPa)
        u_onc_N = R*temperature*(a[0]/alpha_N)
        onc_N = self.project_to_function_space(u_onc_N*1.0e-3)
        # oncotic pressure glial (convert to kPa)
        u_onc_G = R*temperature*(a[1]/alpha_G)
        onc_G = self.project_to_function_space(u_onc_G*1.0e-3)
        # oncotic pressure ECS (convert to kPa)
        u_onc_E = R*temperature*(a[2]/alpha_E)
        onc_E = self.project_to_function_space(u_onc_E*1.0e-3)

        # calculate neuron and glial pressures
        S_M = self.problem.params['S_M']
        alpha_init = self.problem.params['alpha_init']

        tau_N = S_M[0]*(alpha_N - alpha_init[0])
        tau_G = S_M[1]*(alpha_G - alpha_init[1])
        p_N = self.project_to_function_space(p_E + tau_N)
        p_G = self.project_to_function_space(p_E + tau_G)


        # plotting parameters
        xlim = [0.0, 10.0] # range of x values
        lw = 3.0           # line width
        fs = 0.9
        fosi = 17

        # create plot
        fig = plt.figure(figsize=(3.5*4*fs, 10*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(2,3,1, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'$[k]_n$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_N, color=c0, linewidth=lw)
        plot(K_N, color=c1, linewidth=lw)
        plot(Cl_N, color=c2, linewidth=lw)

        ax2 = fig.add_subplot(2,3,2, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'$[k]_g$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_G, color=c0, linewidth=lw)
        plot(K_G, color=c1, linewidth=lw)
        plot(Cl_G, color=c2, linewidth=lw)

        ax3 = fig.add_subplot(2,3,3, xlim=xlim, ylim=[0.0, 150])
        plt.ylabel(r'$[k]_R$ (mM)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_E, color=c0, label=r'Na$^+$', linewidth=lw)
        plot(K_E, color=c1, label=r'K$^+$', linewidth=lw)
        plot(Cl_E, color=c2, label=r'Cl$^-$', linewidth=lw)

        ax4 = fig.add_subplot(2,3,4, xlim=xlim, ylim=[-100, 20])
        plt.ylabel(r'$\phi$ (mV)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plot(phi_N, color=c3, linewidth=lw)
        plot(phi_G, color=c4, linewidth=lw)
        plot(phi_E, color=c5, linewidth=lw)

        ax5 = fig.add_subplot(2,3,5, xlim=xlim, ylim=[-50, 20])
        plt.ylabel(r'$\Delta \alpha (\%)$', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plot(alpha_N_diff, color=c3, linewidth=lw)
        plot(alpha_G_diff, color=c4, linewidth=lw)
        plot(alpha_E_diff, color=c5, linewidth=lw)

        # TODO
        ax6 = fig.add_subplot(2,3,6, xlim=xlim, ylim=[-700, 700])
        #plt.yticks([-600, -500, -400, -300, -200, -100, 0])
        #ax6 = fig.add_subplot(2,3,6, xlim=xlim, ylim=[-25, 5])
        #plt.yticks([-20, -15, -10, -5, 0])
        plt.ylabel(r'p (kPa)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(p_N, color=c3, label=r'neuron', linewidth=lw)
        plot(p_G, color=c4, label=r'glial', linewidth=lw)
        plot(p_E, color=c5, label=r'ECS', linewidth=lw)

        #ax7 = fig.add_subplot(3,3,7, xlim=xlim, ylim=[0, 400])
        #plt.title(r'oncotic pressure')
        #plt.ylabel(r'kPa')
        #plt.xlabel(r'mm')
        #plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([50, 100, 150, 200, 250, 300, 350])
        #plot(onc_N, color=c3, linewidth=lw)
        #plot(onc_G, color=c4, linewidth=lw)
        #plot(onc_E, color=c5, linewidth=lw)

        #ax8 = fig.add_subplot(3,3,8, xlim=xlim, ylim=[-1000, 0])
        #plt.title(r'modified pressure')
        #plt.ylabel(r'kPa')
        #plt.xlabel(r'mm')
        #plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([-800, -600, -400, -200])
        #plot(p_N - onc_N, color=c3, label=r'neuron', linewidth=lw)
        #plot(p_G - onc_G, color=c4, label=r'glial', linewidth=lw)
        #plot(p_E - onc_E, color=c5, label=r'ECS', linewidth=lw)

        # make legend
        plt.figlegend(bbox_to_anchor=(1.0, 0.89))

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.38, hspace=0.3, left=0.2)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', \
                   r'\textbf{D}', r'\textbf{E}', r'\textbf{F}']
        for num, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'spaceplot_n%d' % n
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()

        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        # intracellular fluid velocity
        p_hat_N = p_N - R*temperature*a[0]/alpha_N
        u_N_ = - kappa[0]*grad(p_hat_N) \
               - kappa[0]*F*grad(phi_N)*(z[0]*Na_N + z[1]*K_N + z[2]*Cl_N)

        # intracellular fluid velocity
        p_hat_G = p_G - R*temperature*a[1]/alpha_G
        u_G_ = - kappa[1]*grad(p_hat_G) \
               - kappa[1]*F*grad(phi_G)*(z[0]*Na_G + z[1]*K_G + z[2]*Cl_G)

        # extracellular fluid velocity
        p_hat_E = p_E - R*temperature*a[2]/alpha_E
        u_E_ = - kappa[2]*grad(p_hat_E) \
               - kappa[2]*F*grad(phi_E)*(z[0]*Na_E + z[1]*K_E + z[2]*Cl_E)

        u_N = project(u_N_[0]*1.0e6)
        u_G = project(u_G_[0]*1.0e6)
        u_E = project(u_E_[0]*1.0e6)

        # create plot
        fig = plt.figure(figsize=(12*fs, 4.4*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(1,3,1, xlim=xlim, ylim=[-0.045, 0.045])
        plt.ylabel(r'$u_n (\mu$m/s)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
        plot(u_N, color=c3, label=r'neuron', linewidth=lw)

        ax2 = fig.add_subplot(1,3,2, xlim=xlim, ylim=[-0.5, 1.5])
        plt.ylabel(r'$u_g (\mu$m/s)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25])
        plot(u_G, label=r'glial', color=c4, linewidth=lw)

        ax3 = fig.add_subplot(1,3,3, xlim=xlim, ylim=[-0.05, 0.02])
        plt.ylabel(r'$u_R (\mu$m/s)', fontsize=fosi)
        plt.xlabel(r'x (mm)', fontsize=fosi)
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-0.04, -0.03, -0.02, -0.01, 0, 0.01])
        plot(u_E, label=r'ECS', color=c5,linewidth=lw)

        plt.figlegend(bbox_to_anchor=(1.0, 0.89))

        # make pretty
        ax.axis('off')
        plt.tight_layout()

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}']
        for num, ax in enumerate([ax1, ax2, ax3]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'spaceplot_u_n%d' % n
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()
        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        return

    def make_tmp_frames(self, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # get data
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        Na_N = self.read_from_file(n, 2)
        Na_G = self.read_from_file(n, 3)
        Na_E = self.read_from_file(n, 4)
        K_N = self.read_from_file(n, 5)
        K_G = self.read_from_file(n, 6)
        K_E = self.read_from_file(n, 7)
        Cl_N = self.read_from_file(n, 8)
        Cl_G = self.read_from_file(n, 9)
        Cl_E = self.read_from_file(n, 10)
        phi_N = self.read_from_file(n, 11, scale=1.0e3)
        phi_G = self.read_from_file(n, 12, scale=1.0e3)
        phi_E = self.read_from_file(n, 13, scale=1.0e3)
        p_E = self.read_from_file(n, 14)


        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)

        # get initial volume fractions
        alpha_N_init = float(self.problem.alpha_N_init)
        alpha_G_init = float(self.problem.alpha_G_init)
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

        # calculate charge in volume fractions (alpha) in %
        u_alpha_N_diff = (alpha_N - alpha_N_init)/alpha_N_init*100
        u_alpha_G_diff = (alpha_G - alpha_G_init)/alpha_G_init*100
        u_alpha_E_diff = (alpha_E - alpha_E_init)/alpha_E_init*100
        alpha_N_diff = self.project_to_function_space(u_alpha_N_diff)
        alpha_G_diff = self.project_to_function_space(u_alpha_G_diff)
        alpha_E_diff = self.project_to_function_space(u_alpha_E_diff)

        # plotting parameters
        xlim = [0.0, 10.0]      # range of x values
        lw = 2.5                # line width

        # create plot
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()

        # subplot number 1 - extracellular potential KNP-EMI
        ax1 = fig.add_subplot(3,3,1, xlim=xlim, ylim=[0.0, 150])
        plt.title(r'neuron ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_N, linewidth=lw)
        plot(K_N, linewidth=lw)
        plot(Cl_N, linewidth=lw)

        ax2 = fig.add_subplot(3,3,2, xlim=xlim, ylim=[0.0, 150])
        plt.title(r'glial ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_G, '--', linewidth=lw)
        plot(K_G, '--', linewidth=lw)
        plot(Cl_G, '--', linewidth=lw)

        ax3 = fig.add_subplot(3,3,3, xlim=xlim, ylim=[0.0, 150])
        plt.title(r'ECS ion concentrations')
        plt.ylabel(r'mM')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([0, 20, 40, 60, 80, 100, 120, 140])
        plot(Na_E, '--', linewidth=lw)
        plot(K_E, '--', linewidth=lw)
        plot(Cl_E, '--', linewidth=lw)
        plt.legend([r'Na$^+$', r'K$^+$', r'Cl$^-$'], loc='center right')

        # subplot number 2 - extracellular potential EMI
        ax4 = fig.add_subplot(3,3,4, xlim=xlim, ylim=[-100, 20])
        plt.title(r'potentials')
        plt.ylabel(r'mV')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-90, -70, -50, -30, -10, 10])
        plot(phi_N, linewidth=lw)
        plot(phi_G, linewidth=lw)
        plot(phi_E, linewidth=lw)

        ax5 = fig.add_subplot(3,3,5, xlim=xlim, ylim=[-50, 20])
        plt.title(r'\% change volume fractions')
        plt.ylabel(r'\%')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-40, -30, -20, -10, 0, 10])
        plot(alpha_N_diff, linewidth=lw)
        plot(alpha_G_diff, linewidth=lw)
        plot(alpha_E_diff, linewidth=lw)

        # calculate neuron and glial pressures
        S_M = self.problem.params['S_M']
        alpha_init = self.problem.params['alpha_init']

        tau_N = S_M[0]*(alpha_N - alpha_init[0])
        tau_G = S_M[1]*(alpha_G - alpha_init[1])
        p_N = self.project_to_function_space(p_E + tau_N)
        p_G = self.project_to_function_space(p_E + tau_G)

        ax6 = fig.add_subplot(3,3,6, xlim=xlim, ylim=[-8.0e2, 2.0e2])
        plt.title(r'ECS mechanical pressure')
        plt.ylabel(r'kPa')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(p_N*1.0e-3, linewidth=lw)
        plot(p_G*1.0e-3, linewidth=lw)
        plot(p_E*1.0e-3, linewidth=lw)
        plt.legend([r'neuron', r'glial', r'ECS'], loc='center right')

        # calculate velocities
        kappa = self.problem.params['kappa']
        a = self.problem.params['a']
        z = self.problem.params['z']
        temperature = self.problem.params['temperature']
        R = self.problem.params['R']
        F = self.problem.params['F']

        # intracellular fluid velocity
        p_hat_N = p_N - R*temperature*a[0]/alpha_N
        u_N_ = - kappa[0]*grad(p_hat_N) \
               - kappa[0]*F*grad(phi_N)*(z[0]*Na_N + z[1]*K_N + z[2]*Cl_N)

        # intracellular fluid velocity
        p_hat_G = p_G - R*temperature*a[1]/alpha_G
        u_G_ = - kappa[1]*grad(p_hat_G) \
               - kappa[1]*F*grad(phi_G)*(z[0]*Na_G + z[1]*K_G + z[2]*Cl_G)

        # extracellular fluid velocity
        p_hat_E = p_E - R*temperature*a[2]/alpha_E
        u_E_ = - kappa[2]*grad(p_hat_E) \
               - kappa[2]*F*grad(phi_E)*(z[0]*Na_E + z[1]*K_E + z[2]*Cl_E)

        u_N = project(u_N_[0])
        u_G = project(u_G_[0])
        u_E = project(u_E_[0])

        ax7 = fig.add_subplot(3,3,7, xlim=xlim, ylim=[-1.0e-1, 1.0e-1])
        plt.title(r'fluid velocity neuron')
        plt.ylabel(r'$\mu$m/s')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(u_N, linewidth=lw)

        ax8 = fig.add_subplot(3,3,8, xlim=xlim, ylim=[-1.0e-1, 1.0e-1])
        plt.title(r'fluid velocity glial')
        plt.ylabel(r'$\mu$m/s')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(u_G, linewidth=lw)

        ax9 = fig.add_subplot(3,3,9, xlim=xlim, ylim=[-1.0e-1, 1.0e-1])
        plt.title(r'fluid velocity ECS')
        plt.ylabel(r'$\mu$m/s')
        plt.xlabel(r'mm')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plot(u_E, linewidth=lw)

        # make pretty
        ax.axis('off')
        plt.tight_layout()

        # save figure to file
        fname_res = path_figs + '_tmp_%d' % n
        plt.savefig(fname_res + '.png', format='png')
        plt.close()

        return

    def plot_pressure(self, path_figs, n):
        """ plot ECS concentrations, phi and % change volume fraction at t=n """

        # calculate velocities
        R = self.problem.params['R']
        F = self.problem.params['F']
        temperature = self.problem.params['temperature']
        a = self.problem.params['a']             # immobile ions
        z = self.problem.params['z']             # valence
        kappa = self.problem.params['kappa']     # hydraulic permeability (compartmental)
        gamma_M = self.problem.params['gamma_M'] # membrane area
        nw_M = self.problem.params['nw_M']       # hydraulic permeability (membrane)

        # get data
        alpha_N = self.read_from_file(n, 0)
        alpha_G = self.read_from_file(n, 1)
        Na_N = self.read_from_file(n, 2)
        Na_G = self.read_from_file(n, 3)
        Na_E = self.read_from_file(n, 4)
        K_N = self.read_from_file(n, 5)
        K_G = self.read_from_file(n, 6)
        K_E = self.read_from_file(n, 7)
        Cl_N = self.read_from_file(n, 8)
        Cl_G = self.read_from_file(n, 9)
        Cl_E = self.read_from_file(n, 10)
        phi_N = self.read_from_file(n, 11, scale=1.0e3)
        phi_G = self.read_from_file(n, 12, scale=1.0e3)
        phi_E = self.read_from_file(n, 13, scale=1.0e3)
        p_E = self.read_from_file(n, 14, scale=1.0e-3)

        # calculate neuron and glial pressures
        S_M = self.problem.params['S_M']
        alpha_init = self.problem.params['alpha_init']

        tau_N = S_M[0]*(alpha_N - alpha_init[0])
        tau_G = S_M[1]*(alpha_G - alpha_init[1])
        p_N = self.project_to_function_space(p_E + tau_N)
        p_G = self.project_to_function_space(p_E + tau_G)

        # calculate extracellular volume fraction
        u_alpha_E = 1.0 - alpha_N - alpha_G
        alpha_E = self.project_to_function_space(u_alpha_E)

        # osmotic pressure neuron membrane
        u_osm_N = Na_E - Na_N + K_E - K_N + Cl_E - Cl_N
        # oncotic pressure neuron membrane
        u_onc_N = a[2]/alpha_E - a[0]/alpha_N

        # osmotic pressure glial membrane
        u_osm_G = Na_E - Na_G + K_E - K_G + Cl_E - Cl_G
        # oncotic pressure membrane
        u_onc_G = a[2]/alpha_E - a[1]/alpha_G

        onc_N = self.project_to_function_space(u_onc_N)
        osm_N = self.project_to_function_space(u_osm_N)
        osm_onc_N = self.project_to_function_space(u_osm_N + u_onc_N)

        onc_G = self.project_to_function_space(u_onc_G)
        osm_G = self.project_to_function_space(u_osm_G)
        osm_onc_G = self.project_to_function_space(u_osm_G + u_onc_G)

        # compartmental oncotic pressure
        u_onc_NN = R*temperature*a[0]/alpha_N*1.0e-3
        u_onc_GG = R*temperature*a[1]/alpha_G*1.0e-3
        u_onc_EE = R*temperature*a[2]/alpha_E*1.0e-3
        onc_NN = self.project_to_function_space(u_onc_NN)
        onc_GG = self.project_to_function_space(u_onc_GG)
        onc_EE = self.project_to_function_space(u_onc_EE)

        # plotting parameters
        xlim = [0.0, 10.0]   # range of x values
        lw = 3.0             # line width

        # create plot
        fig = plt.figure(figsize=(18*0.95, 18*0.95))
        ax = plt.gca()
        plt.suptitle("Compartmental pressure",)

        ax1 = fig.add_subplot(3,3,1, xlim=xlim, ylim=[250, 350])
        plt.title(r'oncotic pressure neuron (kPa)')
        plt.ylabel(r'$\Pi$ (kPa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([-300, -200, -100, 0, 100, 200, 300])
        plot(onc_NN, color=c5, linestyle='dashed', linewidth=lw)

        ax2 = fig.add_subplot(3,3,2, xlim=xlim)#, ylim=[-18.0, 12.0])
        plt.title(r'mechanical pressure neuron (kPa)')
        plt.ylabel(r'$p_n$ (Pa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([-15, -10, -5, 0, 5, 10])
        plot(p_N, color=c3, linestyle='dashed', linewidth=lw)

        ax3 = fig.add_subplot(3,3,3, xlim=xlim, ylim=[-1000, 0])
        plt.title(r'modified pressure neuron (kPa)')
        plot(p_N - onc_NN, color=c4, linestyle='dashed', linewidth=lw)
        plt.ylabel(r'$\tilde{p}_n$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-900, -700, -500, -300, -100])

        ax4 = fig.add_subplot(3,3,4, xlim=xlim, ylim=[250, 350])
        plt.title(r'oncotic pressure glial (kPa)')
        plt.ylabel(r'$\Pi$ (kPa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([-300, -200, -100, 0, 100, 200, 300])
        plot(onc_GG, color=c5, linestyle='dashed', linewidth=lw)

        ax5 = fig.add_subplot(3,3,5, xlim=xlim)#, ylim=[-18.0, 12.0])
        plt.title(r'mechanical pressure glial (kPa)')
        plt.ylabel(r'$p_n$ (Pa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([-15, -10, -5, 0, 5, 10])
        plot(p_G, color=c3, linestyle='dashed', linewidth=lw)

        ax6 = fig.add_subplot(3,3,6, xlim=xlim, ylim=[-1000, 0])
        plt.title(r'modified pressure glial (kPa)')
        plot(p_G - onc_GG, color=c4, linestyle='dashed', linewidth=lw)
        plt.ylabel(r'$\tilde{p}_n$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-900, -700, -500, -300, -100])

        ax7 = fig.add_subplot(3,3,7, xlim=xlim)#, ylim=[250, 350])
        plt.title(r'oncotic pressure ECS (kPa)')
        plt.ylabel(r'$\Pi$ (kPa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([-300, -200, -100, 0, 100, 200, 300])
        plot(onc_EE, color=c5, linestyle='dashed', linewidth=lw)

        ax8 = fig.add_subplot(3,3,8, xlim=xlim)#, ylim=[-18.0, 12.0])
        plt.title(r'mechanical pressure ECS (kPa)')
        plt.ylabel(r'$p_n$ (Pa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        #plt.yticks([-15, -10, -5, 0, 5, 10])
        plot(p_E, color=c3, linestyle='dashed', linewidth=lw)

        ax9 = fig.add_subplot(3,3,9, xlim=xlim, ylim=[-1000, 0])
        plt.title(r'modified pressure ECS (kPa)')
        plot(p_E - onc_EE, color=c4, linestyle='dashed', linewidth=lw)
        plt.ylabel(r'$\tilde{p}_n$ (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-900, -700, -500, -300, -100])

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.38, hspace=0.3)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', \
                   r'\textbf{D}', r'\textbf{E}', r'\textbf{F}', \
                   r'\textbf{G}', r'\textbf{H}', r'\textbf{I}']

        for num, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'pressure_n%d' % n
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()

        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        # create plot
        fig = plt.figure(figsize=(18*0.95, 10*0.95))
        ax = plt.gca()
        plt.suptitle("Membrane pressure and osmolarity",)

        ax1 = fig.add_subplot(2,4,1, xlim=xlim, ylim=[-350, 350])
        plt.title(r'pressure neuron (kPa)')
        plt.ylabel(r'$\Pi$ (kPa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-300, -200, -100, 0, 100, 200, 300])
        plot(osm_N*1.0e-3*R*temperature, color=c3, linestyle='dashed', linewidth=lw)
        plot(onc_N*1.0e-3*R*temperature, color=c5, linestyle='dashed', linewidth=lw)
        plt.legend([r'osmotic', r'oncotic'], loc='center right')

        ax2 = fig.add_subplot(2,4,2, xlim=xlim, ylim=[-18.0, 12.0])
        plt.title(r'sum pressure neuron (Pa)')
        plt.ylabel(r'$\Pi$ (Pa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-15, -10, -5, 0, 5, 10])
        plot((onc_N + osm_N)*R*temperature, color=c4, linestyle='dashed', linewidth=lw)

        ax3 = fig.add_subplot(2,4,3, xlim=xlim, ylim=[-175, 175])
        plt.title(r'osmolarity neuron (mM)')
        plt.ylabel(r'oms (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-150, -100, -50, 0, 50, 100, 150])
        plot(osm_N, color=c3, linewidth=lw)
        plot(onc_N, color=c5, linewidth=lw)
        plt.legend([r'osmotic', r'oncotic'], loc='center right')

        ax4 = fig.add_subplot(2,4,4, xlim=xlim, ylim=[-0.007, 0.003])
        plt.title(r'sum osmolarity neuron (mM)')
        plot(onc_N + osm_N, color=c4, linewidth=lw)
        plt.ylabel(r'oms (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-0.006, -0.004, -0.002, 0, 0.002])

        ax5 = fig.add_subplot(2,4,5, xlim=xlim, ylim=[-350, 350])
        plt.title(r'pressure glial (kPa)')
        plt.ylabel(r'$\Pi$ (kPa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-300, -200, -100, 0, 100, 200, 300])
        plot(osm_G*1.0e-3*R*temperature, color=c2, linestyle='dashed', linewidth=lw)
        plot(onc_G*1.0e-3*R*temperature, color=c3, linestyle='dashed', linewidth=lw)
        plt.legend([r'osmotic', r'oncotic'], loc='center right')

        ax6 = fig.add_subplot(2,4,6, xlim=xlim, ylim=[-18.0, 12.0])
        plt.title(r'sum pressure glial (Pa)')
        plt.ylabel(r'$\Pi$ (Pa)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-15, -10, -5, 0, 5, 10])
        plot((onc_G + osm_G)*R*temperature, color=c0, linestyle='dashed', linewidth=lw)

        ax7 = fig.add_subplot(2,4,7, xlim=xlim, ylim=[-175, 175])
        plt.title(r'osmolarity glial (mM)')
        plt.ylabel(r'oms (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-150, -100, -50, 0, 50, 100, 150])
        plot(osm_G, color=c2, linewidth=lw)
        plot(onc_G, color=c3, linewidth=lw)
        plt.legend([r'osmotic', r'oncotic'], loc='center right')

        ax8 = fig.add_subplot(2,4,8, xlim=xlim, ylim=[-0.007, 0.003])
        plt.title(r'sum osmolarity glial (mM)')
        plot(onc_G + osm_G, color=c0, linewidth=lw)
        plt.ylabel(r'oms (mM)')
        plt.xlabel(r'x (mm)')
        plt.xticks([0, 2.5, 5, 7.5, 10])
        plt.yticks([-0.006, -0.004, -0.002, 0, 0.002])

        # make pretty
        ax.axis('off')
        plt.subplots_adjust(wspace=0.38, hspace=0.3)

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}', r'\textbf{C}', \
                   r'\textbf{D}', r'\textbf{E}', r'\textbf{F}', \
                   r'\textbf{G}', r'\textbf{H}']

        for num, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]):
            ax.text(-0.12, 1.06, letters[num], transform=ax.transAxes)

        # save figure to file
        fname_res = path_figs + 'pressure_membrane_n%d' % n
        plt.savefig(fname_res + '.svg', format='svg')
        plt.close()

        # convert from svg to pdf
        os.system('inkscape -D -z --file=' + fname_res + '.svg --export-pdf=' \
                  + fname_res + '.pdf --export-latex')

        return

    def init_wavespeed(self, num_points=11):
        """ initiate calculation of wave speed """
        # define points at which to evaluate speed in between
        self.space_points = np.linspace(0, 10.0, num_points)
        # for saving pairs of space and time coordinates when wave passes
        self.wavefront_space_time = []
        return

    def get_wavespeed(self, n):
        """ save wave speed at given time n """
        # split solution PDEs and get neuron potential
        phi_N = self.read_from_file(n, 11, scale=1.0e3)

        # for saving neuron potential in each given point
        phi_N_values = []
        for i in range(len(self.space_points)):
            # get point
            point = self.space_points[i]
            # save phi N in point
            phi_N_values.append(phi_N(point))

        # get max value of potential (phi) at time n
        index_max = max(range(len(phi_N_values)), key=phi_N_values.__getitem__)
        v_max = phi_N_values[index_max]      # max value of phi N (mV)
        p_max = self.space_points[index_max] # get point of max value (mm)
        t_max = n                            # get time for max value (s)

        # check that wave has passed
        # (assumption: wave has passed if neuron potential phi  > -20 mV)
        if v_max > -20:
            # save point in space and time of wave front
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
                    # wave front has not moved to next point, continue
                    continue
                else:
                    #wave front has not moved to next point, calculate speed
                    # distance between wave front (mm)
                    dx = p_max - p_max_prev 
                    # time between wave front# (convert from second to min)
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
            fname = res_path + 'wavespeed.png'
            plt.savefig(fname)
            plt.close()

        return
