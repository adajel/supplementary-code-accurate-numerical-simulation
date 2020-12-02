from __future__ import print_function

from dolfin import *
import ufl

import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt

class Solver():
    """ Class for solving Moris model with membrane mechanisms subject to
        modelling, and with unknowns w = (alpha_r, k_r, phi_r) where:

        alpha_r - volume fraction in compartment r
        k_r     - concentration of ion species k in compartment r
        phi_r   - potential in compartment r

    and gating variables s. """

    def __init__(self, problem, dt_value, Tstop, MMS_test=False):
        """ initialize solver """
        # time variables
        self.dt = Constant(dt_value)         # time step
        self.Tstop = Tstop                   # end time

        # boolean for specifying whether the simulation is a MMS test case
        self.MMS_test = MMS_test

        # get problem
        self.problem = problem               # problem to be solved
        self.mesh = problem.mesh             # mesh
        self.N_ions = problem.N_ions         # number of ions (3-4)
        self.N_states = problem.N_states     # number of ODE states (5)
        self.N_comparts = problem.N_comparts # number of compartments (2-3)

        # create function spaces for PDEs
        self.setup_function_spaces_PDE()

        # create function spaces and solver for ODEs if problem has states
        if self.N_states == 0:
            self.ss = None
        else:
            self.setup_function_spaces_ODE()
            self.ODE_solver()

        # create PDE solver
        self.PDE_solver()

        return

    def setup_function_spaces_ODE(self):
        """ Create function spaces for ODE solver """
        # number of ODE states
        dim = self.N_states
        # define function space
        self.S = VectorFunctionSpace(self.mesh, "CG", 1, dim=dim)

        # unknowns (start with initial conditions)
        inits_ODE = self.problem.inits_ODE
        self.ss = interpolate(inits_ODE, self.S)

        return

    def setup_function_spaces_PDE(self):
        """ Create function spaces for PDE solver """
        N_comparts = self.N_comparts            # number of compartments
        N_ions = self.N_ions                    # number of ions
        N_unknows = N_comparts*(2 + N_ions) - 1 # number of unknowns

        # define function space
        DG0 = FiniteElement('DG', self.mesh.ufl_cell(), 0)    # DG0 element
        CG1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)    # CG1 element
        alpha_elements = [DG0]*(N_comparts - 1)               # DG0 for alpha
        k_phi_elements = [CG1]*(N_unknows - (N_comparts - 1)) # CG1 for k, phi

        ME = MixedElement(alpha_elements + k_phi_elements)    # mixed element
        self.W = FunctionSpace(self.mesh, ME)                 # function space

        # initial conditions
        inits_PDE = self.problem.inits_PDE
        self.w_ = project(inits_PDE, self.W)

        # unknowns (use initial conditions as guess in Newton solver)
        self.w = interpolate(inits_PDE, self.W)
        return

    def ODE_solver(self):
        """ Create PointIntegralSolver for solving membrane ODEs """
        # trial and test functions
        s = split(self.ss)
        q = split(TestFunction(self.S))
        # get rhs of ODE system
        F = self.problem.F

        # If MMS, add source terms to ODEs
        if self.MMS_test:
            source_terms_ODE = self.problem.source_terms_ODE
            F_exprs = F(self.w_, s, source_terms_ODE, self.problem.t_ODE)
        else:
            F_exprs = F(self.w_, s, self.problem.t_ODE)

        F_exprs_q = ufl.zero()

        for i, expr_i in enumerate(F_exprs.ufl_operands):
            F_exprs_q += expr_i*q[i]
        rhs = F_exprs_q*dP()

        # create ODE scheme (choose between ESDIRK4, BackwardEuler, RK4)
        Scheme = eval("ESDIRK4")
        #Scheme = eval("BackwardEuler")
        #Scheme = eval("RK4")

        scheme = Scheme(rhs, self.ss, self.problem.t_ODE)
        # create ODE solver
        self.pi_solver = PointIntegralSolver(scheme)

        return

    def PDE_solver(self):
        """ Create variational formulation for PDEs """
        # get physical parameters
        params = self.problem.params
        # get number of compartments and ions
        N_comparts = self.problem.N_comparts
        N_ions = self.problem.N_ions

        # extract physical parameters
        temperature = params['temperature'] # temperature
        F = params['F']                     # Faraday's constant
        R = params['R']                     # gas constant
        # membrane parameters
        gamma_M = params['gamma_M']         # area of cell membrane per unit volume of membrane
        nw_M = params['nw_M']               # hydraulic permeability
        C_M = params['C_M']                 # capacitance
        # ion specific parameters
        z = params['z']                     # valence of ions
        D = params['D']                     # diffusion coefficient sodium
        # compartmental parameters
        xie = params['xie']                 # scaling factor diffusion neuron
        a = params['a']                     # amount of immobile ions neuron

        # split function for unknown solution in current step n+1
        ww = split(self.w)
        # split function for known solution in previous time step n
        ww_ = split(self.w_)
        # Define test functions
        vv = TestFunctions(self.W)

        # set transmembrane ion fluxes
        self.problem.set_membrane_fluxes(self.w, self.w_, self.ss)

        # get transmembrane ion fluxes - mol/(m^2s)
        J_M = self.problem.membrane_fluxes

        # define extracellular volume fractions (alpha_E = 1.0 - sum_I alpha_I)
        alpha_E = 1.0   # unknown in current time step n+1
        alpha_E_ = 1.0  # solution in previous time step
        # subtract all intracellular volume fractions
        for j in range(N_comparts - 1):
            alpha_E += - ww[j]
            alpha_E_ += - ww_[j]

        # initiate variational formulation
        A_alpha_I = 0 # for intracellular (ICS) volume fractions
        A_k_I = 0     # for extracellular (ICS) conservation of ions
        A_k_E = 0     # for extracellular (ECS) conservation of ions
        A_phi_I = 0   # for intracellular (ICS) potentials
        A_phi_E = 0   # for extracellular (ECS) potentials

        # shorthands
        phi_E = ww[N_comparts*(2 + N_ions) - 2]   # ECS potential unknown
        v_phi_E = vv[N_comparts*(2 + N_ions) - 2] # ECS potential test function
        a_E = a[N_comparts - 1]                   # amount of immobile ions ECS
        z_0 = z[N_ions]                           # valence of immobile ions

        # add contribution from immobile ions to form for ECS potential (C/m^3)
        A_phi_E += - z_0*F*a_E*v_phi_E*dx

        # ICS contribution to variational formulations
        for j in range(N_comparts - 1):
            # shorthands
            phi_I = ww[N_comparts*(1 + N_ions) - 1 + j]   # ICS potential
            v_phi_I = vv[N_comparts*(1 + N_ions) - 1 + j] # test function for ICS potential
            phi_M = phi_I - phi_E                         # membrane potential
            alpha_I = ww[j]                               # ICS volume fractions
            alpha_I_ = ww_[j]                             # ICS volume fractions
            a_I = a[j]                                    # number of immobile ions ICS

            # add contribution from phi_M to form for ICS potentials (C/m^3)
            A_phi_I += gamma_M[j]*C_M[j]*phi_M*v_phi_I*dx
            # add contribution from immobile ions to form for ICS potentials (C/m^3)
            A_phi_I += - z_0*F*a[j]*v_phi_I*dx
            # add contribution from phi_M to form for ECS potential (C/m^3)
            A_phi_E += - gamma_M[j]*C_M[j]*phi_M*v_phi_E*dx

            # initiate transmembrane water flux (m/s)
            w_M = a_E/alpha_E - a_I/alpha_I

            for i in range(N_ions):
                # index for ion i in ICS compartment j
                index_I = N_comparts*(i + 1) - 1 + j
                # shorthands
                k_I = ww[index_I]   # unknown ion concentration ICS
                k_I_ = ww_[index_I] # previous ion concentration ICS
                v_k_I = vv[index_I] # test function ion concentration ICS
                D_ij = D[i]*xie[j]  # effective diffusion coefficients

                # index for ion i in ECS
                index_E = N_comparts*(i + 2) - 2
                # shorthand
                k_E = ww[index_E]   # unknown ion concentration ECS
                v_k_E = vv[index_E] # test function ion concentration ECS

                # compartmental ion flux for ion i in compartment j - (mol/m^2s)
                J_I = - D_ij*(grad(k_I) + z[i]*F*k_I/(R*temperature)*grad(phi_I))

                # form for conservation of ion i in compartment j - (mol/m^3s)
                A_k_I += 1.0/self.dt*inner(alpha_I*k_I - alpha_I_*k_I_, v_k_I)*dx \
                       - inner(J_I, grad(v_k_I))*dx \
                       + gamma_M[j]*inner(J_M[i][j], v_k_I)*dx

                # form for conservation of ion i in ECS - (mol/m^3s)
                A_k_E += - gamma_M[j]*inner(J_M[i][j], v_k_E)*dx
                # add ion specific part to form for ICS potentials (C/m^3)
                A_phi_I += - F*alpha_I*z[i]*k_I*v_phi_I*dx

                # add contribution from ions to water flux
                w_M += k_E - k_I

            # test function for ICS volume fractions
            v_alpha_I = vv[j]
            # add form for ICS volume fraction (1/s)
            A_alpha_I += 1.0/self.dt*inner(alpha_I - alpha_I_, v_alpha_I)*dx \
                       + gamma_M[j]*nw_M[j]*R*temperature*w_M*v_alpha_I*dx

        # add forms for ECS ions and potential
        for i in range(N_ions):
            # index for ion i in ECS
            index_E = N_comparts*(i + 2) - 2
            # shorthand
            k_E = ww[index_E]          # unknown ion concentration ECS
            k_E_ = ww_[index_E]        # unknown ion concentration ECS
            v_k_E = vv[index_E]        # test function ion concentration ECS
            D_iE = D[i]*alpha_E        # effective diffusion coefficients

            # compartmental ion flux for ion i in compartment j - (mol/m^2s)
            J_E = - D_iE*(grad(k_E) + z[i]*F*k_E/(R*temperature)*grad(phi_E))

            # form for conservation of ion i in ECS - (mol/m^3s)
            A_k_E += 1.0/self.dt*inner(alpha_E*k_E - alpha_E_*k_E_, v_k_E)*dx \
                   - inner(J_E, grad(v_k_E))*dx

            # add ion specific part to form for ECS potential (C/m^3)
            A_phi_E += - F*alpha_E*z[i]*k_E*v_phi_E*dx

        ######################################################################
        A_MMS = 0
        # check if problem is a test problem (MMS test)
        if self.MMS_test:
            # get source and boundary terms
            source_terms_PDE = self.problem.source_terms_PDE
            boundary_terms = self.problem.boundary_terms

            # get facet normal
            n = FacetNormal(self.mesh)

            # add source and boundary terms for
            for i in range(N_comparts*(2 + N_ions) - 1):
                A_MMS += - inner(source_terms_PDE[i], vv[i])*dx
                if boundary_terms[i] is not None:
                    if len(n) == 1:
                        # add boundary terms for 1D case
                        A_MMS += inner(boundary_terms[i], vv[i])*ds
                    else:
                        # add boundary terms for ND, N > 1 case
                        A_MMS += inner(dot(boundary_terms[i], n), vv[i])*ds

        # assemble system
        self.A = A_alpha_I + A_k_I + A_k_E + A_phi_I + A_phi_E + A_MMS
        return

    def solve_system_godunov(self, path_results=False):
        """ Solve PDE system with iterative Newton solver, and ODE system
            with PointIntegralSolver

        Assume that w^{n-1} = [[k]_r^{n-1}, phi_r^{n-1} , ...]
        and s^{n-1} = [s1^{n-1} , ... , s5^{n-1}] are known.

            (1) Update w^n by solving PDEs, with s^{n-1} from ODE step
            (2) Update s^n by solving ODEs, with phi_M^{n-1} from PDE step

        repeat (1)-(2) until global end time is reached """

        # save results at every second
        eval_int = float(1.0/self.dt)

        # initialize saving of results
        if path_results:
            filename = path_results
            self.initialize_h5_savefile_PDE(filename + 'PDE/' + 'results.h5')
            self.initialize_xdmf_savefile_PDE(filename + 'PDE/')
            self.initialize_h5_savefile_ODE(filename + 'ODE/' +'results.h5')
            self.initialize_xdmf_savefile_ODE(filename + 'ODE/')

            # save initial PDE state
            self.save_h5_PDE()
            self.save_xdmf_PDE()
            # save initial ODE state
            self.save_h5_ODE()
            self.save_xdmf_ODE()

        # shorthands
        N_comparts = self.problem.N_comparts
        N_ions = self.problem.N_ions

        # index for ECS potential
        index = N_comparts*(2 + N_ions) - 2
        value = Constant(0.0)
        point = self.problem.boundary_point

        # set Dirichlet bcs (phi_E = 0 in point at boundary)
        bc = DirichletBC(self.W.sub(index), value, point, method='pointwise')
        bcs = [bc]

        # initiate solver
        J = derivative(self.A, self.w)                                # calculate Jacobian
        problem = NonlinearVariationalProblem(self.A, self.w, bcs, J) # create problem
        solver  = NonlinearVariationalSolver(problem)                 # create solver
        prm = solver.parameters                                       # get parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-9             # set absolute tolerance
        prm['newton_solver']['relative_tolerance'] = 1E-9             # set relative tolerance
        prm['newton_solver']['maximum_iterations'] = 10               # set max iterations
        prm['newton_solver']['relaxation_parameter'] = 1.0            # set relaxation parameter

        # initiate iteration number
        k = 1

        while (float(self.problem.t_PDE) <= self.Tstop):
            print("----------------------------------------")
            print("Current time:", float(self.problem.t_PDE))
            print("k:", k)
            print("----------------------------------------")

            # solve PDE system
            self.problem.t_PDE.assign(float(self.dt + self.problem.t_PDE))
            solver.solve()         # solve
            self.w_.assign(self.w) # update previous PDE solutions

            if self.N_states > 0:
                # solve ODEs and (NB!) update current time
                self.pi_solver.step(float(self.dt))

            # save results every eval_int'th time step
            if (k % eval_int == 0) and path_results:

                # save PDE solutions
                self.save_h5_PDE()
                self.save_xdmf_PDE()
                # save ODE solutions
                self.save_h5_ODE()
                self.save_xdmf_ODE()

            # update iteration number
            k += 1

        # close results files
        if path_results:
            self.close_h5_PDE()
            self.close_xdmf_PDE()
            self.close_h5_ODE()
            self.close_xdmf_ODE()

        return

    def initialize_h5_savefile_PDE(self, filename):
        """ initialize h5 file """
        self.h5_idx_PDE = 0
        # save PDE solution
        self.h5_file_PDE = HDF5File(self.mesh.mpi_comm(), filename, 'w')
        self.h5_file_PDE.write(self.mesh, '/mesh')
        self.h5_file_PDE.write(self.w, '/solution',  self.h5_idx_PDE)
        return

    def initialize_h5_savefile_ODE(self, filename):
        """ initialize h5 file """
        self.h5_idx_ODE = 0
        # save ODE solution
        self.h5_file_ODE = HDF5File(self.mesh.mpi_comm(), filename, 'w')
        self.h5_file_ODE.write(self.mesh, '/mesh')
        self.h5_file_ODE.write(self.ss, '/solution',  self.h5_idx_ODE)
        return

    def save_h5_PDE(self):
        """ save results to h5 file """
        self.h5_idx_PDE += 1
        self.h5_file_PDE.write(self.w, '/solution',  self.h5_idx_PDE)
        return

    def save_h5_ODE(self):
        """ save results to h5 file """
        self.h5_idx_ODE += 1
        self.h5_file_ODE.write(self.ss, '/solution',  self.h5_idx_ODE)
        return

    def close_h5_PDE(self):
        """ close h5 file """
        self.h5_file_PDE.close()
        return

    def close_h5_ODE(self):
        """ close h5 file """
        self.h5_file_ODE.close()
        return

    def initialize_xdmf_savefile_PDE(self, file_prefix):
        """ initialize xdmf files """
        # save PDE solutions
        self.xdmf_files_PDE = []
        # number of unknowns
        N_unknows_PDE = self.N_comparts*(2 + self.N_ions) - 1
        for idx in range(N_unknows_PDE):
            filename_xdmf = file_prefix + '_PDE_' + str(idx) + '.xdmf'
            xdmf_file = XDMFFile(self.mesh.mpi_comm(), filename_xdmf)
            xdmf_file.parameters['rewrite_function_mesh'] = False
            xdmf_file.parameters['flush_output'] = True
            self.xdmf_files_PDE.append(xdmf_file)
            xdmf_file.write(self.w.split()[idx], self.problem.t_PDE.values()[0])
        return

    def initialize_xdmf_savefile_ODE(self, file_prefix):
        """ initialize xdmf files """
        # save ODE solutions
        self.xdmf_files_ODE = []
        # number of unknowns
        N_unknows_ODE = self.N_states
        for idx in range(N_unknows_ODE):
            filename_xdmf = file_prefix + '_ODE_' + str(idx) + '.xdmf'
            xdmf_file = XDMFFile(self.mesh.mpi_comm(), filename_xdmf)
            xdmf_file.parameters['rewrite_function_mesh'] = False
            xdmf_file.parameters['flush_output'] = True
            self.xdmf_files_ODE.append(xdmf_file)
            xdmf_file.write(self.ss.split()[idx], self.problem.t_ODE.values()[0])
        return

    def save_xdmf_PDE(self):
        """ save results to xdmf files """
        for idx in range(len(self.xdmf_files_PDE)):
            self.xdmf_files_PDE[idx].write(self.w.split()[idx], self.problem.t_PDE.values()[0])
        return

    def save_xdmf_ODE(self):
        """ save results to xdmf files """
        for idx in range(len(self.xdmf_files_ODE)):
            self.xdmf_files_ODE[idx].write(self.ss.split()[idx], self.problem.t_ODE.values()[0])
        return

    def close_xdmf_PDE(self):
        """ close xdmf files """
        for idx in range(len(self.xdmf_files_PDE)):
            self.xdmf_files_PDE[idx].close()
        return

    def close_xdmf_ODE(self):
        """ close xdmf files """
        for idx in range(len(self.xdmf_files_ODE)):
            self.xdmf_files_ODE[idx].close()
        return
