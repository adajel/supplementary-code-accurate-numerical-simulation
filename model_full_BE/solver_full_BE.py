from __future__ import print_function

from dolfin import *
import ufl

import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt

class Solver():
    """ Class for solving Moris model with
        unknowns w = (alpha_r, k_r, phi_r, p_E) where:

        alpha_r - volume fraction in compartment r
        k_r     - concentration of ion species k in compartment r
        phi_r   - potential in compartment r
        p_E     - extracellular hydrostatic pressure

    and gating variables s. """

    def __init__(self, problem, dt_value, Tstop, MMS_test=False):
        """ initialize solver """
        # time variables
        self.dt = Constant(dt_value)         # time step
        self.Tstop = Tstop                   # end time

        # boolean for specifying whether the simulation is a test case
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
        self.ss = project(inits_ODE, self.S)
        return

    def setup_function_spaces_PDE(self):
        """ Create function spaces for PDE solver """
        N_comparts = self.N_comparts             # number of compartments
        N_ions = self.N_ions                     # number of ions
        self.N_unknows = N_comparts*(2 + N_ions) # number of unknowns

        # define function space
        DG0 = FiniteElement('CG', self.mesh.ufl_cell(), 1)         # CG1 element
        CG1 = FiniteElement('CG', self.mesh.ufl_cell(), 1)         # CG1 element
        alpha_elements = [DG0]*(N_comparts - 1)                    # DG0 for alpha
        k_phi_elements = [CG1]*(self.N_unknows - (N_comparts - 1)) # CG1 for k, phi
        elements = alpha_elements + k_phi_elements                 # elements

        ME = MixedElement(elements)                        # mixed element
        self.W = FunctionSpace(self.mesh, ME)              # function space

        ## initial conditions
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
        F_exprs = F(self.w_, s, self.problem.t_ODE)
        F_exprs_q = ufl.zero()

        for i, expr_i in enumerate(F_exprs.ufl_operands):
            F_exprs_q += expr_i*q[i]
        rhs = F_exprs_q*dP()

        # create ODE scheme
        Scheme = eval("ESDIRK4")
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
        S_M = params['S_M']                 # membrane stiffness

        # ion specific parameters
        z = params['z']                     # valence of ions
        D = params['D']                     # diffusion coefficient sodium

        # compartmental parameters
        xie = params['xie']                 # scaling factor diffusion
        a = params['a']                     # amount of immobile ions
        kappa = params['kappa']             # hydraulic permeabilities
        alpha_init = params['alpha_init']   # initial volume fractions

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
        alpha_E = 1.0      # unknown in current time step n+1
        alpha_E_ = 1.0     # solution in previous time step
        alpha_E_init = 1.0 # initial alpha_E
        # subtract all intracellular volume fractions
        for j in range(N_comparts - 1):
            alpha_E += - ww[j]
            alpha_E_ += - ww_[j]
            alpha_E_init += - alpha_init[j]

        # initiate variational formulation
        A_alpha_I = 0 # for intracellular (ICS) volume fractions
        A_k_I = 0     # for extracellular (ICS) conservation of ions
        A_k_E = 0     # for extracellular (ECS) conservation of ions
        A_phi_I = 0   # for intracellular (ICS) potentials
        A_phi_E = 0   # for extracellular (ECS) potentials
        A_p_E = 0     # for extracellular (ECS) hydrostatic pressure

        # shorthands
        phi_E = ww[N_comparts*(2 + N_ions) - 2]   # ECS potential unknown
        v_phi_E = vv[N_comparts*(2 + N_ions) - 2] # ECS potential test function
        p_E = ww[N_comparts*(2 + N_ions) - 1]     # ECS hydrostatic pressure unknown
        v_p_E = vv[N_comparts*(2 + N_ions) - 1]   # ECS hydrostatic pressure test function
        a_E = a[N_comparts - 1]                   # amount of immobile ions ECS
        kappa_E = kappa[N_comparts - 1]           # hydraulic permeability
        z_0 = z[N_ions]                           # valence of immobile ions

        # hydrostatic and oncotic pressure ECS
        p_hat_E = p_E - R*temperature*a_E/alpha_E

        # fluid velocity ECS
        u_E = - kappa_E*grad(p_hat_E)
        for i in range(N_ions):
            # unknown ion concentration ECS
            k_E = ww[N_comparts*(i + 2) - 2]
            # add ion specific contribution to fluid velocity u
            u_E += - kappa_E*F*grad(phi_E)*z[i]*k_E

        # add contribution from immobile ions to form for ECS potential (C/m^3)
        A_phi_E += - z_0*F*a_E*v_phi_E*dx
        # add contribution from ECS fluid velocity to form for ECS pressure
        A_p_E += - inner(alpha_E*u_E, grad(v_p_E))*dx

        # for DEBUG
        self.u_Is = []
        self.u_E = u_E

        # ICS contribution to variational formulations
        for j in range(N_comparts - 1):
            # shorthands
            phi_I = ww[N_comparts*(1 + N_ions) - 1 + j]   # ICS potential
            v_phi_I = vv[N_comparts*(1 + N_ions) - 1 + j] # test function for ICS potential
            phi_M = phi_I - phi_E                         # membrane potential
            alpha_I = ww[j]                               # ICS volume fractions
            alpha_I_ = ww_[j]                             # ICS volume fractions
            v_alpha_I = vv[j]                             # test function for ICS volume fractions
            a_I = a[j]                                    # number of immobile ions ICS

            # calculate intracellular pressure
            tau = S_M[j]*(alpha_I - alpha_init[j])
            p_I = p_E + tau
            p_hat_I = p_I - R*temperature*a[j]/alpha_I

            # fluid velocity ICS
            u_I = - kappa[j]*grad(p_hat_I)
            for i in range(N_ions):
                # unknown ion concentration ICS
                k_I = ww[N_comparts*(i + 1) - 1 + j]
                # add ion specific contribution to fluid velocity u
                u_I += - kappa[j]*F*grad(phi_I)*z[i]*k_I

            # add intracellular fluid velocities to list (for DEBUG)
            self.u_Is.append(u_I)

            # add contribution from phi_M to form for ICS potentials (C/m^3)
            A_phi_I += gamma_M[j]*C_M[j]*phi_M*v_phi_I*dx
            # add contribution from immobile ions to form for ICS potentials (C/m^3)
            A_phi_I += - z_0*F*a[j]*v_phi_I*dx
            # add contribution from phi_M to form for ECS potential (C/m^3)
            A_phi_E += - gamma_M[j]*C_M[j]*phi_M*v_phi_E*dx
            # add contribution from ICS fluid velocity to form for ECS pressure
            A_p_E += - inner(alpha_I*u_I, grad(v_p_E))*dx

            # initiate transmembrane water flux (m/s)
            w_M = p_I - p_E + R*temperature*(a_E/alpha_E - a_I/alpha_I)

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
                J_I = - D_ij*(grad(k_I) + z[i]*F*k_I/(R*temperature)*grad(phi_I)) \
                      + alpha_I*u_I*k_I

                # form for conservation of ion i in compartment j - (mol/m^3s)
                A_k_I += 1.0/self.dt*inner(alpha_I*k_I - alpha_I_*k_I_, v_k_I)*dx \
                      - inner(J_I, grad(v_k_I))*dx \
                      + gamma_M[j]*inner(J_M[i][j], v_k_I)*dx

                # form for conservation of ion i in ECS - (mol/m^3s)
                A_k_E += - gamma_M[j]*inner(J_M[i][j], v_k_E)*dx
                # add ion specific part to form for ICS potentials (C/m^3)
                A_phi_I += - F*alpha_I*z[i]*k_I*v_phi_I*dx

                # add contribution from ions to water flux
                w_M += R*temperature*(k_E - k_I)

            # add form for ICS volume fraction (1/s)
            A_alpha_I += 1.0/self.dt*inner(alpha_I - alpha_I_, v_alpha_I)*dx \
                       - inner(alpha_I*u_I, grad(v_alpha_I))*dx \
                       + gamma_M[j]*nw_M[j]*w_M*v_alpha_I*dx

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
            J_E = - D_iE*(grad(k_E) + z[i]*F*k_E/(R*temperature)*grad(phi_E)) \
                  + alpha_E*u_E*k_E

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
            source_terms = self.problem.source_terms
            boundary_terms = self.problem.boundary_terms

            # get facet normal
            n = FacetNormal(self.mesh)

            # add source and boundary terms
            for i in range(N_comparts*(2 + N_ions)):
                A_MMS += - inner(source_terms[i], vv[i])*dx
                if boundary_terms[i] is not None:
                    if len(n) == 1:
                        # add boundary terms for 1D case
                        A_MMS += inner(boundary_terms[i], vv[i])*ds
                    else:
                        # add boundary terms for ND, N > 1 case
                        A_MMS += inner(dot(boundary_terms[i], n), vv[i])*ds

        # assemble system
        self.A = A_alpha_I + A_k_I + A_k_E + A_phi_I + A_phi_E + A_p_E + A_MMS

        # shorthands
        N_comparts = self.problem.N_comparts
        N_ions = self.problem.N_ions
        value = Constant(0.0)
        point = self.problem.boundary_point

        # index for ..
        index_1 = N_comparts*(2 + N_ions) - 2 # .. ECS potential
        index_2 = N_comparts*(2 + N_ions) - 1 # .. ECS hydrostatic pressure

        # set Dirichlet bcs
        bc_1 = DirichletBC(self.W.sub(index_1), value, point, method='pointwise')
        bc_2 = DirichletBC(self.W.sub(index_2), value, point, method='pointwise')
        bcs = [bc_1, bc_2]

        # initiate solver
        J = derivative(self.A, self.w)                                # calculate Jacobian
        problem = NonlinearVariationalProblem(self.A, self.w, bcs, J) # create problem
        self.PDE_solver  = NonlinearVariationalSolver(problem)        # create solver
        prm = self.PDE_solver.parameters                              # get parameters
        prm['newton_solver']['absolute_tolerance'] = 1E-9             # set absolute tolerance
        prm['newton_solver']['relative_tolerance'] = 1E-9             # set relative tolerance
        prm['newton_solver']['maximum_iterations'] = 10               # set max iterations
        prm['newton_solver']['relaxation_parameter'] = 1.0            # set relaxation parameter

        return

    def solve_system_godenov(self, path_results=False):
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
            self.initialize_h5_savefile(filename + 'results.h5')
            self.initialize_xdmf_savefile(filename)
            # save initial state
            self.save_h5()
            self.save_xdmf()

        # initiate iteration number
        k = 1

        while (float(self.problem.t_PDE) <= self.Tstop):
            print("Current time:", float(self.problem.t_PDE))

            # update time and solve PDE system
            self.problem.t_PDE.assign(float(self.dt + self.problem.t_PDE))
            self.PDE_solver.solve()     # solve
            self.w_.assign(self.w)      # update previous PDE solutions

            if self.N_states > 0:
                # solve ODEs and (NB!) update current time
                self.pi_solver.step(float(self.dt))

            # save results every eval_int'th time step
            if (k % eval_int==0) and path_results:
                # save results
                self.save_h5()
                self.save_xdmf()

            # update iteration number
            k += 1

        # close results files
        if path_results:
            self.close_h5()
            self.close_xdmf()

        return

    def initialize_h5_savefile(self, filename):
        """ initialize h5 file """
        self.h5_idx = 0
        self.h5_file = HDF5File(self.mesh.mpi_comm(), filename, 'w')
        self.h5_file.write(self.mesh, '/mesh')
        self.h5_file.write(self.w, '/solution',  self.h5_idx)
        return

    def save_h5(self):
        """ save results to h5 file """
        self.h5_idx += 1
        self.h5_file.write(self.w, '/solution',  self.h5_idx)
        return

    def close_h5(self):
        """ close h5 file """
        self.h5_file.close()
        return

    def initialize_xdmf_savefile(self, file_prefix):
        """ initialize xdmf files """
        self.xdmf_files = []
        # number of unknowns
        self.N_unknows = self.N_comparts*(2 + self.N_ions)
        for idx in range(self.N_unknows):
            filename_xdmf = file_prefix + '_' + str(idx) + '.xdmf'
            xdmf_file = XDMFFile(self.mesh.mpi_comm(), filename_xdmf)
            xdmf_file.parameters['rewrite_function_mesh'] = False
            xdmf_file.parameters['flush_output'] = True
            self.xdmf_files.append(xdmf_file)
            xdmf_file.write(self.w.split()[idx], self.problem.t_PDE.values()[0])
        return

    def save_xdmf(self):
        """ save results to xdmf files """
        for idx in range(len(self.xdmf_files)):
            self.xdmf_files[idx].write(self.w.split()[idx], self.problem.t_PDE.values()[0])
        return

    def close_xdmf(self):
        """ close xdmf files """
        for idx in range(len(self.xdmf_files)):
            self.xdmf_files[idx].close()
        return
