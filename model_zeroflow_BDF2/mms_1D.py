from dolfin import *
import ufl

from scipy.integrate import odeint
import sympy as sm
import numpy as np

class MMS:
    """ Class for calculating source terms and boundary terms of the system for
    given exact solutions """

    def __init__(self, time):
        # set time constant
        self.time = time

        # define symbolic variables
        x, t = sm.symbols('x[0] t')
        self.x = x
        self.t = t

        # define div symbolic constants
        F, R, temperature = sm.symbols('F R temperature')
        gamma_NE, C_NE = sm.symbols('gamma_NE C_NE')
        self.F = F
        self.R = R
        self.temperature = temperature
        self.gamma_NE = gamma_NE
        self.C_NE = C_NE

    def get_src_term_alpha(self, alpha_r, memflux):
        # Function for calculating source terms in equations for conservation
        # of ion conservations
        x = self.x; t = self.t
        # calculate source f term: fI = d alpha_I/dt + gamma_M*w_M
        f = sm.diff(alpha_r, t) + memflux
        # return source term and adjusted compartmental ion flux
        return f

    def get_src_term_k(self, alpha_r, k_r, phi_r, D_r, z_r, memflux):
        # Function for calculating source terms in equations for conservation 
        # of ion conservations
        x = self.x; t = self.t
        F = self.F; R = self.R; temperature = self.temperature
        # calculate gradients
        grad_k_r = sm.diff(k_r, x)
        grad_phi_r = sm.diff(phi_r, x)
        # calculate compartmental flux
        J_k_r = - D_r*grad_k_r - D_r*z_r*F/(R*temperature)*k_r*grad_phi_r

        # calculate source f term: f = dk_r/dt + div (J_kr) + memflux
        f = sm.diff(alpha_r*k_r, t) + sm.diff(J_k_r, x) + memflux
        # return source term and adjusted compartmental ion flux
        return f, J_k_r

    def get_src_term_phi(self, Na_r, K_r, Cl_r, z_Na, z_K, z_Cl, z_0, a_r, alpha_r, memflux):
        # Function for calculating source terms for equations for potentials
        F = self.F
        # calculate source term by: fI = rho_0 + F*sum([k]_r z alpha_r) + memflux
        f = - z_0*F*a_r - F*alpha_r*(z_Na*Na_r + z_K*K_r + z_Cl*Cl_r) + memflux
        # return source term
        return f

    def get_MMS_terms(self, params, degree):
        # return source terms and boundary terms for all equations
        x = self.x; t = self.t
        F = self.F; R = self.R; temperature = self.temperature;
        gamma_NE = self.gamma_NE; C_NE = self.C_NE
        # define constants
        D_Na, D_K, D_Cl = sm.symbols('D_Na D_K D_Cl')          # diffusion coefficients
        z_Na, z_K, z_Cl, z_0 = sm.symbols('z_Na z_K z_Cl z_0') # valence
        nw_NE = sm.symbols('nw_NE')
        xie_N = sm.symbols('xie_N')

        g_Na_leak_N = sm.symbols('g_Na_leak_N') # conductance
        g_K_leak_N = sm.symbols('g_K_leak_N')   # conductance
        g_Cl_leak_N = sm.symbols('g_Cl_leak_N') # conductance

        # set manufactured solution and initial conditions
        exact_solutions = self.get_exact_solution()
        initial_conditions = self.get_initial_conditions()

        # unwrap exact solutions
        alpha_N_e = exact_solutions['alpha_N_e']
        Na_N_e = exact_solutions['Na_N_e']
        K_N_e = exact_solutions['K_N_e']
        Cl_N_e = exact_solutions['Cl_N_e']
        Na_E_e = exact_solutions['Na_E_e']
        K_E_e = exact_solutions['K_E_e']
        Cl_E_e = exact_solutions['Cl_E_e']
        phi_N_e = exact_solutions['phi_N_e']
        phi_E_e = exact_solutions['phi_E_e']

        # unwrap initial conditions
        alpha_N_init = initial_conditions['alpha_N_init']
        Na_N_init = initial_conditions['Na_N_init']
        Na_E_init = initial_conditions['Na_E_init']
        K_N_init = initial_conditions['K_N_init']
        K_E_init = initial_conditions['K_E_init']
        Cl_N_init = initial_conditions['Cl_N_init']
        Cl_E_init = initial_conditions['Cl_E_init']
        phi_N_init = initial_conditions['phi_N_init']
        phi_E_init = initial_conditions['phi_E_init']

        # membrane potential
        phi_NE_e = phi_N_e - phi_E_e
        # define ECS volume fraction

        # membrane potential
        phi_NE_e = phi_N_e - phi_E_e
        # define ECS volume fraction
        alpha_E_e = 1.0 - alpha_N_e

        # initial membrane potential
        phi_NE_init = phi_N_init - phi_E_init
        # define initial ECS volume fraction
        alpha_E_init = 1.0 - alpha_N_init

        ################################################################
        # Nernst potential - neuron
        E_Na_N = R*temperature/(F*z_Na)*sm.log(Na_E_e/Na_N_e) # sodium    - (V)
        E_K_N = R*temperature/(F*z_K)*sm.log(K_E_e/K_N_e)     # potassium - (V)
        E_Cl_N = R*temperature/(F*z_Cl)*sm.log(Cl_E_e/Cl_N_e) # chloride  - (V)

        # Leak currents - neuron
        I_Na_NE = g_Na_leak_N*(phi_NE_e - E_Na_N)  # leak sodium    - (A/m^2)
        I_K_NE = g_K_leak_N*(phi_NE_e - E_K_N)     # leak potassium - (A/m^2)
        I_Cl_NE = g_Cl_leak_N*(phi_NE_e - E_Cl_N)  # leak chloride  - (A/m^2)

        ################################################################
        # convert currents currents to flux - neuron
        J_Na_NE = I_Na_NE/(F*z_Na) # sodium    - (mol/(m^2s))
        J_K_NE = I_K_NE/(F*z_K)    # potassium - (mol/(m^2s))
        J_Cl_NE = I_Cl_NE/(F*z_Cl) # chloride  - (mol/(m^2s))

        # amount of immobile ions neuron (mol/m^3)
        a_N = 1.0/(z_0*F)*gamma_NE*C_NE*phi_NE_init \
            - 1.0/z_0*alpha_N_init*(z_Na*Na_N_init \
                                  + z_K*K_N_init \
                                  + z_Cl*Cl_N_init)

        a_E = - 1.0/(z_0*F)*gamma_NE*C_NE*phi_NE_init \
              - 1.0/z_0*alpha_E_init*(z_Na*Na_E_init \
                                    + z_K*K_E_init \
                                    + z_Cl*Cl_E_init)

        # transmembrane water flux neuron
        w_NE = nw_NE*R*temperature*(a_E/alpha_E_e + Na_E_e + K_E_e + Cl_E_e \
                                  - a_N/alpha_N_e - Na_N_e - K_N_e - Cl_N_e)

        #calculate gradients
        grad_phi_N = sm.diff(phi_N_e, x)
        grad_phi_E = sm.diff(phi_E_e, x)

        # calculate source terms and boundary terms - alphas
        f_alpha_N = self.get_src_term_alpha(alpha_N_e, gamma_NE*w_NE)

        # define effective diffusion coefficients
        D_Na_N = D_Na*xie_N
        D_K_N = D_K*xie_N
        D_Cl_N = D_Cl*xie_N

        D_Na_E = D_Na*alpha_E_e
        D_K_E = D_K*alpha_E_e
        D_Cl_E = D_Cl*alpha_E_e

        # calculate source terms and boundary terms - Na concentration
        f_Na_N, J_Na_N = self.get_src_term_k(alpha_N_e, Na_N_e, phi_N_e, \
                D_Na_N, z_Na, gamma_NE*J_Na_NE)
        f_Na_E, J_Na_E = self.get_src_term_k(alpha_E_e, Na_E_e, phi_E_e, \
                D_Na_E, z_Na, - gamma_NE*J_Na_NE)

        # calculate source terms and boundary terms - K concentration
        f_K_N, J_K_N = self.get_src_term_k(alpha_N_e, K_N_e, phi_N_e, \
                D_K_N, z_K, gamma_NE*J_K_NE)
        f_K_E, J_K_E = self.get_src_term_k(alpha_E_e, K_E_e, phi_E_e, \
                D_K_E, z_K, - gamma_NE*J_K_NE)

        # calculate source terms and boundary terms - Cl concentration
        f_Cl_N, J_Cl_N = self.get_src_term_k(alpha_N_e, Cl_N_e, phi_N_e, \
                D_Cl_N, z_Cl, gamma_NE*J_Cl_NE)
        f_Cl_E, J_Cl_E = self.get_src_term_k(alpha_E_e, Cl_E_e, phi_E_e, \
                D_Cl_E, z_Cl, - gamma_NE*J_Cl_NE)

        # calculate source terms and boundary terms - potentials
        f_phi_N = self.get_src_term_phi(Na_N_e, K_N_e, Cl_N_e, \
                  z_Na, z_K, z_Cl, z_0, a_N, alpha_N_e, \
                  gamma_NE*C_NE*phi_NE_e)
        f_phi_E = self.get_src_term_phi(Na_E_e, K_E_e, Cl_E_e, \
                  z_Na, z_K, z_Cl, z_0, a_E, alpha_E_e, \
                  - gamma_NE*C_NE*phi_NE_e)

        # get physical parameters
        temperature = params['temperature']
        R = params['R']
        F = params['F']
        z_Na = params['z'][0]
        z_K = params['z'][1]
        z_Cl = params['z'][2]
        z_0 = params['z'][3]
        D_Na = params['D'][0]
        D_K = params['D'][1]
        D_Cl = params['D'][2]
        xie_N = params['xie'][0]
        gamma_NE = params['gamma_M'][0]
        nw_NE = params['nw_M'][0]
        C_NE = params['C_M'][0]

        g_Na_leak_N = params['g_Na_leak_N']
        g_K_leak_N = params['g_K_leak_N']
        g_Cl_leak_N = params['g_Cl_leak_N']

        time = self.time

        # convert exact solutions to Expressions
        alphaNe, NaNe, NaEe, KNe, KEe, ClNe, ClEe, phiNe, phiEe  = \
                [Expression(sm.printing.ccode(foo), t=time, degree=4)
                    for foo in (alpha_N_e, Na_N_e, Na_E_e, K_N_e, K_E_e, \
                            Cl_N_e, Cl_E_e, phi_N_e, phi_E_e)]

        falphaN, fNaN, fNaE, fKN, fKE, fClN, fClE, fphiN, fphiE = \
                [Expression(sm.printing.ccode(foo), z_Na=z_Na, z_K=z_K, \
                    z_Cl=z_Cl, z_0=z_0, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl, F=F, R=R, \
                    C_NE=C_NE, nw_NE=nw_NE,\
                    temperature=temperature, gamma_NE=gamma_NE, \
                    g_Na_leak_N=g_Na_leak_N, \
                    g_K_leak_N=g_K_leak_N, \
                    g_Cl_leak_N=g_Cl_leak_N,
                    t=time, xie_N=xie_N, degree=4)
                    for foo in (f_alpha_N, \
                                f_Na_N, f_Na_E, \
                                f_K_N, f_K_E, \
                                f_Cl_N, f_Cl_E,\
                                f_phi_N, f_phi_E)]

        JNaN, JNaE, JKN, JKE, JClN, JClE  = \
                [Expression((sm.printing.ccode(foo)),\
                    z_Na=z_Na, z_K=z_K, z_Cl=z_Cl, z_0=z_0, F=F, R=R,
                    temperature=temperature, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl,
                    xie_N=xie_N, t=time, degree=4)
                    for foo in (J_Na_N, J_Na_E, J_K_N, J_K_E, J_Cl_N,  J_Cl_E)]

        alphaNinit,  NaNinit, NaEinit, KNinit, KEinit, ClNinit, ClEinit, \
                phiNinit, phiEinit = \
                [Expression((sm.printing.ccode(foo)), t=Constant(0.0), degree=4)
                    for foo in (alpha_N_init, Na_N_init, \
                        Na_E_init, K_N_init, K_E_init, \
                        Cl_N_init, Cl_E_init, phi_N_init, \
                        phi_E_init)]

        # gather source terms in FEniCS Expression format
        src_terms = [falphaN, fNaN, fNaE, fKN, fKE, fClN, fClE, fphiN, fphiE]

        # gather boundary terms in FEniCS Expression format
        bndry_terms = [None, JNaN, JNaE, JKN, JKE, JClN, JClE, None, None]

        # gather exact solutions in FEniCS Expression format
        exact_sols = {'alphaNe':alphaNe,
                      'NaNe':NaNe, 'NaEe':NaEe,
                      'KNe':KNe, 'KEe':KEe,
                      'ClNe':ClNe, 'ClEe':ClEe,
                      'phiNe':phiNe, 'phiEe':phiEe}

        # initial conditions in FEniCS Expression format
        init_conds = {'alphaNinit':alphaNinit,
                      'NaNinit':NaNinit, 'NaEinit':NaEinit,
                      'KNinit':KNinit, 'KEinit':KEinit,
                      'ClNinit':ClNinit, 'ClEinit':ClEinit,
                      'phiNinit':phiNinit, 'phiEinit':phiEinit}

        #return all terms in FEniCS Expression format
        return src_terms, bndry_terms, exact_sols, init_conds

    def get_exact_solution(self):
        # define manufactured solutions sins and cos'
        x = self.x; t = self.t
        # volume fraction
        alpha_N_e = 0.3 - 0.1*sm.sin(2*pi*x)*sm.exp(-t) # intracellular
        # sodium (Na) concentration
        Na_N_e = 0.7 + 0.3*sm.sin(pi*x)*sm.exp(-t)      # intracellular
        Na_E_e = 1.0 + 0.6*sm.sin(pi*x)*sm.exp(-t)      # extracellular
        # potassium (K) concentration
        K_N_e = 0.3 + 0.3*sm.sin(pi*x)*sm.exp(-t)       # intracellular
        K_E_e = 1.0 + 0.2*sm.sin(pi*x)*sm.exp(-t)       # extracellular
        # chloride (Cl) concentration
        Cl_N_e = 1.0 + 0.6*sm.sin(pi*x)*sm.exp(-t)      # intracellular
        Cl_E_e = 2.0 + 0.8*sm.sin(pi*x)*sm.exp(-t)      # extracellular
        # potential
        phi_N_e = sm.sin(2*pi*x)*sm.exp(-t)             # intracellular
        phi_E_e = sm.sin(2*pi*x)*(1 + sm.exp(-t))       # extracellular

        exact_solutions = {'alpha_N_e':alpha_N_e, \
                           'Na_N_e':Na_N_e, 'K_N_e':K_N_e, 'Cl_N_e':Cl_N_e, \
                           'Na_E_e':Na_E_e, 'K_E_e':K_E_e, 'Cl_E_e':Cl_E_e, \
                           'phi_N_e':phi_N_e, 'phi_E_e':phi_E_e}

        return exact_solutions

    def get_initial_conditions(self):
        # define manufactured solutions sins and cos'
        x = self.x; t = self.t
        # volume fraction
        alpha_N_init = 0.3 - 0.1*sm.sin(2*pi*x) # intracellular
        # sodium (Na) concentration
        Na_N_init = 0.7 + 0.3*sm.sin(pi*x)  # intracellular
        Na_E_init = 1.0 + 0.6*sm.sin(pi*x)  # extracellular
        # potassium (K) concentration
        K_N_init = 0.3 + 0.3*sm.sin(pi*x)   # intracellular
        K_E_init = 1.0 + 0.2*sm.sin(pi*x)   # extracellular
        # chloride (Cl) concentration
        Cl_N_init = 1.0 + 0.6*sm.sin(pi*x)  # intracellular
        Cl_E_init = 2.0 + 0.8*sm.sin(pi*x)  # extracellular
        # potential
        phi_N_init = sm.sin(2*pi*x)         # intracellular
        phi_E_init = sm.sin(2*pi*x)*2       # extracellular

        initial_conditions = {'alpha_N_init':alpha_N_init,
                              'Na_N_init':Na_N_init,
                              'Na_E_init':Na_E_init,
                              'K_N_init':K_N_init,
                              'K_E_init':K_E_init,
                              'Cl_N_init':Cl_N_init,
                              'Cl_E_init':Cl_E_init,
                              'phi_N_init':phi_N_init,
                              'phi_E_init':phi_E_init}

        return initial_conditions

class ProblemMMS():
    """ Problem for method of manufactured solution (MMS) test """
    def __init__(self, mesh, boundary_point, t_PDE, t_ODE):
        self.mesh = mesh                     # mesh
        self.boundary_point = boundary_point # point to pin phi_E to zero
        self.t_PDE = t_PDE      # time constant (for updating source and boundary terms in PDEs)
        self.t_ODE = t_ODE      # time constant (for updating source and boundary terms in ODEs)
        self.N_ions = 3         # number of ions
        self.N_comparts = 2     # number of compartments
        self.N_states = 0       # number of ODE states
        self.set_parameters()   # parameters

        # get MMS terms at current time step n+1 (BDF2 scheme)
        M = MMS(self.t_PDE)
        source_terms_PDE, boundary_terms, exact_solutions, initial_conditions = \
                M.get_MMS_terms(self.params, 4)
        # set source and boundary terms and exact solutions
        self.source_terms_PDE = source_terms_PDE
        self.boundary_terms = boundary_terms
        self.exact_solutions = exact_solutions
        self.initial_conditions = initial_conditions

        # set initial conditions and number of immobile ions
        self.set_initial_conds_PDE()
        self.set_initial_conds_ODE()
        self.set_immobile_ions()

        return

    def set_parameters(self):
        """ set the problems physical parameters """
        # physical model parameters
        temperature = Constant(310.15) # temperature - (K)
        F = Constant(96485.332)        # Faraday's constant - (C/mol)
        R = Constant(8.3144598)        # gas constant - (J/(mol*K))

        # area of membrane per volume (1/m)
        gamma_NE = Constant(6.3849e5)  # neuron
        gamma_M = [gamma_NE]

        # hydraulic permeability (m/s/(mol/m^3))
        nw_NE = Constant(5.4e-10)      # neuron
        nw_M = [nw_NE]

        # capacitances (F/m^2)
        C_NE = Constant(0.75e-2)       # neuron
        C_M = [C_NE]

        # diffusion coefficients (m^2/s)
        D_Na = Constant(1.33e-9)       # sodium (Na)
        D_K = Constant(1.96e-9)        # potassium (K)
        D_Cl = Constant(2.03e-9)       # chloride (Cl)

        D = [D_Na, D_K, D_Cl]

        # valences
        z_Na = Constant(1.0)           # sodium (Na)
        z_K = Constant(1.0)            # potassium (K)
        z_Cl = Constant(-1.0)          # chloride (Cl)
        z_0 = Constant(-1.0)           # immobile ions
        z = [z_Na, z_K, z_Cl, z_0]

        #scaling factors effective diffusion
        xie_N = Constant(0.0)          # neuron
        xie = [xie_N]

        ################################################################
        # conductivity for leak currents
        g_Na_leak_N = Constant(2.0e-1) # sodium (Na)         - neuron (S/m^2)
        g_K_leak_N  = Constant(7.0e-1) # potassium (K)       - neuron (S/m^2)
        g_Cl_leak_N = Constant(2.0)    # chloride (Cl)       - neuron (S/m^2)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'gamma_M':gamma_M, 'nw_M':nw_M, 'C_M':C_M,
                  'D':D, 'z':z, 'xie':xie,
                  'g_Na_leak_N':g_Na_leak_N, 'g_K_leak_N':g_K_leak_N,
                  'g_Cl_leak_N':g_Cl_leak_N}

        self.params = params
        return

    def set_immobile_ions(self):
        """ define amount of immobile ions """
        # get initial conditions
        for key in self.initial_conditions:
            exec('%s = self.initial_conditions["%s"]' % (key, key))

        # get initial membrane potential
        phiNEinit = phiNinit - phiEinit
        # get initial volume fractions
        alphaEinit = 1.0 - alphaNinit

        F = self.params['F']
        C_NE = self.params['C_M'][0]
        gamma_NE = self.params['gamma_M'][0]
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]
        z_0 = self.params['z'][3]

        # amount of immobile ions neuron (mol/m^3)
        a_N = 1.0/(z_0*F)*gamma_NE*C_NE*phiNEinit \
            - 1.0/z_0*alphaNinit*(z_Na*NaNinit \
                                + z_K*KNinit \
                                + z_Cl*ClNinit)

        # amount of immobile ions ECS (mol/m^3)
        a_E = - 1.0/(z_0*F)*gamma_NE*C_NE*phiNEinit \
              - 1.0/z_0*alphaEinit*(z_Na*NaEinit \
                                  + z_K*KEinit \
                                  + z_Cl*ClEinit)
        # set immobile ions
        a = [a_N, a_E]
        self.params['a'] = a
        return

    def set_initial_conds_PDE(self):
        """ set the PDE problems initial conditions """
        # get initial conditions
        for key in self.initial_conditions:
            exec('%s = self.initial_conditions["%s"]' % (key, key))

        self.inits_PDE = Expression(('alphaNinit',
                                     'NaNinit',  'NaEinit',
                                     'KNinit',  'KEinit',
                                     'ClNinit', 'ClEinit',
                                     'phiNinit', 'phiEinit'),
                                     alphaNinit=alphaNinit, \
                                     NaNinit=NaNinit, NaEinit=NaEinit, \
                                     KNinit=KNinit, KEinit=KEinit, \
                                     ClNinit=ClNinit, ClEinit=ClEinit, \
                                     phiNinit=phiNinit, \
                                     phiEinit=phiEinit, degree=4)
        return

    def set_initial_conds_ODE(self):
        """ set the ODE problems initial conditions """
        self.inits_ODE = None
        return

    def set_membrane_fluxes(self, w, w_, ss):
        """ set the problems transmembrane ion fluxes. Note that the passive
        fluxes are treated implicitly (w_), while active currents are treated
        explicitly (w), except for the gating variables (ss). """

        # Get parameters
        temperature = self.params['temperature']
        R = self.params['R']
        F = self.params['F']
        g_Na_leak_N = self.params['g_Na_leak_N']
        g_K_leak_N = self.params['g_K_leak_N']
        g_Cl_leak_N = self.params['g_Cl_leak_N']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]

        # split unknowns (PDEs)
        alpha_N, Na_N, Na_E, K_N, K_E, Cl_N, Cl_E, phi_N, phi_E = split(w)

        # calculate membrane potentials
        phi_NE = phi_N - phi_E  # neuron (V)

        ################################################################
        # Nernst potential - neuron
        E_Na_N = R*temperature/(F*z_Na)*ln(Na_E/Na_N) # sodium    - (V)
        E_K_N = R*temperature/(F*z_K)*ln(K_E/K_N)     # potassium - (V)
        E_Cl_N = R*temperature/(F*z_Cl)*ln(Cl_E/Cl_N) # chloride  - (V)

        ################################################################
        # Leak currents - neuron
        I_Na_leak_N = g_Na_leak_N*(phi_NE - E_Na_N)  # sodium    - (A/m^2)
        I_K_leak_N = g_K_leak_N*(phi_NE - E_K_N)     # potassium - (A/m^2)
        I_Cl_leak_N = g_Cl_leak_N*(phi_NE - E_Cl_N)  # chloride  - (A/m^2)

        ################################################################
        # Total transmembrane ion currents - neuron
        I_Na_NE = I_Na_leak_N       # sodium    - (A/m^2)
        I_K_NE = I_K_leak_N         # potassium -(A/m^2)
        I_Cl_NE = I_Cl_leak_N       # chloride  - (A/m^2)

        # convert currents currents to flux - neuron
        J_Na_NE = I_Na_NE/(F*z_Na)  # sodium    - (mol/(m^2s))
        J_K_NE = I_K_NE/(F*z_K)     # potassium - (mol/(m^2s))
        J_Cl_NE = I_Cl_NE/(F*z_Cl)  # chloride  - (mol/(m^2s))

        J_M = [[J_Na_NE], [J_K_NE], [J_Cl_NE]]

        # set problem's membrane fluxes
        self.membrane_fluxes = J_M
        return
