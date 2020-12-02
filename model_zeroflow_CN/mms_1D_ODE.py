from dolfin import *
import ufl

from scipy.integrate import odeint
import sympy as sm
import numpy as np

class MMS:
    """ Class for calculating source terms and boundary terms of the system for
    given exact solutions """

    def __init__(self, t_PDE, t_PDE_prev):
        # set time constant
        self.t_PDE = t_PDE
        self.t_PDE_prev = t_PDE_prev

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
        F = self.F; R = self.R; temperature = self.temperature

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

    def get_src_term_gat(self, s, phi_NE_e):
        t = self.t
        # ds/dt = phi_M + f
        f = sm.diff(s, t) - phi_NE_e
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
        m_e = exact_solutions['m_e']
        h_e = exact_solutions['h_e']
        g_e = exact_solutions['g_e']

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
        m_init = initial_conditions['m_init']
        h_init = initial_conditions['h_init']
        g_init = initial_conditions['g_init']

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
        I_Na_NE = - m_e**2 # sodium    - (A/m^2)
        I_K_NE = - h_e**2  # potassium - (A/m^2)
        I_Cl_NE = - g_e**2 # chloride  - (A/m^2)

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

        a_E = - 1.0/(z_0*F)*gamma_NE*C_NE*(phi_NE_init) \
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

        # calculate source terms - gating
        f_m = self.get_src_term_gat(m_e, phi_NE_e)
        f_h = self.get_src_term_gat(h_e, phi_NE_e)
        f_g = self.get_src_term_gat(g_e, phi_NE_e)

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

        t_PDE = self.t_PDE
        t_PDE_prev = self.t_PDE_prev

        # convert exact solutions to Expressions
        alphaNe, NaNe, NaEe, KNe, KEe, ClNe, ClEe, phiNe, phiEe, me, he, ge = \
                [Expression(sm.printing.ccode(foo), t=t_PDE, degree=4)
                    for foo in (alpha_N_e, Na_N_e, Na_E_e, K_N_e, K_E_e, \
                            Cl_N_e, Cl_E_e, phi_N_e, phi_E_e, m_e, h_e, g_e)]

        # convert source terms to Expressions
        falphaN, fNaN, fNaE, fKN, fKE, fClN, fClE, fphiN, fphiE, fm, fh, fg = \
                [Expression(sm.printing.ccode(foo), z_Na=z_Na, z_K=z_K, \
                    z_Cl=z_Cl, z_0=z_0, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl, F=F, R=R, \
                    C_NE=C_NE, nw_NE=nw_NE,\
                    temperature=temperature, gamma_NE=gamma_NE, \
                    g_Na_leak_N=g_Na_leak_N, \
                    g_K_leak_N=g_K_leak_N, \
                    g_Cl_leak_N=g_Cl_leak_N,
                    t=t_PDE, xie_N=xie_N, degree=4)
                    for foo in (f_alpha_N, \
                                f_Na_N, f_Na_E, \
                                f_K_N, f_K_E, \
                                f_Cl_N, f_Cl_E, \
                                f_phi_N, f_phi_E, \
                                f_m, f_h, f_g)]

        # convert source terms previous to Expressions 
        falphaN_, fNaN_, fNaE_, fKN_, fKE_, fClN_, fClE_, fphiN_, fphiE_, fm_, fh_, fg_ = \
                [Expression(sm.printing.ccode(foo), z_Na=z_Na, z_K=z_K, \
                    z_Cl=z_Cl, z_0=z_0, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl, F=F, R=R, \
                    C_NE=C_NE, nw_NE=nw_NE,\
                    temperature=temperature, gamma_NE=gamma_NE, \
                    g_Na_leak_N=g_Na_leak_N, \
                    g_K_leak_N=g_K_leak_N, \
                    g_Cl_leak_N=g_Cl_leak_N,
                    t=t_PDE_prev, xie_N=xie_N, degree=4)
                    for foo in (f_alpha_N, \
                                f_Na_N, f_Na_E, \
                                f_K_N, f_K_E, \
                                f_Cl_N, f_Cl_E, \
                                f_phi_N, f_phi_E, \
                                f_m, f_h, f_g)]


        # convert boundary terms to Expressions
        JNaN, JNaE, JKN, JKE, JClN, JClE  = \
                [Expression((sm.printing.ccode(foo)),\
                    z_Na=z_Na, z_K=z_K, z_Cl=z_Cl, z_0=z_0, F=F, R=R,
                    temperature=temperature, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl,
                    xie_N=xie_N, t=t_PDE, degree=4)
                    for foo in (J_Na_N, J_Na_E, J_K_N, J_K_E, J_Cl_N, J_Cl_E)]

        # convert boundary terms previous to Expressions
        JNaN_, JNaE_, JKN_, JKE_, JClN_, JClE_  = \
                [Expression((sm.printing.ccode(foo)),\
                    z_Na=z_Na, z_K=z_K, z_Cl=z_Cl, z_0=z_0, F=F, R=R,
                    temperature=temperature, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl,
                    xie_N=xie_N, t=t_PDE_prev, degree=4)
                    for foo in (J_Na_N, J_Na_E, J_K_N, J_K_E, J_Cl_N, J_Cl_E)]



        alphaNinit,  NaNinit, NaEinit, KNinit, KEinit, ClNinit, ClEinit, \
                phiNinit, phiEinit, minit, hinit, ginit = \
                [Expression((sm.printing.ccode(foo)), t=Constant(0.0), degree=4)
                    for foo in (alpha_N_init, Na_N_init, \
                        Na_E_init, K_N_init, K_E_init, \
                        Cl_N_init, Cl_E_init, phi_N_init, \
                        phi_E_init, m_init, h_init, g_init)]

        # gather source terms in FEniCS Expression format
        src_terms_PDE = [falphaN, fNaN, fNaE, fKN, fKE, fClN, fClE, fphiN, fphiE]
        src_terms_PDE_ = [falphaN_, fNaN_, fNaE_, fKN_, fKE_, fClN_, fClE_, fphiN_, fphiE_]
        src_terms_ODE = [fm, fh, fg]

        # gather boundary terms in FEniCS Expression format
        bndry_terms = [None, JNaN, JNaE, JKN, JKE, JClN, JClE, None, None]
        bndry_terms_ = [None, JNaN_, JNaE_, JKN_, JKE_, JClN_, JClE_, None, None]

        # gather exact solutions in FEniCS Expression format
        exact_sols = {'alphaNe':alphaNe,
                      'NaNe':NaNe, 'NaEe':NaEe,
                      'KNe':KNe, 'KEe':KEe,
                      'ClNe':ClNe, 'ClEe':ClEe,
                      'phiNe':phiNe, 'phiEe':phiEe,
                      'me':me, 'he':he, 'ge':ge}

        # initial conditions in FEniCS Expression format
        init_conds = {'alphaNinit':alphaNinit,
                      'NaNinit':NaNinit, 'NaEinit':NaEinit,
                      'KNinit':KNinit, 'KEinit':KEinit,
                      'ClNinit':ClNinit, 'ClEinit':ClEinit,
                      'phiNinit':phiNinit, 'phiEinit':phiEinit,
                      'minit':minit, 'hinit':hinit, 'ginit':ginit}

        #return all terms in FEniCS Expression format
        return src_terms_PDE, src_terms_PDE_, src_terms_ODE, \
                bndry_terms, bndry_terms_, exact_sols, init_conds

    def get_exact_solution(self):
        # define manufactured solutions sins and cos'
        x = self.x; t = self.t
        # volume fraction
        alpha_N_e = 0.3 - 0.1*sm.sin(2*pi*x)*sm.exp(-t) # intracellular
        # sodium (Na) concentration
        Na_N_e = 0.7 + 0.3*sm.sin(pi*x)*sm.exp(-t)  # intracellular
        Na_E_e = 1.0 + 0.6*sm.sin(pi*x)*sm.exp(-t)  # extracellular
        # potassium (K) concentration
        K_N_e = 0.3 + 0.3*sm.sin(pi*x)*sm.exp(-t)   # intracellular
        K_E_e = 1.0 + 0.2*sm.sin(pi*x)*sm.exp(-t)   # extracellular
        # chloride (Cl) concentration
        Cl_N_e = 1.0 + 0.6*sm.sin(pi*x)*sm.exp(-t)  # intracellular
        Cl_E_e = 2.0 + 0.8*sm.sin(pi*x)*sm.exp(-t)  # extracellular
        # potential
        phi_N_e = sm.sin(2*pi*x)*sm.exp(-t)         # intracellular
        phi_E_e = sm.sin(2*pi*x)*(1 + sm.exp(-t))   # extracellular

        m_e = sm.cos(t)*sm.cos(pi*x)
        h_e = sm.cos(t)*sm.cos(pi*x)
        g_e = sm.cos(t)*sm.cos(pi*x)

        exact_solutions = {'alpha_N_e':alpha_N_e, \
                           'Na_N_e':Na_N_e, 'K_N_e':K_N_e, 'Cl_N_e':Cl_N_e, \
                           'Na_E_e':Na_E_e, 'K_E_e':K_E_e, 'Cl_E_e':Cl_E_e, \
                           'phi_N_e':phi_N_e, 'phi_E_e':phi_E_e, \
                           'm_e':m_e, 'h_e':h_e, 'g_e':g_e}

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

        m_init = sm.cos(0.0)*sm.cos(pi*x)
        h_init = sm.cos(0.0)*sm.cos(pi*x)
        g_init = sm.cos(0.0)*sm.cos(pi*x)

        initial_conditions = {'alpha_N_init':alpha_N_init,
                              'Na_N_init':Na_N_init,
                              'Na_E_init':Na_E_init,
                              'K_N_init':K_N_init,
                              'K_E_init':K_E_init,
                              'Cl_N_init':Cl_N_init,
                              'Cl_E_init':Cl_E_init,
                              'phi_N_init':phi_N_init,
                              'phi_E_init':phi_E_init,
                              'm_init':m_init,
                              'h_init':h_init,
                              'g_init':g_init}

        return initial_conditions

class ProblemMMS():
    """ Problem for method of manufactured solution (MMS) test """
    def __init__(self, mesh, boundary_point, t_PDE, t_PDE_prev):
        self.mesh = mesh                # mesh
        self.boundary_point = boundary_point # point to pin phi_E to zero
        self.t_PDE = t_PDE              # time constant (for updating source and boundary terms)
        self.t_PDE_prev = t_PDE_prev    # time constant (for updating source and boundary terms)
        self.t_ODE = Constant(0.0)      # time constant (for updating source and boundary terms)
        self.N_ions = 3                 # number of ions
        self.N_comparts = 2             # number of compartments
        self.N_states = 3               # number of ODE states
        self.set_parameters()           # parameters

        # get MMS terms
        M = MMS(self.t_PDE, self.t_PDE_prev)
        source_terms_PDE, source_terms_PDE_, \
                source_terms_ODE, \
                boundary_terms, boundary_terms_, \
                exact_solutions, initial_conditions = M.get_MMS_terms(self.params, 4)
        # set source and boundary terms and exact solutions
        self.source_terms_PDE = source_terms_PDE
        self.source_terms_PDE_ = source_terms_PDE_
        self.source_terms_ODE = source_terms_ODE
        self.boundary_terms = boundary_terms
        self.boundary_terms_ = boundary_terms_
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
        # get initial conditions
        for key in self.initial_conditions:
            exec('%s = self.initial_conditions["%s"]' % (key, key))

        self.inits_ODE = Expression(('minit', 'hinit', 'ginit'), \
                minit=minit, hinit=hinit, ginit=ginit, degree=4)
        return

    def voltage_gated_currents(self, ss):
        """ Voltage gated currents - neuron (I_NaP, I_KDR, I_KA) """
        # get physical parameters
        # split previous solution (ODEs)
        m_, h_, g_= split(ss)

        # define and return currents
        return m_, h_, g_

    def get_membrane_fluxes(self, w, ss_):
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

        # Nernst potential - neuron
        E_Na_N = R*temperature/(F*z_Na)*ln(Na_E/Na_N) # sodium    - (V)
        E_K_N = R*temperature/(F*z_K)*ln(K_E/K_N)     # potassium - (V)
        E_Cl_N = R*temperature/(F*z_Cl)*ln(Cl_E/Cl_N) # chloride  - (V)

        ################################################################
        # Total transmembrane ion currents - neuron
        m_, h_, g_ = self.voltage_gated_currents(ss_)

        I_Na_NE = - m_**2         # sodium    - (A/m^2)
        I_K_NE = - h_**2          # potassium -(A/m^2)
        I_Cl_NE = - g_**2         # chloride  - (A/m^2)

        # convert currents currents to flux - neuron
        J_Na_NE = I_Na_NE/(F*z_Na)      # sodium    - (mol/(m^2s))
        J_K_NE = I_K_NE/(F*z_K)         # potassium - (mol/(m^2s))
        J_Cl_NE = I_Cl_NE/(F*z_Cl)      # chloride  - (mol/(m^2s))

        J_M = [[J_Na_NE], [J_K_NE], [J_Cl_NE]]

        # set problem's membrane fluxes
        return J_M

    def F(self, w_, ss, src_terms_ODE, time=None):
        """ Right hand side of the ODE system """
        time = time if time else Constant(0.0)

        f_m = src_terms_ODE[0]
        f_h = src_terms_ODE[1]
        f_g = src_terms_ODE[2]

        # Assign states
        assert(len(ss) == self.N_states)
        m_, h_, g_ = ss

        alpha_N, Na_N, Na_E, K_N, K_E, Cl_N, Cl_E, phi_N, phi_E = split(w_)
        phi_NE = phi_N - phi_E

        # Initial return arguments
        F_expressions = [ufl.zero()]*self.N_states

        # Expressions
        F_expressions[0] = phi_NE + f_m
        F_expressions[1] = phi_NE + f_h
        F_expressions[2] = phi_NE + f_g

        # Return results
        return as_vector(F_expressions)
