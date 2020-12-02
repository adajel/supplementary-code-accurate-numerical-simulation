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
        gamma_NE, gamma_GE, C_NE, C_GE = sm.symbols('gamma_NE gamma_GE C_NE C_GE')
        self.F = F
        self.R = R
        self.temperature = temperature
        self.gamma_NE = gamma_NE
        self.gamma_GE = gamma_GE
        self.C_NE = C_NE
        self.C_GE = C_GE

    def div(self, f):
        """ calculate divergence of u """
        x = self.x
        # calculate divergence
        div_f = sm.diff(f, x)
        return div_f

    def get_src_term_alpha(self, alpha_r, u_r, memflux):
        """ calculate source terms for alpha """
        x = self.x; t = self.t
        # calculate source f term: fI = d alpha_I/dt + div(alpha_I u_I) + gamma_M*w_M
        f = sm.diff(alpha_r, t) + sm.diff(alpha_r*u_r, x)  + memflux
        # return source term and adjusted compartmental ion flux
        return f

    def get_src_term_k(self, alpha_r, k_r, phi_r, D_r, z_r, u_r, memflux):
        """ calculate source and boundary terms for conservation of ions """
        x = self.x; t = self.t
        F = self.F; R = self.R; temperature = self.temperature
        # calculate gradients
        grad_k_r = sm.diff(k_r, x)
        grad_phi_r = sm.diff(phi_r, x)
        # calculate compartmental flux
        J_k_r = - D_r*grad_k_r - D_r*z_r*F/(R*temperature)*k_r*grad_phi_r \
                + alpha_r*u_r*k_r

        # calculate source f term: fI = dk_r/dt + div (J_kr) - gamma_M J_M
        # calculate source f term: fE = dk_r/dt + div (J_kr) + gamma_M J_M
        f = sm.diff(alpha_r*k_r, t) + self.div(J_k_r) + memflux
        # return source term and adjusted compartmental ion flux
        return f, J_k_r

    def get_src_term_phi(self, Na_r, K_r, Cl_r, z_Na, z_K, z_Cl, z_0, a_r, alpha_r, memflux):
        # Function for calculating source terms for equations for potentials
        F = self.F
        # calculate source term by: fI = rho_0 + F*sum([k]_r z alpha_r) + memflux
        f = z_0*F*a_r + F*alpha_r*(z_Na*Na_r + z_K*K_r + z_Cl*Cl_r) + memflux
        # return source term
        return f

    def get_source_term_pE(self, alpha_N, alpha_G, alpha_E, u_N, u_G, u_E):
        # Function for calculating source and boundary terms for p_E
        x = self.x
        # calculate source f term: fE = div(sum_r alpha_r u_r)
        f = self.div(alpha_N*u_N + alpha_G*u_G + alpha_E*u_E) \
        # boundary term
        J_p_E = alpha_N*u_N + alpha_G*u_G + alpha_E*u_E

        # return source term and adjusted compartmental ion flux
        return f, J_p_E

    def get_MMS_terms(self, params, degree):
        # return source terms and boundary terms for all equations
        x = self.x; t = self.t
        F = self.F; R = self.R; temperature = self.temperature;
        gamma_NE = self.gamma_NE; gamma_GE = self.gamma_GE
        C_NE = self.C_NE; C_GE = self.C_GE
        # define constants
        D_Na, D_K, D_Cl = sm.symbols('D_Na D_K D_Cl')          # diffusion coefficients
        z_Na, z_K, z_Cl, z_0 = sm.symbols('z_Na z_K z_Cl z_0') # valence
        nw_NE = sm.symbols('nw_NE')
        nw_GE = sm.symbols('nw_GE')
        xie_N = sm.symbols('xie_N')
        xie_G = sm.symbols('xie_G')

        S_NE, S_GE = sm.symbols('S_NE S_GE')
        kappa_N, kappa_G, kappa_E = sm.symbols('kappa_N kappa_G kappa_E')

        g_Na_leak_N = sm.symbols('g_Na_leak_N') # conductance
        g_K_leak_N = sm.symbols('g_K_leak_N')   # conductance
        g_Cl_leak_N = sm.symbols('g_Cl_leak_N') # conductance
        g_Na_leak_G = sm.symbols('g_Na_leak_G') # conductance
        g_KIR_0 = sm.symbols('g_KIR_0')         # conductance
        g_Cl_leak_G = sm.symbols('g_Cl_leak_G') # conductance

        # set manufactured solution and initial conditions
        exact_solutions = self.get_exact_solution()
        initial_conditions = self.get_initial_conditions()

        # unwrap exact solutions
        for key in exact_solutions:
            exec('%s = exact_solutions["%s"]' % (key, key))

        for key in initial_conditions:
            exec('%s = initial_conditions["%s"]' % (key, key))

        # membrane potential
        phi_NE_e = phi_N_e - phi_E_e
        phi_GE_e = phi_G_e - phi_E_e
        # define ECS volume fraction
        alpha_E_e = 1.0 - alpha_N_e - alpha_G_e

        # initial membrane potential
        phi_NE_init = phi_N_init - phi_E_init
        phi_GE_init = phi_G_init - phi_E_init
        # define initial ECS volume fraction
        alpha_E_init = 1.0 - alpha_N_init - alpha_G_init

        ################################################################
        # Nernst potential - neuron
        E_Na_N = R*temperature/(F*z_Na)*sm.log(Na_E_e/Na_N_e) # sodium    - (V)
        E_K_N = R*temperature/(F*z_K)*sm.log(K_E_e/K_N_e)     # potassium - (V)
        E_Cl_N = R*temperature/(F*z_Cl)*sm.log(Cl_E_e/Cl_N_e) # chloride  - (V)

        # Nernst potential - glial
        E_Na_G = R*temperature/(F*z_Na)*sm.log(Na_E_e/Na_G_e) # sodium    - (V)
        E_K_G = R*temperature/(F*z_K)*sm.log(K_E_e/K_G_e)     # potassium - (V)
        E_Cl_G = R*temperature/(F*z_Cl)*sm.log(Cl_E_e/Cl_G_e) # chloride  - (V)

        ################################################################
        # Potassium inward rectifier (KIR) current - glial
        A = 18.5/42.5                               # shorthand
        B = 1.0e3*(phi_GE_e - E_K_G + 18.5e-3)/42.5 # shorthand
        C = (-118.6 - 85.2)/44.1                    # shorthand
        D = 1.0e3*(-118.6e-3 + phi_GE_e)/44.1       # shorthand

        # inward rectifying conductance
        g_KIR = sm.sqrt(K_G_e/K_G_init)*(1 + sm.exp(A))/(1 + sm.exp(B))*\
                                        (1 + sm.exp(C))/(1 + sm.exp(D))
        # Leak currents - neuron
        I_Na_NE = g_Na_leak_N*(phi_NE_e - E_Na_N)  # leak sodium    - (A/m^2)
        I_K_NE = g_K_leak_N*(phi_NE_e - E_K_N)     # leak potassium - (A/m^2)
        I_Cl_NE = g_Cl_leak_N*(phi_NE_e - E_Cl_N)  # leak chloride  - (A/m^2)

        # Leak currents - glial
        I_Na_GE = g_Na_leak_G*(phi_GE_e - E_Na_G)  # leak sodium         - (A/m^2)
        I_K_GE = g_KIR_0*g_KIR*(phi_GE_e - E_K_G)  # inward rectifying K - (A/m^2)
        I_Cl_GE = g_Cl_leak_G*(phi_GE_e - E_Cl_G)  # leak chloride       - (A/m^2)

        ################################################################
        # convert currents currents to flux - neuron
        J_Na_NE = I_Na_NE/(F*z_Na) # sodium    - (mol/(m^2s))
        J_K_NE = I_K_NE/(F*z_K)    # potassium - (mol/(m^2s))
        J_Cl_NE = I_Cl_NE/(F*z_Cl) # chloride  - (mol/(m^2s))

        # convert currents currents to flux - glial
        J_Na_GE = I_Na_GE/(F*z_Na) # sodium    - (mol/(m^2s))
        J_K_GE = I_K_GE/(F*z_K)    # potassium - (mol/(m^2s))
        J_Cl_GE = I_Cl_GE/(F*z_Cl) # chloride  - (mol/(m^2s))

        # define amount of immobile ions
        # amount of immobile ions neuron (mol/m^3)
        a_N = 1.0/(z_0*F)*gamma_NE*C_NE*phi_NE_init \
            - 1.0/z_0*alpha_N_init*(z_Na*Na_N_init \
                                  + z_K*K_N_init \
                                  + z_Cl*Cl_N_init)

        a_G = 1.0/(z_0*F)*gamma_GE*C_GE*phi_GE_init \
            - 1.0/z_0*alpha_G_init*(z_Na*Na_G_init \
                                  + z_K*K_G_init \
                                  + z_Cl*Cl_G_init)

        a_E = - 1.0/(z_0*F)*gamma_NE*C_NE*(phi_NE_init) \
              - 1.0/(z_0*F)*gamma_GE*C_GE*(phi_GE_init) \
              - 1.0/z_0*alpha_E_init*(z_Na*Na_E_init \
                                    + z_K*K_E_init \
                                    + z_Cl*Cl_E_init)

        # mechanical tension per unit area of membrane
        tau_N = S_NE*(alpha_N_e - alpha_N_init) # mechanical tension neuron
        tau_G = S_GE*(alpha_G_e - alpha_G_init) # mechanical tension glial

        alpha_E_init  = 1.0 - alpha_N_init - alpha_G_init

        # expressions for hydrostatic pressures
        p_N = p_E_e + tau_N
        p_G = p_E_e + tau_G

        # expressions for total pressures
        p_hat_N = p_N - R*temperature*a_N/alpha_N_e
        p_hat_G = p_G - R*temperature*a_G/alpha_G_e
        p_hat_E = p_E_e - R*temperature*a_E/alpha_E_e

        w_NE = nw_NE*(p_E_e - p_N \
                + R*temperature*(a_E/alpha_E_e + Na_E_e + K_E_e + Cl_E_e \
                - a_N/alpha_N_e - Na_N_e - K_N_e - Cl_N_e))
        # transmembrane water flux glial
        w_GE = nw_GE*(p_E_e - p_G \
                + R*temperature*(a_E/alpha_E_e + Na_E_e + K_E_e + Cl_E_e \
                - a_G/alpha_G_e - Na_G_e - K_G_e - Cl_G_e))

        #calculate gradients
        grad_phi_N = sm.diff(phi_N_e, x)
        grad_phi_G = sm.diff(phi_G_e, x)
        grad_phi_E = sm.diff(phi_E_e, x)

        grad_p_N = sm.diff(p_hat_N, x)
        grad_p_G = sm.diff(p_hat_G, x)
        grad_p_E = sm.diff(p_hat_E, x)

        # compartmental fluid velocity
        u_N = - kappa_N*((grad_p_N) + F*grad_phi_N*(z_Na*Na_N_e + z_K*K_N_e + z_Cl*Cl_N_e))
        u_G = - kappa_G*((grad_p_G) + F*grad_phi_G*(z_Na*Na_G_e + z_K*K_G_e + z_Cl*Cl_G_e))
        u_E = - kappa_E*((grad_p_E) + F*grad_phi_E*(z_Na*Na_E_e + z_K*K_E_e + z_Cl*Cl_E_e))

        # calculate source terms and boundary terms - alphas
        f_alpha_N = self.get_src_term_alpha(alpha_N_e, u_N, gamma_NE*w_NE)
        f_alpha_G = self.get_src_term_alpha(alpha_G_e, u_G, gamma_GE*w_GE)

        # define effective diffusion coefficients
        D_Na_N = D_Na*xie_N
        D_K_N = D_K*xie_N
        D_Cl_N = D_Cl*xie_N

        D_Na_G = D_Na*xie_G
        D_K_G = D_K*xie_G
        D_Cl_G = D_Cl*xie_G

        D_Na_E = D_Na*alpha_E_e
        D_K_E = D_K*alpha_E_e
        D_Cl_E = D_Cl*alpha_E_e

        # calculate source terms and boundary terms - Na concentration
        f_Na_N, J_Na_N = self.get_src_term_k(alpha_N_e, Na_N_e, phi_N_e, \
                D_Na_N, z_Na, u_N, gamma_NE*J_Na_NE)
        f_Na_G, J_Na_G = self.get_src_term_k(alpha_G_e, Na_G_e, phi_G_e, \
                D_Na_G, z_Na, u_G, gamma_GE*J_Na_GE)
        f_Na_E, J_Na_E = self.get_src_term_k(alpha_E_e, Na_E_e, phi_E_e, \
                D_Na_E, z_Na, u_E, - gamma_NE*J_Na_NE - gamma_GE*J_Na_GE)

        # calculate source terms and boundary terms - K concentration
        f_K_N, J_K_N = self.get_src_term_k(alpha_N_e, K_N_e, phi_N_e, \
                D_K_N, z_K, u_N, gamma_NE*J_K_NE)
        f_K_G, J_K_G = self.get_src_term_k(alpha_G_e, K_G_e, phi_G_e, \
                D_K_G, z_K, u_G, gamma_GE*J_K_GE)
        f_K_E, J_K_E = self.get_src_term_k(alpha_E_e, K_E_e, phi_E_e, \
                D_K_E, z_K, u_E, - gamma_NE*J_K_NE - gamma_GE*J_K_GE)

        # calculate source terms and boundary terms - Cl concentration
        f_Cl_N, J_Cl_N = self.get_src_term_k(alpha_N_e, Cl_N_e, phi_N_e, \
                D_Cl_N, z_Cl, u_N, gamma_NE*J_Cl_NE)
        f_Cl_G, J_Cl_G = self.get_src_term_k(alpha_G_e, Cl_G_e, phi_G_e, \
                D_Cl_G, z_Cl, u_G, gamma_GE*J_Cl_GE)
        f_Cl_E, J_Cl_E = self.get_src_term_k(alpha_E_e, Cl_E_e, phi_E_e, \
                D_Cl_E, z_Cl, u_E, - gamma_NE*J_Cl_NE - gamma_GE*J_Cl_GE)

        # calculate source terms and boundary terms - potentials
        f_phi_N = self.get_src_term_phi(Na_N_e, K_N_e, Cl_N_e, \
                  z_Na, z_K, z_Cl, z_0, a_N, alpha_N_e, \
                  - gamma_NE*C_NE*phi_NE_e)
        f_phi_G = self.get_src_term_phi(Na_G_e, K_G_e, Cl_G_e, \
                  z_Na, z_K, z_Cl, z_0, a_G, alpha_G_e, \
                  - gamma_GE*C_GE*phi_GE_e)
        f_phi_E = self.get_src_term_phi(Na_E_e, K_E_e, Cl_E_e, \
                  z_Na, z_K, z_Cl, z_0, a_E, alpha_E_e, \
                  gamma_NE*C_NE*phi_NE_e + gamma_GE*C_GE*phi_GE_e)

        # calculate source term and boundary term - extracellular pressure
        f_p_E, J_p_E = self.get_source_term_pE(alpha_N_e, alpha_G_e, alpha_E_e, u_N, u_G, u_E)

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
        xie_G = params['xie'][1]
        kappa_N = params['kappa'][0]
        kappa_G = params['kappa'][1]
        kappa_E = params['kappa'][2]
        gamma_NE = params['gamma_M'][0]
        gamma_GE = params['gamma_M'][1]
        nw_NE = params['nw_M'][0]
        nw_GE = params['nw_M'][1]
        C_NE = params['C_M'][0]
        C_GE = params['C_M'][1]
        S_NE = params['S_M'][0]
        S_GE = params['S_M'][1]

        g_Na_leak_N = params['g_Na_leak_N']
        g_K_leak_N = params['g_K_leak_N']
        g_Cl_leak_N = params['g_Cl_leak_N']
        g_Na_leak_G = params['g_Na_leak_G']
        g_Cl_leak_G = params['g_Cl_leak_G']
        g_KIR_0 = params['g_KIR_0']

        time = self.time

        # convert exact solutions to Expressions
        alphaNe, alphaGe, NaNe, NaGe, NaEe, KNe, KGe, KEe, \
                ClNe, ClGe, ClEe, phiNe, phiGe, phiEe, pEe  = \
                [Expression(sm.printing.ccode(foo), t=time, degree=4)
                    for foo in (alpha_N_e, alpha_G_e, Na_N_e, Na_G_e,\
                        Na_E_e, K_N_e, K_G_e, K_E_e, Cl_N_e, Cl_G_e, Cl_E_e,\
                        phi_N_e, phi_G_e, phi_E_e, p_E_e)]

        falphaN, falphaG, fNaN, fNaG, fNaE, fKN, fKG, fKE, fClN, \
                fClG, fClE, fphiN, fphiG, fphiE, fpE = \
                [Expression(sm.printing.ccode(foo), z_Na=z_Na, z_K=z_K, \
                    z_Cl=z_Cl, z_0=z_0, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl, F=F, R=R, \
                    C_NE=C_NE, C_GE=C_GE,\
                    temperature=temperature, gamma_NE=gamma_NE, \
                    gamma_GE=gamma_GE, nw_NE=nw_NE, nw_GE=nw_GE, \
                    g_Na_leak_N=g_Na_leak_N, g_Na_leak_G=g_Na_leak_G,\
                    g_K_leak_N=g_K_leak_N, g_KIR_0=g_KIR_0, \
                    g_Cl_leak_N=g_Cl_leak_N, g_Cl_leak_G=g_Cl_leak_G, \
                    t=time, xie_N=xie_N, xie_G=xie_G, \
                    kappa_N=kappa_N, kappa_G=kappa_G, kappa_E=kappa_E, \
                    S_NE=S_NE, S_GE=S_GE, \
                    degree=4)
                    for foo in (f_alpha_N, f_alpha_G, \
                                f_Na_N, f_Na_G, f_Na_E, \
                                f_K_N, f_K_G, f_K_E, \
                                f_Cl_N, f_Cl_G, f_Cl_E,\
                                f_phi_N, f_phi_G, f_phi_E, f_p_E)]

        JNaN, JNaG, JNaE, JKN, JKG, JKE, JClN, JClG, JClE, JpE  = \
                [Expression(sm.printing.ccode(foo),\
                    z_Na=z_Na, z_K=z_K, z_Cl=z_Cl, z_0=z_0, F=F, R=R,
                    temperature=temperature, D_Na=D_Na, D_K=D_K, D_Cl=D_Cl,
                    xie_N=xie_N, xie_G=xie_G, 
                    kappa_N=kappa_N, kappa_G=kappa_G, kappa_E=kappa_E,
                    S_NE=S_NE, S_GE=S_GE,
                    gamma_NE=gamma_NE, gamma_GE=gamma_GE,
                    C_NE=C_NE, C_GE=C_GE,
                    t=time, degree=4)
                    for foo in (J_Na_N, J_Na_G, J_Na_E, J_K_N, J_K_G, J_K_E, \
                            J_Cl_N, J_Cl_G, J_Cl_E, J_p_E)]

        alphaNinit, alphaGinit, NaNinit, NaGinit, NaEinit, KNinit,\
                KGinit, KEinit, ClNinit, ClGinit, ClEinit, \
                phiNinit, phiGinit, phiEinit, pEinit = \
                [Expression((sm.printing.ccode(foo)), t=Constant(0.0), degree=4)
                    for foo in (alpha_N_init, alpha_G_init, Na_N_init, \
                        Na_G_init, Na_E_init, K_N_init, K_G_init, K_E_init, \
                        Cl_N_init, Cl_G_init, Cl_E_init, phi_N_init, \
                        phi_G_init, phi_E_init, p_E_init)]

        # gather source terms in FEniCS Expression format
        src_terms = [falphaN, falphaG, fNaN, fNaG, fNaE, fKN, fKG, fKE,
                     fClN, fClG, fClE, fphiN, fphiG, fphiE, fpE]

        # gather boundary terms in FEniCS Expression format
        bndry_terms = [None, None, JNaN, JNaG, JNaE, JKN, JKG, JKE,
                        JClN, JClG, JClE, None, None, None, JpE]

        # gather exact solutions in FEniCS Expression format
        exact_sols = {'alphaNe':alphaNe, 'alphaGe':alphaGe,
                      'NaNe':NaNe, 'NaGe':NaGe, 'NaEe':NaEe,
                      'KNe':KNe, 'KGe':KGe, 'KEe':KEe,
                      'ClNe':ClNe,  'ClGe':ClGe, 'ClEe':ClEe,
                      'phiNe':phiNe, 'phiGe':phiGe, 'phiEe':phiEe, 
                      'pEe':pEe}

        # initial conditions in FEniCS Expression format
        init_conds = {'alphaNinit':alphaNinit, 'alphaGinit':alphaGinit,
                      'NaNinit':NaNinit, 'NaGinit':NaGinit, 'NaEinit':NaEinit,
                      'KNinit':KNinit, 'KGinit':KGinit, 'KEinit':KEinit,
                      'ClNinit':ClNinit, 'ClGinit':ClGinit, 'ClEinit':ClEinit,
                      'phiNinit':phiNinit, 'phiGinit':phiGinit,
                      'phiEinit':phiEinit, 'pEinit':pEinit}

        #return all terms in FEniCS Expression format
        return src_terms, bndry_terms, exact_sols, init_conds

    def get_exact_solution(self):
        # define manufactured solutions sins and cos'
        x = self.x; t = self.t
        # volume fraction
        alpha_N_e = 0.3 - 0.1*sm.sin(2*pi*x)#*sm.exp(-t) # intracellular
        alpha_G_e = 0.2 - 0.1*sm.sin(2*pi*x)#*sm.exp(-t) # intracellular
        # sodium (Na) concentration
        Na_N_e = 0.7 + 0.3*sm.sin(pi*x)*sm.exp(-t)      # intracellular
        Na_G_e = 0.5 + 0.6*sm.sin(pi*x)*sm.exp(-t)      # intracellular
        Na_E_e = 1.0 + 0.6*sm.sin(pi*x)*sm.exp(-t)      # extracellular
        # potassium (K) concentration
        K_N_e = 0.3 + 0.3*sm.sin(pi*x)*sm.exp(-t)       # intracellular
        K_G_e = 0.5 + 0.2*sm.sin(pi*x)*sm.exp(-t)       # intracellular
        K_E_e = 1.0 + 0.2*sm.sin(pi*x)*sm.exp(-t)       # extracellular
        # chloride (Cl) concentration
        Cl_N_e = 1.0 + 0.6*sm.sin(pi*x)*sm.exp(-t)      # intracellular
        Cl_G_e = 1.0 + 0.8*sm.sin(pi*x)*sm.exp(-t)      # intracellular
        Cl_E_e = 2.0 + 0.8*sm.sin(pi*x)*sm.exp(-t)      # extracellular
        # potential
        phi_N_e = sm.sin(2*pi*x)*sm.exp(-t)             # intracellular
        phi_G_e = sm.sin(2*pi*x)*sm.exp(-t)             # intracellular
        phi_E_e = sm.sin(2*pi*x)*(1 + sm.exp(-t))       # extracellular

        p_E_e = sm.sin(2*pi*x)*sm.exp(-t)               # extracellular

        exact_solutions = {'alpha_N_e':alpha_N_e, 'alpha_G_e':alpha_G_e, \
                           'Na_N_e':Na_N_e, 'K_N_e':K_N_e, 'Cl_N_e':Cl_N_e, \
                           'Na_G_e':Na_G_e, 'K_G_e':K_G_e, 'Cl_G_e':Cl_G_e, \
                           'Na_E_e':Na_E_e, 'K_E_e':K_E_e, 'Cl_E_e':Cl_E_e, \
                           'phi_N_e':phi_N_e, 'phi_G_e':phi_G_e, \
                           'phi_E_e':phi_E_e, 'p_E_e':p_E_e}

        return exact_solutions

    def get_initial_conditions(self):
        # define manufactured solutions sins and cos'
        x = self.x; t = self.t
        # volume fraction
        alpha_N_init = 0.3 - 0.1*sm.sin(2*pi*x) # intracellular
        alpha_G_init = 0.2 - 0.1*sm.sin(2*pi*x) # intracellular
        # sodium (Na) concentration
        Na_N_init = 0.7 + 0.3*sm.sin(pi*x)      # intracellular
        Na_G_init = 0.5 + 0.6*sm.sin(pi*x)      # intracellular
        Na_E_init = 1.0 + 0.6*sm.sin(pi*x)      # extracellular
        # potassium (K) concentration
        K_N_init = 0.3 + 0.3*sm.sin(pi*x)       # intracellular
        K_G_init = 0.5 + 0.2*sm.sin(pi*x)       # intracellular
        K_E_init = 1.0 + 0.2*sm.sin(pi*x)       # extracellular
        # chloride (Cl) concentration
        Cl_N_init = 1.0 + 0.6*sm.sin(pi*x)      # intracellular
        Cl_G_init = 1.0 + 0.8*sm.sin(pi*x)      # intracellular
        Cl_E_init = 2.0 + 0.8*sm.sin(pi*x)      # extracellular
        # potential
        phi_N_init = sm.sin(2*pi*x)             # intracellular
        phi_G_init = sm.sin(2*pi*x)             # intracellular
        phi_E_init = sm.sin(2*pi*x)*2           # extracellular
        # hydraulic pressure
        p_E_init = sm.sin(2*pi*x)               # extracellular

        initial_conditions = {'alpha_N_init':alpha_N_init,
                              'alpha_G_init':alpha_G_init,
                              'Na_N_init':Na_N_init,
                              'Na_G_init':Na_G_init,
                              'Na_E_init':Na_E_init,
                              'K_N_init':K_N_init,
                              'K_G_init':K_G_init,
                              'K_E_init':K_E_init,
                              'Cl_N_init':Cl_N_init,
                              'Cl_G_init':Cl_G_init,
                              'Cl_E_init':Cl_E_init,
                              'phi_N_init':phi_N_init,
                              'phi_G_init':phi_G_init,
                              'phi_E_init':phi_E_init, 
                              'p_E_init':p_E_init}

        return initial_conditions

class ProblemMMS():
    """ Problem for method of manufactured solution (MMS) test """
    def __init__(self, mesh, boundary_point, t_PDE, t_ODE):
        self.mesh = mesh        # mesh
        self.boundary_point = boundary_point # point to pin phi_E to zero
        self.t_PDE = t_PDE      # time constant (for updating source and boundary terms)
        self.t_ODE = t_ODE      # time constant (for updating source and boundary terms)
        self.N_ions = 3         # number of ions
        self.N_comparts = 3     # number of compartments
        self.N_states = 0       # number of ODE states
        self.set_parameters()   # parameters

        # get MMS terms
        M = MMS(self.t_PDE)
        source_terms, boundary_terms, exact_solutions, initial_conditions = \
                M.get_MMS_terms(self.params, 4)
        # set source and boundary terms and exact solutions
        self.source_terms = source_terms
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
        gamma_GE = Constant(6.3849e5)  # glial
        gamma_M = [gamma_NE, gamma_GE]

        # hydraulic permeability (m/s/(mol/m^3))
        nw_NE = Constant(5.4e-10)      # neuron
        nw_GE = Constant(5.4e-10)      # glial
        nw_M = [nw_NE, nw_GE]

        S_NE = Constant(0.0)          # membrane stiffness - neuron (Pa/m^3)
        S_GE = Constant(0.0)          # membrane stiffness - glial  (Pa/m^3)
        S_M = [S_NE, S_GE]

        # capacitances (F/m^2)
        C_NE = Constant(0.75e-2)       # neuron
        C_GE = Constant(0.75e-2)       # glial
        C_M = [C_NE, C_GE]

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
        xie_G = Constant(0.0)          # glial
        xie = [xie_N, xie_G]

        #kappa_N = Constant(5.0e-10)    # hydraulic permeability neuron - (m^4/(N s))
        #kappa_G = Constant(5.0e-10)    # hydraulic permeability glial  - (m^4/(N s))
        #kappa_E = Constant(5.0e-10)    # hydraulic permeability ECS    - (m^4/(N s))
        kappa_N = Constant(5.0e-18)    # hydraulic permeability neuron - (m^4/(N s))
        kappa_G = Constant(5.0e-18)    # hydraulic permeability glial  - (m^4/(N s))
        kappa_E = Constant(5.0e-18)    # hydraulic permeability ECS    - (m^4/(N s))
 
        kappa = [kappa_N, kappa_G, kappa_E]

        ################################################################
        # conductivity for leak currents
        g_Na_leak_N = Constant(2.0e-1) # sodium (Na)         - neuron (S/m^2)
        g_K_leak_N  = Constant(7.0e-1) # potassium (K)       - neuron (S/m^2)
        g_Cl_leak_N = Constant(2.0)    # chloride (Cl)       - neuron (S/m^2)
        g_Na_leak_G = Constant(7.2e-2) # sodium (Na)         - glial  (S/m^2)
        g_Cl_leak_G = Constant(5.0e-1) # chloride (Cl)       - neuron (S/m^2)

        # other membrane mechanisms
        g_KIR_0 = Constant(1.3)        # K inward rectifier  - glial  (S/m^2)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'gamma_M':gamma_M, 'nw_M':nw_M, 'C_M':C_M, 'S_M':S_M,
                  'D':D, 'z':z, 'xie':xie, 'kappa':kappa,
                  'g_Na_leak_N':g_Na_leak_N, 'g_K_leak_N':g_K_leak_N,
                  'g_Cl_leak_N':g_Cl_leak_N,
                  'g_Na_leak_G':g_Na_leak_G, 'g_Cl_leak_G':g_Cl_leak_G,
                  'g_KIR_0':g_KIR_0}

        self.params = params
        return

    def set_immobile_ions(self):
        """ define amount of immobile ions """
        # get initial conditions
        for key in self.initial_conditions:
            exec('%s = self.initial_conditions["%s"]' % (key, key))

        # get initial membrane potential
        phiNEinit = phiNinit - phiEinit
        phiGEinit = phiGinit - phiEinit
        # get initial volume fractions
        alphaEinit = 1.0 - alphaNinit - alphaGinit

        F = self.params['F']
        C_NE = self.params['C_M'][0]
        C_GE = self.params['C_M'][1]
        gamma_NE = self.params['gamma_M'][0]
        gamma_GE = self.params['gamma_M'][1]

        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]
        z_0 = self.params['z'][3]

        # amount of immobile ions neuron (mol/m^3)
        a_N = 1.0/(z_0*F)*gamma_NE*C_NE*phiNEinit \
            - 1.0/z_0*alphaNinit*(z_Na*NaNinit \
                                + z_K*KNinit \
                                + z_Cl*ClNinit)
        # amount of immobile ions glial (mol/m^3)
        a_G = 1.0/(z_0*F)*gamma_GE*C_GE*phiGEinit \
            - 1.0/z_0*alphaGinit*(z_Na*NaGinit \
                                + z_K*KGinit \
                                + z_Cl*ClGinit)
        # amount of immobile ions ECS (mol/m^3)
        a_E = - 1.0/(z_0*F)*gamma_NE*C_NE*phiNEinit \
              - 1.0/(z_0*F)*gamma_GE*C_GE*phiGEinit \
              - 1.0/z_0*alphaEinit*(z_Na*NaEinit \
                                  + z_K*KEinit \
                                  + z_Cl*ClEinit)

        # project onto function space over mesh
        Vi = FunctionSpace(self.mesh, "CG", 1)
        a_N_i = project(a_N, Vi)
        a_G_i = project(a_G, Vi)
        a_E_i = project(a_E, Vi)
        alphaNinit_i = project(alphaNinit, Vi)
        alphaGinit_i = project(alphaGinit, Vi)

        # set immobile ions
        a = [a_N_i, a_G_i, a_E_i]
        self.params['a'] = a

        # initial volume fractions neuron and glial
        alpha_init = [alphaNinit_i, alphaGinit_i]
        self.params['alpha_init'] = alpha_init

        return

    def set_initial_conds_PDE(self):
        """ set the PDE problems initial conditions """
        # get initial conditions
        for key in self.initial_conditions:
            exec('%s = self.initial_conditions["%s"]' % (key, key))

        self.inits_PDE = Expression(('alphaNinit', 'alphaGinit',
                                     'NaNinit', 'NaGinit', 'NaEinit',
                                     'KNinit', 'KGinit', 'KEinit',
                                     'ClNinit', 'ClGinit', 'ClEinit',
                                     'phiNinit', 'phiGinit', 'phiEinit',
                                     'pEinit'),
                                     alphaNinit=alphaNinit, \
                                     alphaGinit=alphaGinit, \
                                     NaNinit=NaNinit, NaGinit=NaGinit, NaEinit=NaEinit, \
                                     KNinit=KNinit, KGinit=KGinit, KEinit=KEinit, \
                                     ClNinit=ClNinit, ClGinit=ClGinit, ClEinit=ClEinit, \
                                     phiNinit=phiNinit, phiGinit=phiGinit, \
                                     phiEinit=phiEinit, pEinit=pEinit, degree=4)
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
        g_Na_leak_G = self.params['g_Na_leak_G']
        g_Cl_leak_G = self.params['g_Cl_leak_G']
        g_KIR_0 = self.params['g_KIR_0']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]

        # split unknowns (PDEs)
        alpha_N, alpha_G, Na_N, Na_G, Na_E, K_N, K_G, K_E, Cl_N, Cl_G, Cl_E, \
                phi_N, phi_G, phi_E, p_E = split(w)

        # calculate membrane potentials
        phi_NE = phi_N - phi_E  # neuron (V)
        phi_GE = phi_G - phi_E  # glial  (V)

        ################################################################
        # Nernst potential - neuron
        E_Na_N = R*temperature/(F*z_Na)*ln(Na_E/Na_N) # sodium    - (V)
        E_K_N = R*temperature/(F*z_K)*ln(K_E/K_N)     # potassium - (V)
        E_Cl_N = R*temperature/(F*z_Cl)*ln(Cl_E/Cl_N) # chloride  - (V)

        # Nernst potential - glial
        E_Na_G = R*temperature/(F*z_Na)*ln(Na_E/Na_G) # sodium    - (V)
        E_K_G = R*temperature/(F*z_K)*ln(K_E/K_G)     # potassium - (V)
        E_Cl_G = R*temperature/(F*z_Cl)*ln(Cl_E/Cl_G) # chloride  - (V)

        ################################################################
        # Leak currents - neuron
        I_Na_leak_N = g_Na_leak_N*(phi_NE - E_Na_N)  # sodium    - (A/m^2)
        I_K_leak_N = g_K_leak_N*(phi_NE - E_K_N)     # potassium - (A/m^2)
        I_Cl_leak_N = g_Cl_leak_N*(phi_NE - E_Cl_N)  # chloride  - (A/m^2)

        # Leak currents - glial
        I_Na_leak_G = g_Na_leak_G*(phi_GE - E_Na_G)  # sodium    - (A/m^2)
        I_Cl_leak_G = g_Cl_leak_G*(phi_GE - E_Cl_G)  # chloride  - (A/m^2)

        ################################################################
        # Potassium inward rectifier (KIR) current - glial
        A = 18.5/42.5                               # shorthand
        B = 1.0e3*(phi_GE - E_K_G + 18.5e-3)/42.5   # shorthand
        C = (-118.6 - 85.2)/44.1                    # shorthand
        D = 1.0e3*(-118.6e-3 + phi_GE)/44.1         # shorthand
        # inward rectifying conductance
        #K_G_init = float(self.K_G_init)
        KGinit = self.initial_conditions['KGinit']
        g_KIR = sqrt(K_G/KGinit)*(1 + exp(A))/(1 + exp(B))*\
                                 (1 + exp(C))/(1 + exp(D))
        # inward rectifying current
        I_KIR = g_KIR_0*g_KIR*(phi_GE - E_K_G)      # (A/m^2)

        ################################################################
        # Total transmembrane ion currents - neuron
        I_Na_NE = I_Na_leak_N       # sodium    - (A/m^2)
        I_K_NE = I_K_leak_N         # potassium -(A/m^2)
        I_Cl_NE = I_Cl_leak_N       # chloride  - (A/m^2)

        # total transmembrane ion currents glial
        I_Na_GE = I_Na_leak_G       # sodium    - (A/m^2)
        I_K_GE = I_KIR              # potassium - (A/m^2)
        I_Cl_GE = I_Cl_leak_G       # chloride  - (A/m^2)

        # convert currents currents to flux - neuron
        J_Na_NE = I_Na_NE/(F*z_Na)  # sodium    - (mol/(m^2s))
        J_K_NE = I_K_NE/(F*z_K)     # potassium - (mol/(m^2s))
        J_Cl_NE = I_Cl_NE/(F*z_Cl)  # chloride  - (mol/(m^2s))

        # convert currents currents to flux - glial
        J_Na_GE = I_Na_GE/(F*z_Na)  # sodium    - (mol/(m^2s))
        J_K_GE = I_K_GE/(F*z_K)     # potassium - (mol/(m^2s))
        J_Cl_GE = I_Cl_GE/(F*z_Cl)  # chloride  - (mol/(m^2s))

        J_M = [[J_Na_NE, J_Na_GE], [J_K_NE, J_K_GE], [J_Cl_NE, J_Cl_GE]]

        # set problem's membrane fluxes
        self.membrane_fluxes = J_M
        return
