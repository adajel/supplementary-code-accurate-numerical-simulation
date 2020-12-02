from dolfin import *
import ufl

from problem_base import ProblemBase

class Problem(ProblemBase):
    """ Problem where CSD wave is initiated by excitatory currents """
    def __init(self, mesh, t_PDE, t_ODE):
        ProblemBase.__init__(self, mesh, t_PDE, t_ODE)

    def excitatory_currents(self, phi_NE, E_Na_N, E_K_N, E_Cl_N):
        """ Excitatory currents for initiating wave - neuron """
        # set conductance
        Gmax = 5.0          # max conductance (S/m^2)
        LE = 2.0e-5         # stimulate zone at leftmost part of domain
        tE = 2.0            # time of stimuli (s)
        GE = Expression('Gmax*cos(pi*x[0]/(2.0*LE))*cos(pi*x[0]/(2.0*LE))*\
                         sin(pi*t/tE)*(x[0] <= LE)*(t <= tE)',
                         Gmax=Gmax, LE=LE, tE=tE, t=self.t_PDE, degree=4)

        # define and return currents
        I_Na_ex = GE*(phi_NE - E_Na_N)  # sodium    - A/m^2
        I_K_ex = GE*(phi_NE - E_K_N)    # potassium - A/m^2
        I_Cl_ex = GE*(phi_NE - E_Cl_N)  # chloride  - A/m^2
        return I_Na_ex, I_K_ex, I_Cl_ex

    def set_membrane_fluxes(self, w, w_, ss):
        """ set the problems transmembrane ion fluxes. Note that the passive
        fluxes are treated implicitly (w_), while active currents (i.e pumps)
        are treated explicitly (w), except for the gating variables (ss). """
        # get physical parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]

        # split unknowns (PDEs)
        alpha_N, Na_N, Na_E, K_N, K_E, Cl_N, Cl_E, phi_N, phi_E = split(w)

        # split solution from previous time step (PDEs)
        alpha_N_, Na_N_, Na_E_, K_N_, K_E_, \
                Cl_N_, Cl_E_, phi_N_, phi_E_ = split(w_)

        # calculate membrane potentials
        phi_NE = phi_N - phi_E     # neuron (V)

        ################################################################
        # define Nernst potential - neuron
        E_Na_N = R*temperature/(F*z_Na)*ln(Na_E/Na_N) # sodium    - (V)
        E_K_N = R*temperature/(F*z_K)*ln(K_E/K_N)     # potassium - (V)
        E_Cl_N = R*temperature/(F*z_Cl)*ln(Cl_E/Cl_N) # chloride  - (V)

        ################################################################
        # get currents
        I_NaP, I_KDR, I_KA = self.voltage_gated_currents(phi_NE, Na_N, Na_E, K_N, K_E, ss)
        I_ATP_N = self.I_ATP_N(K_E_, Na_N_)
        I_Na_leak_N, I_K_leak_N, I_Cl_leak_N = self.leak_currents_neuron(phi_NE, E_Na_N, E_K_N, E_Cl_N)
        # get excitatory currents
        I_Na_ex, I_K_ex, I_Cl_ex = self.excitatory_currents(phi_NE, E_Na_N, E_K_N, E_Cl_N)

        # Total transmembrane ion currents - neuron
        I_Na_NE = I_Na_leak_N + I_NaP + 3.0*I_ATP_N + I_Na_ex     # sodium    - (A/m^2)
        I_K_NE = I_K_leak_N + I_KDR + I_KA - 2.0*I_ATP_N + I_K_ex # potassium -(A/m^2)
        I_Cl_NE = I_Cl_leak_N + I_Cl_ex                           # chloride  - (A/m^2)

        ################################################################
        # convert currents currents to flux - neuron
        J_Na_NE = I_Na_NE/(F*z_Na)      # sodium    - (mol/(m^2s))
        J_K_NE = I_K_NE/(F*z_K)         # potassium - (mol/(m^2s))
        J_Cl_NE = I_Cl_NE/(F*z_Cl)      # chloride  - (mol/(m^2s))

        J_M = [[J_Na_NE], [J_K_NE], [J_Cl_NE]]

        # set problem's membrane fluxes
        self.membrane_fluxes = J_M
        return
