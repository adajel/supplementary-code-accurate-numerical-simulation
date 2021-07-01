from dolfin import *

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

#from solver import Solver
from solver_BDF2 import Solver
from mms_1D_ODE import ProblemMMS

def space_time():
    # create directory for saving results if it does not already exist
    directory = "results/mms/1D_ODE"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create files for results
    title_f1 = directory + "/convergence_table_alpha_L2_st.txt"
    title_f2 = directory + "/convergence_table_Na_L2_st.txt"
    title_f3 = directory + "/convergence_table_K_L2_st.txt"
    title_f4 = directory + "/convergence_table_Cl_L2_st.txt"
    title_f5 = directory + "/convergence_table_phi_L2_st.txt"

    title_f6 = directory + "/convergence_table_alpha_H1_st.txt"
    title_f7 = directory + "/convergence_table_Na_H1_st.txt"
    title_f8 = directory + "/convergence_table_K_H1_st.txt"
    title_f9 = directory + "/convergence_table_Cl_H1_st.txt"
    title_f10 = directory + "/convergence_table_phi_H1_st.txt"

    title_f11 = directory + "/convergence_table_gat_L2_st.txt"

    title_sum_L2 =  directory + "/convergence_table_summary_L2_st.txt"
    title_sum_H1 =  directory + "/convergence_table_summary_H1_st.txt"

    f1 = open(title_f1, 'w+')
    f2 = open(title_f2, 'w+')
    f3 = open(title_f3, 'w+')
    f4 = open(title_f4, 'w+')
    f5 = open(title_f5, 'w+')

    f6 = open(title_f6, 'w+')
    f7 = open(title_f7, 'w+')
    f8 = open(title_f8, 'w+')
    f9 = open(title_f9, 'w+')
    f10 = open(title_f10, 'w+')

    f11 = open(title_f11, 'w+')

    fsum_L2 = open(title_sum_L2, 'w+')
    fsum_H1 = open(title_sum_H1, 'w+')

    # baseline time step
    dt_0 = 1.0e-3
    # baseline end time
    Tstop = 2*dt_0

    # space resolutions
    resolutions = [3, 4, 5, 6, 7]
    # number of iterations
    i = 0

    for resolution in resolutions:
        # create mesh
        N = 2**resolution            # number of cells
        mesh = IntervalMesh(N, 0, 1) # mesh
        h = mesh.hmin()              # minimum diameter of cells
        boundary_point = "near(x[0], 0.0)"

        # time variables
        dt = dt_0/(2**i)      # time step

        # create problem
        t_PDE = Constant(0.0)           # time constant
        t_ODE = Constant(0.0)           # time constant
        problem = ProblemMMS(mesh, boundary_point, t_PDE, t_ODE)
        # solve system
        S = Solver(problem, dt, Tstop, MMS_test=True)
        S.solve_system_strang()

        print("-------------------------------")
        print("N", N)
        print("h", h)
        print("dt", dt)
        print("Tstop", Tstop)
        print("problem.t_PDE", float(problem.t_PDE))
        print("-------------------------------")

        # get sub functions
        alpha_N, Na_N, Na_E, K_N, K_E, Cl_N, \
                Cl_E, phi_N, phi_E  = S.w.split(deepcopy=True)

        ss_m, ss_h, ss_g = S.ss.split(deepcopy=True)

        # unwrap exact solutions
        alphaNe = problem.exact_solutions['alphaNe']
        NaNe = problem.exact_solutions['NaNe']
        KNe = problem.exact_solutions['KNe']
        ClNe = problem.exact_solutions['ClNe']
        NaEe = problem.exact_solutions['NaEe']
        KEe = problem.exact_solutions['KEe']
        ClEe = problem.exact_solutions['ClEe']
        phiNe = problem.exact_solutions['phiNe']
        phiEe = problem.exact_solutions['phiEe']
        ss_me = problem.exact_solutions['me']
        ss_he = problem.exact_solutions['he']
        ss_ge = problem.exact_solutions['ge']

        # function space for exact solutions
        FE = FiniteElement('CG', mesh.ufl_cell(), 5)    # define element
        FE_DG = FiniteElement('DG', mesh.ufl_cell(), 5) # define element
        VE = FunctionSpace(mesh, FE)                    # define function space
        VE_DG = FunctionSpace(mesh, FE_DG)              # define function space

        # get exact solutions as functions
        alpha_N_e = interpolate(alphaNe, VE_DG) # Na intracellular

        Na_N_e = interpolate(NaNe, VE)          # Na intracellular
        Na_E_e = interpolate(NaEe, VE)          # Na extracellular

        K_N_e = interpolate(KNe, VE)            # K intracellular
        K_E_e = interpolate(KEe, VE)            # K extracellular

        Cl_N_e = interpolate(ClNe, VE)          # Cl intracellular
        Cl_E_e = interpolate(ClEe, VE)          # Cl extracellular

        phi_N_e = interpolate(phiNe, VE)        # phi intracellular
        phi_E_e = interpolate(phiEe, VE)        # phi extracellular

        ss_m_e = interpolate(ss_me, VE)         # gating extracellular
        ss_h_e = interpolate(ss_he, VE)         # gating extracellular
        ss_g_e = interpolate(ss_ge, VE)         # gating extracellular

        # get error L2
        alphaN_L2 = errornorm(alpha_N_e, alpha_N, "L2", degree_rise=2)
        NaN_L2 = errornorm(Na_N_e, Na_N, "L2", degree_rise=2)
        NaE_L2 = errornorm(Na_E_e, Na_E, "L2", degree_rise=2)
        KN_L2 = errornorm(K_N_e, K_N, "L2", degree_rise=2)
        KE_L2 = errornorm(K_E_e, K_E, "L2", degree_rise=2)
        ClN_L2 = errornorm(Cl_N_e, Cl_N, "L2", degree_rise=2)
        ClE_L2 = errornorm(Cl_E_e, Cl_E, "L2", degree_rise=2)
        phiN_L2 = errornorm(phi_N_e, phi_N, "L2", degree_rise=2)
        phiE_L2 = errornorm(phi_E_e, phi_E, "L2", degree_rise=2)
        m_L2 = errornorm(ss_m_e, ss_m, "L2", degree_rise=2)
        h_L2 = errornorm(ss_h_e, ss_h, "L2", degree_rise=2)
        g_L2 = errornorm(ss_g_e, ss_g, "L2", degree_rise=2)

        # get error H1
        alphaN_H1 = errornorm(alpha_N_e, alpha_N, "H1", degree_rise=2)
        NaN_H1 = errornorm(Na_N_e, Na_N, "H1", degree_rise=2)
        NaE_H1 = errornorm(Na_E_e, Na_E, "H1", degree_rise=2)
        KN_H1 = errornorm(K_N_e, K_N, "H1", degree_rise=2)
        KE_H1 = errornorm(K_E_e, K_E, "H1", degree_rise=2)
        ClN_H1 = errornorm(Cl_N_e, Cl_N, "H1", degree_rise=2)
        ClE_H1 = errornorm(Cl_E_e, Cl_E, "H1", degree_rise=2)
        phiN_H1 = errornorm(phi_N_e, phi_N, "H1", degree_rise=2)
        phiE_H1 = errornorm(phi_E_e, phi_E, "H1", degree_rise=2)

        if i == 0:
            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E & %.2E & %.2E(-- --) \\\\' % (N, h, dt, alphaN_L2,))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N, h,dt, NaN_L2, NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N,h,dt, KN_L2, KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N,h,dt, ClN_L2, ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N,h,dt, phiN_L2, phiE_L2))

            # write to file - L2/H1 err and rate - alpha
            f6.write('%g & %.2E & %.2E &%.2E(--) \\\\' % (N,h,dt, alphaN_H1))
            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N,h,dt, NaN_H1,  NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N,h,dt, KN_H1, KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N,h,dt, ClN_H1,  ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) \\\\' % (N,h,dt, phiN_H1, phiE_H1))

            # write to file - L2/H1 err and rate - Cl
            f11.write('%g & %.2E & %.2E &%.2E(--) & %.2E(--) & %.2E(--)\\\\' % (N,h,dt, m_L2,  h_L2, g_L2))

            # write to file - summary
            fsum_L2.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) & %.2E(-----)\\\\' % (N, NaE_L2, phiN_L2, alphaN_L2, m_L2))

            # write to file - summary
            fsum_H1.write('%g & %.2E(-----) & %.2E(-----)\\\\' % (N, NaE_H1, phiN_H1))

        if i > 0:
            # L2 error rates
            r_alphaN_L2 = np.log(alphaN_L2/alphaN_L2_0)/np.log(h/h0)
            r_NaN_L2 = np.log(NaN_L2/NaN_L2_0)/np.log(h/h0)
            r_NaE_L2 = np.log(NaE_L2/NaE_L2_0)/np.log(h/h0)
            r_KN_L2 = np.log(KN_L2/KN_L2_0)/np.log(h/h0)
            r_KE_L2 = np.log(KE_L2/KE_L2_0)/np.log(h/h0)
            r_ClN_L2 = np.log(ClN_L2/ClN_L2_0)/np.log(h/h0)
            r_ClE_L2 = np.log(ClE_L2/ClE_L2_0)/np.log(h/h0)
            r_phiN_L2 = np.log(phiN_L2/phiN_L2_0)/np.log(h/h0)
            r_phiE_L2 = np.log(phiE_L2/phiE_L2_0)/np.log(h/h0)

            r_m_L2 = np.log(m_L2/m_L2_0)/np.log(h/h0)
            r_h_L2 = np.log(h_L2/h_L2_0)/np.log(h/h0)
            r_g_L2 = np.log(g_L2/g_L2_0)/np.log(h/h0)

            # H1 error rates
            r_alphaN_H1 = np.log(alphaN_H1/alphaN_H1_0)/np.log(h/h0)
            r_NaN_H1 = np.log(NaN_H1/NaN_H1_0)/np.log(h/h0)
            r_NaE_H1 = np.log(NaE_H1/NaE_H1_0)/np.log(h/h0)
            r_KN_H1 = np.log(KN_H1/KN_H1_0)/np.log(h/h0)
            r_KE_H1 = np.log(KE_H1/KE_H1_0)/np.log(h/h0)
            r_ClN_H1 = np.log(ClN_H1/ClN_H1_0)/np.log(h/h0)
            r_ClE_H1 = np.log(ClE_H1/ClE_H1_0)/np.log(h/h0)
            r_phiN_H1 = np.log(phiN_H1/phiN_H1_0)/np.log(h/h0)
            r_phiE_H1 = np.log(phiE_H1/phiE_H1_0)/np.log(h/h0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E & %.2E & %.2E(%.2f) \\\\' % (N, h, dt, alphaN_L2, r_alphaN_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N, h,dt,\
                            NaN_L2, r_NaN_L2, NaE_L2, r_NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            KN_L2, r_KN_L2, KE_L2, r_KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            ClN_L2, r_ClN_L2, ClE_L2, r_ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            phiN_L2, r_phiN_L2, phiE_L2, r_phiE_L2))

            # write to file - L2/H1 err and rate - alpha
            f6.write('%g & %.2E & %.2E &%.2E(%.2f) \\\\' % (N,h,dt,\
                            alphaN_H1, r_alphaN_H1))
            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            NaN_H1, r_NaN_H1, NaE_H1, r_NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            KN_H1, r_KN_H1, KE_H1, r_KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            ClN_H1, r_ClN_H1, ClE_H1, r_ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            phiN_H1, r_phiN_H1, phiE_H1, r_phiE_H1))

            # write to file - L2/H1 err and rate - Cl
            f11.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N,h,dt,\
                            m_L2, r_m_L2, h_L2, r_h_L2, g_L2, r_g_L2))

            # write to file - summary
            fsum_L2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_L2, r_NaE_L2, \
                            phiN_L2, r_phiN_L2, \
                            alphaN_L2, r_alphaN_L2, \
                            m_L2, r_m_L2))

            # write to file - summary
            fsum_H1.write('%g & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_H1, r_NaE_H1, \
                            phiN_H1, r_phiN_H1))

        # update prev h
        h0 = h
        # update prev L2
        alphaN_L2_0, NaN_L2_0, NaE_L2_0, \
                KN_L2_0, KE_L2_0, ClN_L2_0, ClE_L2_0,\
                phiN_L2_0,  phiE_L2_0 = alphaN_L2, \
                NaN_L2, NaE_L2, KN_L2, \
                KE_L2, ClN_L2, ClE_L2, phiN_L2, phiE_L2, \

        m_L2_0, h_L2_0, g_L2_0 = m_L2, h_L2, g_L2

        # update prev H1
        alphaN_H1_0, NaN_H1_0,  NaE_H1_0, \
                KN_H1_0, KE_H1_0, ClN_H1_0, ClE_H1_0,\
                phiN_H1_0, phiE_H1_0 = alphaN_H1, \
                NaN_H1, NaE_H1, KN_H1, \
                KE_H1, ClN_H1, ClE_H1, phiN_H1, phiE_H1

        # update iteration number
        i += 1

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()

    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()

    f11.close()

    fsum_L2.close()
    fsum_H1.close()

    return

def space():
    # create directory for saving results if it does not already exist
    directory = "results/mms/1D_ODE"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create files for results
    title_f1 = directory + "/convergence_table_alpha_L2_s.txt"
    title_f2 = directory + "/convergence_table_Na_L2_s.txt"
    title_f3 = directory + "/convergence_table_K_L2_s.txt"
    title_f4 = directory + "/convergence_table_Cl_L2_s.txt"
    title_f5 = directory + "/convergence_table_phi_L2_s.txt"

    title_f6 = directory + "/convergence_table_alpha_H1_s.txt"
    title_f7 = directory + "/convergence_table_Na_H1_s.txt"
    title_f8 = directory + "/convergence_table_K_H1_s.txt"
    title_f9 = directory + "/convergence_table_Cl_H1_s.txt"
    title_f10 = directory + "/convergence_table_phi_H1_s.txt"

    title_f11 = directory + "/convergence_table_gat_L2_s.txt"

    title_sum_L2 =  directory + "/convergence_table_summary_L2_s.txt"
    title_sum_H1 =  directory + "/convergence_table_summary_H1_s.txt"

    f1 = open(title_f1, 'w+')
    f2 = open(title_f2, 'w+')
    f3 = open(title_f3, 'w+')
    f4 = open(title_f4, 'w+')
    f5 = open(title_f5, 'w+')

    f6 = open(title_f6, 'w+')
    f7 = open(title_f7, 'w+')
    f8 = open(title_f8, 'w+')
    f9 = open(title_f9, 'w+')
    f10 = open(title_f10, 'w+')

    f11 = open(title_f11, 'w+')

    fsum_L2 = open(title_sum_L2, 'w+')
    fsum_H1 = open(title_sum_H1, 'w+')

    # baseline time step
    dt = 1.0e-3
    Tstop = 2*dt

    # space resolutions
    resolutions = [3, 4, 5, 6, 7]
    # number of iterations
    i = 0

    for resolution in resolutions:
        # create mesh
        N = 2**resolution            # number of cells
        mesh = IntervalMesh(N, 0, 1) # mesh
        h = mesh.hmin()              # minimum diameter of cells
        boundary_point = "near(x[0], 0.0)"

        # create problem
        t_PDE = Constant(0.0)           # time constant
        t_ODE = Constant(0.0)           # time constant
        problem = ProblemMMS(mesh, boundary_point, t_PDE, t_ODE)

        # solve system
        S = Solver(problem, dt, Tstop, MMS_test=True)
        S.solve_system_strang()

        print("-------------------------------")
        print("N", N)
        print("h", h)
        print("dt", dt)
        print("Tstop", Tstop)
        print("problem.t_PDE", float(problem.t_PDE))
        print("-------------------------------")

        # get sub functions
        alpha_N, Na_N, Na_E, K_N, K_E, Cl_N, \
                Cl_E, phi_N, phi_E  = S.w.split(deepcopy=True)

        ss_m, ss_h, ss_g = S.ss.split(deepcopy=True)

        # unwrap exact solutions
        alphaNe = problem.exact_solutions['alphaNe']
        NaNe = problem.exact_solutions['NaNe']
        KNe = problem.exact_solutions['KNe']
        ClNe = problem.exact_solutions['ClNe']
        NaEe = problem.exact_solutions['NaEe']
        KEe = problem.exact_solutions['KEe']
        ClEe = problem.exact_solutions['ClEe']
        phiNe = problem.exact_solutions['phiNe']
        phiEe = problem.exact_solutions['phiEe']
        ss_me = problem.exact_solutions['me']
        ss_he = problem.exact_solutions['he']
        ss_ge = problem.exact_solutions['ge']

        # function space for exact solutions
        FE = FiniteElement('CG', mesh.ufl_cell(), 5)    # define element
        FE_DG = FiniteElement('DG', mesh.ufl_cell(), 5) # define element
        VE = FunctionSpace(mesh, FE)                    # define function space
        VE_DG = FunctionSpace(mesh, FE_DG)              # define function space

        # get exact solutions as functions
        alpha_N_e = interpolate(alphaNe, VE_DG) # Na intracellular

        Na_N_e = interpolate(NaNe, VE)          # Na intracellular
        Na_E_e = interpolate(NaEe, VE)          # Na extracellular

        K_N_e = interpolate(KNe, VE)            # K intracellular
        K_E_e = interpolate(KEe, VE)            # K extracellular

        Cl_N_e = interpolate(ClNe, VE)          # Cl intracellular
        Cl_E_e = interpolate(ClEe, VE)          # Cl extracellular

        phi_N_e = interpolate(phiNe, VE)        # phi intracellular
        phi_E_e = interpolate(phiEe, VE)        # phi extracellular

        ss_m_e = interpolate(ss_me, VE)         # gating extracellular
        ss_h_e = interpolate(ss_he, VE)         # gating extracellular
        ss_g_e = interpolate(ss_ge, VE)         # gating extracellular

        # get error L2
        alphaN_L2 = errornorm(alpha_N_e, alpha_N, "L2", degree_rise=2)
        NaN_L2 = errornorm(Na_N_e, Na_N, "L2", degree_rise=2)
        NaE_L2 = errornorm(Na_E_e, Na_E, "L2", degree_rise=2)
        KN_L2 = errornorm(K_N_e, K_N, "L2", degree_rise=2)
        KE_L2 = errornorm(K_E_e, K_E, "L2", degree_rise=2)
        ClN_L2 = errornorm(Cl_N_e, Cl_N, "L2", degree_rise=2)
        ClE_L2 = errornorm(Cl_E_e, Cl_E, "L2", degree_rise=2)
        phiN_L2 = errornorm(phi_N_e, phi_N, "L2", degree_rise=2)
        phiE_L2 = errornorm(phi_E_e, phi_E, "L2", degree_rise=2)
        m_L2 = errornorm(ss_m_e, ss_m, "L2", degree_rise=2)
        h_L2 = errornorm(ss_h_e, ss_h, "L2", degree_rise=2)
        g_L2 = errornorm(ss_g_e, ss_g, "L2", degree_rise=2)

        # get error H1
        alphaN_H1 = errornorm(alpha_N_e, alpha_N, "H1", degree_rise=2)
        NaN_H1 = errornorm(Na_N_e, Na_N, "H1", degree_rise=2)
        NaE_H1 = errornorm(Na_E_e, Na_E, "H1", degree_rise=2)
        KN_H1 = errornorm(K_N_e, K_N, "H1", degree_rise=2)
        KE_H1 = errornorm(K_E_e, K_E, "H1", degree_rise=2)
        ClN_H1 = errornorm(Cl_N_e, Cl_N, "H1", degree_rise=2)
        ClE_H1 = errornorm(Cl_E_e, Cl_E, "H1", degree_rise=2)
        phiN_H1 = errornorm(phi_N_e, phi_N, "H1", degree_rise=2)
        phiE_H1 = errornorm(phi_E_e, phi_E, "H1", degree_rise=2)

        if i > 0:
            # L2 error rates
            r_alphaN_L2 = np.log(alphaN_L2/alphaN_L2_0)/np.log(h/h0)
            r_NaN_L2 = np.log(NaN_L2/NaN_L2_0)/np.log(h/h0)
            r_NaE_L2 = np.log(NaE_L2/NaE_L2_0)/np.log(h/h0)
            r_KN_L2 = np.log(KN_L2/KN_L2_0)/np.log(h/h0)
            r_KE_L2 = np.log(KE_L2/KE_L2_0)/np.log(h/h0)
            r_ClN_L2 = np.log(ClN_L2/ClN_L2_0)/np.log(h/h0)
            r_ClE_L2 = np.log(ClE_L2/ClE_L2_0)/np.log(h/h0)
            r_phiN_L2 = np.log(phiN_L2/phiN_L2_0)/np.log(h/h0)
            r_phiE_L2 = np.log(phiE_L2/phiE_L2_0)/np.log(h/h0)

            r_m_L2 = np.log(m_L2/m_L2_0)/np.log(h/h0)
            r_h_L2 = np.log(h_L2/h_L2_0)/np.log(h/h0)
            r_g_L2 = np.log(g_L2/g_L2_0)/np.log(h/h0)

            # H1 error rates
            r_alphaN_H1 = np.log(alphaN_H1/alphaN_H1_0)/np.log(h/h0)
            r_NaN_H1 = np.log(NaN_H1/NaN_H1_0)/np.log(h/h0)
            r_NaE_H1 = np.log(NaE_H1/NaE_H1_0)/np.log(h/h0)
            r_KN_H1 = np.log(KN_H1/KN_H1_0)/np.log(h/h0)
            r_KE_H1 = np.log(KE_H1/KE_H1_0)/np.log(h/h0)
            r_ClN_H1 = np.log(ClN_H1/ClN_H1_0)/np.log(h/h0)
            r_ClE_H1 = np.log(ClE_H1/ClE_H1_0)/np.log(h/h0)
            r_phiN_H1 = np.log(phiN_H1/phiN_H1_0)/np.log(h/h0)
            r_phiE_H1 = np.log(phiE_H1/phiE_H1_0)/np.log(h/h0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E & %.2E & %.2E(%.2f) \\\\' % (N, h, dt, alphaN_L2, r_alphaN_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N, h,dt,\
                            NaN_L2, r_NaN_L2, NaE_L2, r_NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            KN_L2, r_KN_L2, KE_L2, r_KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            ClN_L2, r_ClN_L2, ClE_L2, r_ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            phiN_L2, r_phiN_L2, phiE_L2, r_phiE_L2))

            # write to file - L2/H1 err and rate - alpha
            f6.write('%g & %.2E & %.2E &%.2E(%.2f) \\\\' % (N,h,dt,\
                            alphaN_H1, r_alphaN_H1))
            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            NaN_H1, r_NaN_H1, NaE_H1, r_NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            KN_H1, r_KN_H1, KE_H1, r_KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            ClN_H1, r_ClN_H1, ClE_H1, r_ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            phiN_H1, r_phiN_H1, phiE_H1, r_phiE_H1))

            # write to file - L2/H1 err and rate - Cl
            f11.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N,h,dt,\
                            m_L2, r_m_L2, h_L2, r_h_L2, g_L2, r_g_L2))

            # write to file - summary
            fsum_L2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_L2, r_NaE_L2, \
                            phiN_L2, r_phiN_L2, \
                            alphaN_L2, r_alphaN_L2, \
                            m_L2, r_m_L2))

            # write to file - summary
            fsum_H1.write('%g & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_H1, r_NaE_H1, \
                            phiN_H1, r_phiN_H1))

        # update prev h
        h0 = h
        # update prev L2
        alphaN_L2_0, NaN_L2_0, NaE_L2_0, \
                KN_L2_0, KE_L2_0, ClN_L2_0, ClE_L2_0,\
                phiN_L2_0,  phiE_L2_0 = alphaN_L2, \
                NaN_L2, NaE_L2, KN_L2, \
                KE_L2, ClN_L2, ClE_L2, phiN_L2, phiE_L2, \

        m_L2_0, h_L2_0, g_L2_0 = m_L2, h_L2, g_L2

        # update prev H1
        alphaN_H1_0, NaN_H1_0,  NaE_H1_0, \
                KN_H1_0, KE_H1_0, ClN_H1_0, ClE_H1_0,\
                phiN_H1_0, phiE_H1_0 = alphaN_H1, \
                NaN_H1, NaE_H1, KN_H1, \
                KE_H1, ClN_H1, ClE_H1, phiN_H1, phiE_H1

        # update iteration number
        i += 1

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()

    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()

    f11.close()

    fsum_L2.close()
    fsum_H1.close()

    return

def time():
    # create directory for saving results if it does not already exist
    directory = "results/mms/1D_ODE"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create files for results
    title_f1 = directory + "/convergence_table_alpha_L2_t.txt"
    title_f2 = directory + "/convergence_table_Na_L2_t.txt"
    title_f3 = directory + "/convergence_table_K_L2_t.txt"
    title_f4 = directory + "/convergence_table_Cl_L2_t.txt"
    title_f5 = directory + "/convergence_table_phi_L2_t.txt"

    title_f6 = directory + "/convergence_table_alpha_H1_t.txt"
    title_f7 = directory + "/convergence_table_Na_H1_t.txt"
    title_f8 = directory + "/convergence_table_K_H1_t.txt"
    title_f9 = directory + "/convergence_table_Cl_H1_t.txt"
    title_f10 = directory + "/convergence_table_phi_H1_t.txt"

    title_f11 = directory + "/convergence_table_gat_L2_t.txt"

    title_sum_L2 =  directory + "/convergence_table_summary_L2_t.txt"
    title_sum_H1 =  directory + "/convergence_table_summary_H1_t.txt"

    f1 = open(title_f1, 'w+')
    f2 = open(title_f2, 'w+')
    f3 = open(title_f3, 'w+')
    f4 = open(title_f4, 'w+')
    f5 = open(title_f5, 'w+')

    f6 = open(title_f6, 'w+')
    f7 = open(title_f7, 'w+')
    f8 = open(title_f8, 'w+')
    f9 = open(title_f9, 'w+')
    f10 = open(title_f10, 'w+')

    f11 = open(title_f11, 'w+')

    fsum_L2 = open(title_sum_L2, 'w+')
    fsum_H1 = open(title_sum_H1, 'w+')

    # create mesh
    N = 2**10                    # number of cells
    mesh = IntervalMesh(N, 0, 1) # mesh
    h = mesh.hmin()              # minimum diameter of cells
    boundary_point = "near(x[0], 0.0)"

    # baseline time step
    dts = [8.0e-2, 4.0e-2, 2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3]

    Tstop = 2*dts[0]

    # number of iterations
    i = 0

    for dt in dts:

        # create problem
        t_PDE = Constant(0.0) # time constant
        t_ODE = Constant(0.0) # time constant
        problem = ProblemMMS(mesh, boundary_point, t_PDE, t_ODE)

        # solve system
        S = Solver(problem, dt, Tstop, MMS_test=True)
        S.solve_system_strang()

        print("-------------------------------")
        print("N", N)
        print("h", h)
        print("dt", dt)
        print("Tstop", Tstop)
        print("problem.t_PDE", float(problem.t_PDE))
        print("-------------------------------")

        # get sub functions
        alpha_N, Na_N, Na_E, K_N, K_E, Cl_N, \
                Cl_E, phi_N, phi_E  = S.w.split(deepcopy=True)

        ss_m, ss_h, ss_g = S.ss.split(deepcopy=True)

        # unwrap exact solutions
        alphaNe = problem.exact_solutions['alphaNe']
        NaNe = problem.exact_solutions['NaNe']
        KNe = problem.exact_solutions['KNe']
        ClNe = problem.exact_solutions['ClNe']
        NaEe = problem.exact_solutions['NaEe']
        KEe = problem.exact_solutions['KEe']
        ClEe = problem.exact_solutions['ClEe']
        phiNe = problem.exact_solutions['phiNe']
        phiEe = problem.exact_solutions['phiEe']
        ss_me = problem.exact_solutions['me']
        ss_he = problem.exact_solutions['he']
        ss_ge = problem.exact_solutions['ge']

        # function space for exact solutions
        FE = FiniteElement('CG', mesh.ufl_cell(), 5)    # define element
        FE_DG = FiniteElement('DG', mesh.ufl_cell(), 5) # define element
        VE = FunctionSpace(mesh, FE)                    # define function space
        VE_DG = FunctionSpace(mesh, FE_DG)              # define function space

        # get exact solutions as functions
        alpha_N_e = interpolate(alphaNe, VE_DG) # Na intracellular

        Na_N_e = interpolate(NaNe, VE)          # Na intracellular
        Na_E_e = interpolate(NaEe, VE)          # Na extracellular

        K_N_e = interpolate(KNe, VE)            # K intracellular
        K_E_e = interpolate(KEe, VE)            # K extracellular

        Cl_N_e = interpolate(ClNe, VE)          # Cl intracellular
        Cl_E_e = interpolate(ClEe, VE)          # Cl extracellular

        phi_N_e = interpolate(phiNe, VE)        # phi intracellular
        phi_E_e = interpolate(phiEe, VE)        # phi extracellular

        ss_m_e = interpolate(ss_me, VE)         # gating extracellular
        ss_h_e = interpolate(ss_he, VE)         # gating extracellular
        ss_g_e = interpolate(ss_ge, VE)         # gating extracellular

        # get error L2
        alphaN_L2 = errornorm(alpha_N_e, alpha_N, "L2", degree_rise=2)
        NaN_L2 = errornorm(Na_N_e, Na_N, "L2", degree_rise=2)
        NaE_L2 = errornorm(Na_E_e, Na_E, "L2", degree_rise=2)
        KN_L2 = errornorm(K_N_e, K_N, "L2", degree_rise=2)
        KE_L2 = errornorm(K_E_e, K_E, "L2", degree_rise=2)
        ClN_L2 = errornorm(Cl_N_e, Cl_N, "L2", degree_rise=2)
        ClE_L2 = errornorm(Cl_E_e, Cl_E, "L2", degree_rise=2)
        phiN_L2 = errornorm(phi_N_e, phi_N, "L2", degree_rise=2)
        phiE_L2 = errornorm(phi_E_e, phi_E, "L2", degree_rise=2)
        m_L2 = errornorm(ss_m_e, ss_m, "L2", degree_rise=2)
        h_L2 = errornorm(ss_h_e, ss_h, "L2", degree_rise=2)
        g_L2 = errornorm(ss_g_e, ss_g, "L2", degree_rise=2)

        # get error H1
        alphaN_H1 = errornorm(alpha_N_e, alpha_N, "H1", degree_rise=2)
        NaN_H1 = errornorm(Na_N_e, Na_N, "H1", degree_rise=2)
        NaE_H1 = errornorm(Na_E_e, Na_E, "H1", degree_rise=2)
        KN_H1 = errornorm(K_N_e, K_N, "H1", degree_rise=2)
        KE_H1 = errornorm(K_E_e, K_E, "H1", degree_rise=2)
        ClN_H1 = errornorm(Cl_N_e, Cl_N, "H1", degree_rise=2)
        ClE_H1 = errornorm(Cl_E_e, Cl_E, "H1", degree_rise=2)
        phiN_H1 = errornorm(phi_N_e, phi_N, "H1", degree_rise=2)
        phiE_H1 = errornorm(phi_E_e, phi_E, "H1", degree_rise=2)

        if i > 0:
            # L2 error rates
            r_alphaN_L2 = np.log(alphaN_L2/alphaN_L2_0)/np.log(dt/dt0)
            r_NaN_L2 = np.log(NaN_L2/NaN_L2_0)/np.log(dt/dt0)
            r_NaE_L2 = np.log(NaE_L2/NaE_L2_0)/np.log(dt/dt0)
            r_KN_L2 = np.log(KN_L2/KN_L2_0)/np.log(dt/dt0)
            r_KE_L2 = np.log(KE_L2/KE_L2_0)/np.log(dt/dt0)
            r_ClN_L2 = np.log(ClN_L2/ClN_L2_0)/np.log(dt/dt0)
            r_ClE_L2 = np.log(ClE_L2/ClE_L2_0)/np.log(dt/dt0)
            r_phiN_L2 = np.log(phiN_L2/phiN_L2_0)/np.log(dt/dt0)
            r_phiE_L2 = np.log(phiE_L2/phiE_L2_0)/np.log(dt/dt0)

            r_m_L2 = np.log(m_L2/m_L2_0)/np.log(dt/dt0)
            r_h_L2 = np.log(h_L2/h_L2_0)/np.log(dt/dt0)
            r_g_L2 = np.log(g_L2/g_L2_0)/np.log(dt/dt0)

            # H1 error rates
            r_alphaN_H1 = np.log(alphaN_H1/alphaN_H1_0)/np.log(dt/dt0)
            r_NaN_H1 = np.log(NaN_H1/NaN_H1_0)/np.log(dt/dt0)
            r_NaE_H1 = np.log(NaE_H1/NaE_H1_0)/np.log(dt/dt0)
            r_KN_H1 = np.log(KN_H1/KN_H1_0)/np.log(dt/dt0)
            r_KE_H1 = np.log(KE_H1/KE_H1_0)/np.log(dt/dt0)
            r_ClN_H1 = np.log(ClN_H1/ClN_H1_0)/np.log(dt/dt0)
            r_ClE_H1 = np.log(ClE_H1/ClE_H1_0)/np.log(dt/dt0)
            r_phiN_H1 = np.log(phiN_H1/phiN_H1_0)/np.log(dt/dt0)
            r_phiE_H1 = np.log(phiE_H1/phiE_H1_0)/np.log(dt/dt0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E & %.2E & %.2E(%.2f) \\\\' % (N, h, dt, alphaN_L2, r_alphaN_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N, h,dt,\
                            NaN_L2, r_NaN_L2, NaE_L2, r_NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            KN_L2, r_KN_L2, KE_L2, r_KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            ClN_L2, r_ClN_L2, ClE_L2, r_ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            phiN_L2, r_phiN_L2, phiE_L2, r_phiE_L2))

            # write to file - L2/H1 err and rate - alpha
            f6.write('%g & %.2E & %.2E &%.2E(%.2f) \\\\' % (N,h,dt,\
                            alphaN_H1, r_alphaN_H1))
            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            NaN_H1, r_NaN_H1, NaE_H1, r_NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            KN_H1, r_KN_H1, KE_H1, r_KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            ClN_H1, r_ClN_H1, ClE_H1, r_ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) \\\\' % (N,h,dt,\
                            phiN_H1, r_phiN_H1, phiE_H1, r_phiE_H1))

            # write to file - L2/H1 err and rate - Cl
            f11.write('%g & %.2E & %.2E &%.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N,h,dt,\
                            m_L2, r_m_L2, h_L2, r_h_L2, g_L2, r_g_L2))

            # write to file - summary
            fsum_L2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_L2, r_NaE_L2, \
                            phiN_L2, r_phiN_L2, \
                            alphaN_L2, r_alphaN_L2, \
                            m_L2, r_m_L2))

            # write to file - summary
            fsum_H1.write('%g & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_H1, r_NaE_H1, \
                            phiN_H1, r_phiN_H1))

        # update prev h
        dt0 = dt
        # update prev L2
        alphaN_L2_0, NaN_L2_0, NaE_L2_0, \
                KN_L2_0, KE_L2_0, ClN_L2_0, ClE_L2_0,\
                phiN_L2_0,  phiE_L2_0 = alphaN_L2, \
                NaN_L2, NaE_L2, KN_L2, \
                KE_L2, ClN_L2, ClE_L2, phiN_L2, phiE_L2, \

        m_L2_0, h_L2_0, g_L2_0 = m_L2, h_L2, g_L2

        # update prev H1
        alphaN_H1_0, NaN_H1_0,  NaE_H1_0, \
                KN_H1_0, KE_H1_0, ClN_H1_0, ClE_H1_0,\
                phiN_H1_0, phiE_H1_0 = alphaN_H1, \
                NaN_H1, NaE_H1, KN_H1, \
                KE_H1, ClN_H1, ClE_H1, phiN_H1, phiE_H1

        # update iteration number
        i += 1

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()

    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()

    f11.close()

    fsum_L2.close()
    fsum_H1.close()

    return

if __name__ == '__main__':
    # run refinement test
    space_time()
    space()
    time()
