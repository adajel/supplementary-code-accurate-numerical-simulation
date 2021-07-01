from dolfin import *

import os
import sys
import numpy as np

# set path to solver
from solver_full_BE import Solver
from mms_1D import ProblemMMS

def space_time():
    # create directory for saving results if it does not already exist
    directory = "results/mms/1D"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create files for results
    title_f1 = directory + "/convergence_table_alpha_L2_st.txt"
    title_f2 = directory + "/convergence_table_Na_L2_st.txt"
    title_f3 = directory + "/convergence_table_K_L2_st.txt"
    title_f4 = directory + "/convergence_table_Cl_L2_st.txt"
    title_f5 = directory + "/convergence_table_phi_L2_st.txt"
    title_f6 = directory + "/convergence_table_p_L2_st.txt"

    title_f7 = directory + "/convergence_table_Na_H1_st.txt"
    title_f8 = directory + "/convergence_table_K_H1_st.txt"
    title_f9 = directory + "/convergence_table_Cl_H1_st.txt"
    title_f10 = directory + "/convergence_table_phi_H1_st.txt"
    title_f11 = directory + "/convergence_table_p_H1_st.txt"

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
    resolutions = [2, 3, 4, 5, 6, 7]
    # number of iterations
    i = 0

    for resolution in resolutions:
        # create mesh
        N = 2**resolution               # number of cells
        mesh = IntervalMesh(N, 0, 1)    # mesh
        h = mesh.hmin()                 # minimum diameter of cells
        boundary_point = "near(x[0], 0.0)"

        # time variables
        dt_value = dt_0/(4**i)          # time step
        t_PDE = Constant(0.0)           # time constant
        t_ODE = Constant(0.0)           # time constant

        problem = ProblemMMS(mesh, boundary_point, t_PDE, t_ODE)
        # solve system
        S = Solver(problem, dt_value, Tstop, MMS_test=True)
        w = S.solve_system_godenov()

        print("-------------------------------")
        print("N", N)
        print("dt", dt_value)
        print("Tstop", Tstop)
        print("problem.t", float(problem.t_PDE))
        print("-------------------------------")

        # get sub functions
        alpha_N, alpha_G, Na_N, Na_G, Na_E, K_N, K_G, K_E, Cl_N, \
                Cl_G, Cl_E, phi_N, phi_G, phi_E, p_E  = S.w.split(deepcopy=True)

        # extract exact solutions
        exact_solutions = problem.exact_solutions
        for key in exact_solutions:
            exec('%s = exact_solutions["%s"]' % (key, key))

        # function space for exact solutions
        CG5 = FiniteElement('CG', mesh.ufl_cell(), 5) # define element
        V_CG = FunctionSpace(mesh, CG5)               # define function space

        # get exact solutions as functions
        alpha_N_e = interpolate(alphaNe, V_CG)   # Na intracellular
        alpha_G_e = interpolate(alphaGe, V_CG)   # Na intracellular

        Na_N_e = interpolate(NaNe, V_CG)         # Na intracellular
        Na_G_e = interpolate(NaGe, V_CG)         # Na intracellular
        Na_E_e = interpolate(NaEe, V_CG)         # Na extracellular

        K_N_e = interpolate(KNe, V_CG)           # K intracellular
        K_G_e = interpolate(KGe, V_CG)           # K intracellular
        K_E_e = interpolate(KEe, V_CG)           # K extracellular

        Cl_N_e = interpolate(ClNe, V_CG)         # Cl intracellular
        Cl_G_e = interpolate(ClGe, V_CG)         # Cl intracellular
        Cl_E_e = interpolate(ClEe, V_CG)         # Cl extracellular

        phi_N_e = interpolate(phiNe, V_CG)       # phi intracellular
        phi_G_e = interpolate(phiGe, V_CG)       # phi intracellular
        phi_E_e = interpolate(phiEe, V_CG)       # phi extracellular

        p_E_e = interpolate(pEe, V_CG)       # phi extracellular

        # get error L2
        alphaN_L2 = errornorm(alpha_N_e, alpha_N, "L2", degree_rise=4)
        alphaG_L2 = errornorm(alpha_G_e, alpha_G, "L2", degree_rise=4)
        NaN_L2 = errornorm(Na_N_e, Na_N, "L2", degree_rise=4)
        NaG_L2 = errornorm(Na_G_e, Na_G, "L2", degree_rise=4)
        NaE_L2 = errornorm(Na_E_e, Na_E, "L2", degree_rise=4)
        KN_L2 = errornorm(K_N_e, K_N, "L2", degree_rise=4)
        KG_L2 = errornorm(K_G_e, K_G, "L2", degree_rise=4)
        KE_L2 = errornorm(K_E_e, K_E, "L2", degree_rise=4)
        ClN_L2 = errornorm(Cl_N_e, Cl_N, "L2", degree_rise=4)
        ClG_L2 = errornorm(Cl_G_e, Cl_G, "L2", degree_rise=4)
        ClE_L2 = errornorm(Cl_E_e, Cl_E, "L2", degree_rise=4)
        phiN_L2 = errornorm(phi_N_e, phi_N, "L2", degree_rise=4)
        phiG_L2 = errornorm(phi_G_e, phi_G, "L2", degree_rise=4)
        phiE_L2 = errornorm(phi_E_e, phi_E, "L2", degree_rise=4)
        pE_L2 = errornorm(p_E_e, p_E, "L2", degree_rise=4)

        # get error H1
        alphaN_H1 = errornorm(alpha_N_e, alpha_N, "H1", degree_rise=4)
        alphaG_H1 = errornorm(alpha_G_e, alpha_G, "H1", degree_rise=4)
        NaN_H1 = errornorm(Na_N_e, Na_N, "H1", degree_rise=4)
        NaG_H1 = errornorm(Na_G_e, Na_G, "H1", degree_rise=4)
        NaE_H1 = errornorm(Na_E_e, Na_E, "H1", degree_rise=4)
        KN_H1 = errornorm(K_N_e, K_N, "H1", degree_rise=4)
        KG_H1 = errornorm(K_G_e, K_G, "H1", degree_rise=4)
        KE_H1 = errornorm(K_E_e, K_E, "H1", degree_rise=4)
        ClN_H1 = errornorm(Cl_N_e, Cl_N, "H1", degree_rise=4)
        ClG_H1 = errornorm(Cl_G_e, Cl_G, "H1", degree_rise=4)
        ClE_H1 = errornorm(Cl_E_e, Cl_E, "H1", degree_rise=4)
        phiN_H1 = errornorm(phi_N_e, phi_N, "H1", degree_rise=4)
        phiG_H1 = errornorm(phi_G_e, phi_G, "H1", degree_rise=4)
        phiE_H1 = errornorm(phi_E_e, phi_E, "H1", degree_rise=4)
        pE_H1 = errornorm(p_E_e, p_E, "H1", degree_rise=4)

        if i == 0:
            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E(-----) & %.2E(-----) \\\\' % (N, alphaN_L2, alphaG_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) \\\\' % (N, NaN_L2, NaG_L2, NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) \\\\' % (N,\
                            KN_L2, KG_L2, KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) \\\\' % (N, ClN_L2, ClG_L2, ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----)  \\\\' % (N, phiN_L2, phiG_L2, phiE_L2))
            # write to file - L2/H1 err and rate - pressure
            f6.write('%g & %.2E(-----) \\\\' % (N, pE_L2))

            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) \\\\' % (N, NaN_H1, NaG_H1, NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) \\\\' % (N,\
                            KN_H1,  KG_H1, KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) \\\\' % (N,\
                                ClN_H1, ClG_H1, ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----)  \\\\' % (N,\
                            phiN_H1, phiG_H1, phiE_H1))
            # write to file - L2/H1 err and rate - pressure
            f11.write('%g & %.2E(-----) \\\\' % (N, pE_H1))

                        # write to file - summary
            fsum_L2.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) & %.2E(-----)\\\\' % (N, \
                            NaE_L2, phiN_L2, alphaN_L2, pE_L2))

            fsum_H1.write('%g & %.2E(-----) & %.2E(-----) & %.2E(-----) & %.2E(-----)\\\\' % (N, \
                            NaE_H1, phiN_H1, alphaN_H1, pE_H1))

        if i > 0:
            # L2 errors
            r_alphaN_L2 = np.log(alphaN_L2/alphaN_L2_0)/np.log(h/h0)
            r_alphaG_L2 = np.log(alphaG_L2/alphaG_L2_0)/np.log(h/h0)
            r_alphaN_H1 = np.log(alphaN_H1/alphaN_H1_0)/np.log(h/h0)
            r_alphaG_H1 = np.log(alphaG_H1/alphaG_H1_0)/np.log(h/h0)
            r_NaN_L2 = np.log(NaN_L2/NaN_L2_0)/np.log(h/h0)
            r_NaG_L2 = np.log(NaG_L2/NaG_L2_0)/np.log(h/h0)
            r_NaE_L2 = np.log(NaE_L2/NaE_L2_0)/np.log(h/h0)
            r_KN_L2 = np.log(KN_L2/KN_L2_0)/np.log(h/h0)
            r_KG_L2 = np.log(KG_L2/KG_L2_0)/np.log(h/h0)
            r_KE_L2 = np.log(KE_L2/KE_L2_0)/np.log(h/h0)
            r_ClN_L2 = np.log(ClN_L2/ClN_L2_0)/np.log(h/h0)
            r_ClG_L2 = np.log(ClG_L2/ClG_L2_0)/np.log(h/h0)
            r_ClE_L2 = np.log(ClE_L2/ClE_L2_0)/np.log(h/h0)
            r_phiN_L2 = np.log(phiN_L2/phiN_L2_0)/np.log(h/h0)
            r_phiG_L2 = np.log(phiG_L2/phiG_L2_0)/np.log(h/h0)
            r_phiE_L2 = np.log(phiE_L2/phiE_L2_0)/np.log(h/h0)
            r_pE_L2 = np.log(pE_L2/pE_L2_0)/np.log(h/h0)

            r_NaN_H1 = np.log(NaN_H1/NaN_H1_0)/np.log(h/h0)
            r_NaG_H1 = np.log(NaG_H1/NaG_H1_0)/np.log(h/h0)
            r_NaE_H1 = np.log(NaE_H1/NaE_H1_0)/np.log(h/h0)
            r_KN_H1 = np.log(KN_H1/KN_H1_0)/np.log(h/h0)
            r_KG_H1 = np.log(KG_H1/KG_H1_0)/np.log(h/h0)
            r_KE_H1 = np.log(KE_H1/KE_H1_0)/np.log(h/h0)
            r_ClN_H1 = np.log(ClN_H1/ClN_H1_0)/np.log(h/h0)
            r_ClG_H1 = np.log(ClG_H1/ClG_H1_0)/np.log(h/h0)
            r_ClE_H1 = np.log(ClE_H1/ClE_H1_0)/np.log(h/h0)
            r_phiN_H1 = np.log(phiN_H1/phiN_H1_0)/np.log(h/h0)
            r_phiG_H1 = np.log(phiG_H1/phiG_H1_0)/np.log(h/h0)
            r_phiE_H1 = np.log(phiE_H1/phiE_H1_0)/np.log(h/h0)
            r_pE_H1 = np.log(pE_H1/pE_H1_0)/np.log(h/h0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            alphaN_L2, r_alphaN_L2, alphaG_L2, r_alphaG_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_L2, r_NaN_L2, NaG_L2, r_NaG_L2,\
                            NaE_L2, r_NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_L2, r_KN_L2, KG_L2, r_KG_L2,\
                            KE_L2, r_KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_L2, r_ClN_L2, ClG_L2, r_ClG_L2,\
                            ClE_L2, r_ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)  \\\\' % (N,\
                            phiN_L2, r_phiN_L2, phiG_L2, r_phiG_L2,\
                            phiE_L2, r_phiE_L2))
            # write to file - L2/H1 err and rate - pressure
            f6.write('%g & %.2E(%.2f) \\\\' % (N, pE_L2, r_pE_L2))

            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_H1, r_NaN_H1, NaG_H1, r_NaG_H1,\
                            NaE_H1, r_NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_H1, r_KN_H1, KG_H1, r_KG_H1,\
                            KE_H1, r_KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_H1, r_ClN_H1, ClG_H1, r_ClG_H1,\
                            ClE_H1, r_ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)  \\\\' % (N,\
                            phiN_H1, r_phiN_H1, phiG_H1, r_phiG_H1,\
                            phiE_H1, r_phiE_H1))
            # write to file - L2/H1 err and rate - pressure
            f11.write('%g & %.2E(%.2f) \\\\' % (N, pE_H1, r_pE_H1))

            # write to file - summary
            fsum_L2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_L2, r_NaE_L2, \
                            phiN_L2, r_phiN_L2, \
                            alphaN_L2, r_alphaN_L2, \
                            pE_L2, r_pE_L2))

            # write to file - summary
            fsum_H1.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_H1, r_NaE_H1, \
                            phiN_H1, r_phiN_H1, \
                            alphaN_H1, r_alphaN_H1, \
                            pE_H1, r_pE_H1))

        # update prev h
        h0 = h
        # update prev L2
        alphaN_L2_0, alphaG_L2_0, NaN_L2_0, NaG_L2_0, NaE_L2_0, \
                KN_L2_0, KG_L2_0, KE_L2_0, ClN_L2_0, ClG_L2_0, ClE_L2_0,\
                phiN_L2_0, phiG_L2_0, phiE_L2_0, pE_L2_0 = alphaN_L2, \
                alphaG_L2, NaN_L2, NaG_L2, NaE_L2, KN_L2, KG_L2, \
                KE_L2, ClN_L2, ClG_L2, ClE_L2, phiN_L2, phiG_L2, phiE_L2, pE_L2 \
        # update prev H1
        alphaN_H1_0, alphaG_H1_0, NaN_H1_0, NaG_H1_0, NaE_H1_0, \
                KN_H1_0, KG_H1_0, KE_H1_0, ClN_H1_0, ClG_H1_0, ClE_H1_0,\
                phiN_H1_0, phiG_H1_0, phiE_H1_0, pE_H1_0 = alphaN_H1, \
                alphaG_H1, NaN_H1, NaG_H1, NaE_H1, \
                KN_H1, KG_H1, KE_H1, ClN_H1, ClG_H1, ClE_H1, phiN_H1, phiG_H1,\
                phiE_H1, pE_H1

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
    directory = "results/mms/1D"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create files for results
    title_f1 = directory + "/convergence_table_alpha_L2_s.txt"
    title_f2 = directory + "/convergence_table_Na_L2_s.txt"
    title_f3 = directory + "/convergence_table_K_L2_s.txt"
    title_f4 = directory + "/convergence_table_Cl_L2_s.txt"
    title_f5 = directory + "/convergence_table_phi_L2_s.txt"
    title_f6 = directory + "/convergence_table_p_L2_s.txt"

    title_f7 = directory + "/convergence_table_Na_H1_s.txt"
    title_f8 = directory + "/convergence_table_K_H1_s.txt"
    title_f9 = directory + "/convergence_table_Cl_H1_s.txt"
    title_f10 = directory + "/convergence_table_phi_H1_s.txt"
    title_f11 = directory + "/convergence_table_p_H1_s.txt"

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
    # baseline end time
    Tstop = 2*dt

    # space resolutions
    resolutions = [2, 3, 4, 5, 6, 7]
    # number of iterations
    i = 0

    for resolution in resolutions:
        # create mesh
        N = 2**resolution               # number of cells
        mesh = IntervalMesh(N, 0, 1)    # mesh
        h = mesh.hmin()                 # minimum diameter of cells
        boundary_point = "near(x[0], 0.0)"

        # time variables
        t_PDE = Constant(0.0)           # time constant
        t_ODE = Constant(0.0)           # time constant

        problem = ProblemMMS(mesh, boundary_point, t_PDE, t_ODE)
        # solve system
        S = Solver(problem, dt, Tstop, MMS_test=True)
        w = S.solve_system_godenov()

        print("-------------------------------")
        print("N", N)
        print("dt", dt)
        print("Tstop", Tstop)
        print("problem.t", float(problem.t_PDE))
        print("-------------------------------")

        # get sub functions
        alpha_N, alpha_G, Na_N, Na_G, Na_E, K_N, K_G, K_E, Cl_N, \
                Cl_G, Cl_E, phi_N, phi_G, phi_E, p_E  = S.w.split(deepcopy=True)

        # extract exact solutions
        exact_solutions = problem.exact_solutions
        for key in exact_solutions:
            exec('%s = exact_solutions["%s"]' % (key, key))

        # function space for exact solutions
        CG5 = FiniteElement('CG', mesh.ufl_cell(), 5) # define element
        V_CG = FunctionSpace(mesh, CG5)               # define function space

        # get exact solutions as functions
        alpha_N_e = interpolate(alphaNe, V_CG)   # Na intracellular
        alpha_G_e = interpolate(alphaGe, V_CG)   # Na intracellular

        Na_N_e = interpolate(NaNe, V_CG)         # Na intracellular
        Na_G_e = interpolate(NaGe, V_CG)         # Na intracellular
        Na_E_e = interpolate(NaEe, V_CG)         # Na extracellular

        K_N_e = interpolate(KNe, V_CG)           # K intracellular
        K_G_e = interpolate(KGe, V_CG)           # K intracellular
        K_E_e = interpolate(KEe, V_CG)           # K extracellular

        Cl_N_e = interpolate(ClNe, V_CG)         # Cl intracellular
        Cl_G_e = interpolate(ClGe, V_CG)         # Cl intracellular
        Cl_E_e = interpolate(ClEe, V_CG)         # Cl extracellular

        phi_N_e = interpolate(phiNe, V_CG)       # phi intracellular
        phi_G_e = interpolate(phiGe, V_CG)       # phi intracellular
        phi_E_e = interpolate(phiEe, V_CG)       # phi extracellular

        p_E_e = interpolate(pEe, V_CG)       # phi extracellular

        # get error L2
        alphaN_L2 = errornorm(alpha_N_e, alpha_N, "L2", degree_rise=4)
        alphaG_L2 = errornorm(alpha_G_e, alpha_G, "L2", degree_rise=4)
        NaN_L2 = errornorm(Na_N_e, Na_N, "L2", degree_rise=4)
        NaG_L2 = errornorm(Na_G_e, Na_G, "L2", degree_rise=4)
        NaE_L2 = errornorm(Na_E_e, Na_E, "L2", degree_rise=4)
        KN_L2 = errornorm(K_N_e, K_N, "L2", degree_rise=4)
        KG_L2 = errornorm(K_G_e, K_G, "L2", degree_rise=4)
        KE_L2 = errornorm(K_E_e, K_E, "L2", degree_rise=4)
        ClN_L2 = errornorm(Cl_N_e, Cl_N, "L2", degree_rise=4)
        ClG_L2 = errornorm(Cl_G_e, Cl_G, "L2", degree_rise=4)
        ClE_L2 = errornorm(Cl_E_e, Cl_E, "L2", degree_rise=4)
        phiN_L2 = errornorm(phi_N_e, phi_N, "L2", degree_rise=4)
        phiG_L2 = errornorm(phi_G_e, phi_G, "L2", degree_rise=4)
        phiE_L2 = errornorm(phi_E_e, phi_E, "L2", degree_rise=4)
        pE_L2 = errornorm(p_E_e, p_E, "L2", degree_rise=4)

        # get error H1
        alphaN_H1 = errornorm(alpha_N_e, alpha_N, "H1", degree_rise=4)
        alphaG_H1 = errornorm(alpha_G_e, alpha_G, "H1", degree_rise=4)
        NaN_H1 = errornorm(Na_N_e, Na_N, "H1", degree_rise=4)
        NaG_H1 = errornorm(Na_G_e, Na_G, "H1", degree_rise=4)
        NaE_H1 = errornorm(Na_E_e, Na_E, "H1", degree_rise=4)
        KN_H1 = errornorm(K_N_e, K_N, "H1", degree_rise=4)
        KG_H1 = errornorm(K_G_e, K_G, "H1", degree_rise=4)
        KE_H1 = errornorm(K_E_e, K_E, "H1", degree_rise=4)
        ClN_H1 = errornorm(Cl_N_e, Cl_N, "H1", degree_rise=4)
        ClG_H1 = errornorm(Cl_G_e, Cl_G, "H1", degree_rise=4)
        ClE_H1 = errornorm(Cl_E_e, Cl_E, "H1", degree_rise=4)
        phiN_H1 = errornorm(phi_N_e, phi_N, "H1", degree_rise=4)
        phiG_H1 = errornorm(phi_G_e, phi_G, "H1", degree_rise=4)
        phiE_H1 = errornorm(phi_E_e, phi_E, "H1", degree_rise=4)
        pE_H1 = errornorm(p_E_e, p_E, "H1", degree_rise=4)

        if i > 0:
            # L2 errors
            r_alphaN_L2 = np.log(alphaN_L2/alphaN_L2_0)/np.log(h/h0)
            r_alphaG_L2 = np.log(alphaG_L2/alphaG_L2_0)/np.log(h/h0)
            r_NaN_L2 = np.log(NaN_L2/NaN_L2_0)/np.log(h/h0)
            r_NaG_L2 = np.log(NaG_L2/NaG_L2_0)/np.log(h/h0)
            r_NaE_L2 = np.log(NaE_L2/NaE_L2_0)/np.log(h/h0)
            r_KN_L2 = np.log(KN_L2/KN_L2_0)/np.log(h/h0)
            r_KG_L2 = np.log(KG_L2/KG_L2_0)/np.log(h/h0)
            r_KE_L2 = np.log(KE_L2/KE_L2_0)/np.log(h/h0)
            r_ClN_L2 = np.log(ClN_L2/ClN_L2_0)/np.log(h/h0)
            r_ClG_L2 = np.log(ClG_L2/ClG_L2_0)/np.log(h/h0)
            r_ClE_L2 = np.log(ClE_L2/ClE_L2_0)/np.log(h/h0)
            r_phiN_L2 = np.log(phiN_L2/phiN_L2_0)/np.log(h/h0)
            r_phiG_L2 = np.log(phiG_L2/phiG_L2_0)/np.log(h/h0)
            r_phiE_L2 = np.log(phiE_L2/phiE_L2_0)/np.log(h/h0)
            r_pE_L2 = np.log(pE_L2/pE_L2_0)/np.log(h/h0)

            r_NaN_H1 = np.log(NaN_H1/NaN_H1_0)/np.log(h/h0)
            r_NaG_H1 = np.log(NaG_H1/NaG_H1_0)/np.log(h/h0)
            r_NaE_H1 = np.log(NaE_H1/NaE_H1_0)/np.log(h/h0)
            r_KN_H1 = np.log(KN_H1/KN_H1_0)/np.log(h/h0)
            r_KG_H1 = np.log(KG_H1/KG_H1_0)/np.log(h/h0)
            r_KE_H1 = np.log(KE_H1/KE_H1_0)/np.log(h/h0)
            r_ClN_H1 = np.log(ClN_H1/ClN_H1_0)/np.log(h/h0)
            r_ClG_H1 = np.log(ClG_H1/ClG_H1_0)/np.log(h/h0)
            r_ClE_H1 = np.log(ClE_H1/ClE_H1_0)/np.log(h/h0)
            r_phiN_H1 = np.log(phiN_H1/phiN_H1_0)/np.log(h/h0)
            r_phiG_H1 = np.log(phiG_H1/phiG_H1_0)/np.log(h/h0)
            r_phiE_H1 = np.log(phiE_H1/phiE_H1_0)/np.log(h/h0)
            r_pE_H1 = np.log(pE_H1/pE_H1_0)/np.log(h/h0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            alphaN_L2, r_alphaN_L2, alphaG_L2, r_alphaG_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_L2, r_NaN_L2, NaG_L2, r_NaG_L2,\
                            NaE_L2, r_NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_L2, r_KN_L2, KG_L2, r_KG_L2,\
                            KE_L2, r_KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_L2, r_ClN_L2, ClG_L2, r_ClG_L2,\
                            ClE_L2, r_ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)  \\\\' % (N,\
                            phiN_L2, r_phiN_L2, phiG_L2, r_phiG_L2,\
                            phiE_L2, r_phiE_L2))
            # write to file - L2/H1 err and rate - pressure
            f6.write('%g & %.2E(%.2f) \\\\' % (N, pE_L2, r_pE_L2))

            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_H1, r_NaN_H1, NaG_H1, r_NaG_H1,\
                            NaE_H1, r_NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_H1, r_KN_H1, KG_H1, r_KG_H1,\
                            KE_H1, r_KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_H1, r_ClN_H1, ClG_H1, r_ClG_H1,\
                            ClE_H1, r_ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)  \\\\' % (N,\
                            phiN_H1, r_phiN_H1, phiG_H1, r_phiG_H1,\
                            phiE_H1, r_phiE_H1))
            # write to file - L2/H1 err and rate - pressure
            f11.write('%g & %.2E(%.2f) \\\\' % (N, pE_H1, r_pE_H1))

                        # write to file - summary
            fsum_L2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_L2, r_NaE_L2, \
                            phiN_L2, r_phiN_L2, \
                            alphaN_L2, r_alphaN_L2, \
                            pE_L2, r_pE_L2))

            # write to file - summary
            fsum_H1.write('%g & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_H1, r_NaE_H1, \
                            phiN_H1, r_phiN_H1))

        # update prev h
        h0 = h
        # update prev L2
        alphaN_L2_0, alphaG_L2_0, NaN_L2_0, NaG_L2_0, NaE_L2_0, \
                KN_L2_0, KG_L2_0, KE_L2_0, ClN_L2_0, ClG_L2_0, ClE_L2_0,\
                phiN_L2_0, phiG_L2_0, phiE_L2_0, pE_L2_0 = alphaN_L2, \
                alphaG_L2, NaN_L2, NaG_L2, NaE_L2, KN_L2, KG_L2, \
                KE_L2, ClN_L2, ClG_L2, ClE_L2, phiN_L2, phiG_L2, phiE_L2, pE_L2 \
        # update prev H1
        NaN_H1_0, NaG_H1_0, NaE_H1_0, \
                KN_H1_0, KG_H1_0, KE_H1_0, ClN_H1_0, ClG_H1_0, ClE_H1_0,\
                phiN_H1_0, phiG_H1_0, phiE_H1_0, pE_H1_0 = NaN_H1, NaG_H1, NaE_H1, \
                KN_H1, KG_H1, KE_H1, ClN_H1, ClG_H1, ClE_H1, phiN_H1, phiG_H1,\
                phiE_H1, pE_H1

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

def time():
    # create directory for saving results if it does not already exist
    directory = "results/mms/1D"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create files for results
    title_f1 = directory + "/convergence_table_alpha_L2_t.txt"
    title_f2 = directory + "/convergence_table_Na_L2_t.txt"
    title_f3 = directory + "/convergence_table_K_L2_t.txt"
    title_f4 = directory + "/convergence_table_Cl_L2_t.txt"
    title_f5 = directory + "/convergence_table_phi_L2_t.txt"
    title_f6 = directory + "/convergence_table_p_L2_t.txt"

    title_f7 = directory + "/convergence_table_Na_H1_t.txt"
    title_f8 = directory + "/convergence_table_K_H1_t.txt"
    title_f9 = directory + "/convergence_table_Cl_H1_t.txt"
    title_f10 = directory + "/convergence_table_phi_H1_t.txt"
    title_f11 = directory + "/convergence_table_p_H1_t.txt"

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

    # time resolutions
    dts = [2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3]
    Tstop = dts[0]*2

    # spatial resolution
    N = 2**12                       # number of cells
    mesh = IntervalMesh(N, 0, 1)    # mesh
    h = mesh.hmin()                 # minimum diameter of cells
    boundary_point = "near(x[0], 0.0)"

    # number of iterations
    i = 0

    for dt in dts:
        # time variables
        t_PDE = Constant(0.0)           # time constant
        t_ODE = Constant(0.0)           # time constant

        problem = ProblemMMS(mesh, boundary_point, t_PDE, t_ODE)
        # solve system
        S = Solver(problem, dt, Tstop, MMS_test=True)
        w = S.solve_system_godenov()

        print("-------------------------------")
        print("N", N)
        print("dt", dt)
        print("Tstop", Tstop)
        print("problem.t", float(problem.t_PDE))
        print("-------------------------------")

        # get sub functions
        alpha_N, alpha_G, Na_N, Na_G, Na_E, K_N, K_G, K_E, Cl_N, \
                Cl_G, Cl_E, phi_N, phi_G, phi_E, p_E  = S.w.split(deepcopy=True)

        # extract exact solutions
        exact_solutions = problem.exact_solutions
        for key in exact_solutions:
            exec('%s = exact_solutions["%s"]' % (key, key))

        # function space for exact solutions
        CG5 = FiniteElement('CG', mesh.ufl_cell(), 5) # define element
        V_CG = FunctionSpace(mesh, CG5)               # define function space

        # get exact solutions as functions
        alpha_N_e = interpolate(alphaNe, V_CG)   # Na intracellular
        alpha_G_e = interpolate(alphaGe, V_CG)   # Na intracellular

        Na_N_e = interpolate(NaNe, V_CG)         # Na intracellular
        Na_G_e = interpolate(NaGe, V_CG)         # Na intracellular
        Na_E_e = interpolate(NaEe, V_CG)         # Na extracellular

        K_N_e = interpolate(KNe, V_CG)           # K intracellular
        K_G_e = interpolate(KGe, V_CG)           # K intracellular
        K_E_e = interpolate(KEe, V_CG)           # K extracellular

        Cl_N_e = interpolate(ClNe, V_CG)         # Cl intracellular
        Cl_G_e = interpolate(ClGe, V_CG)         # Cl intracellular
        Cl_E_e = interpolate(ClEe, V_CG)         # Cl extracellular

        phi_N_e = interpolate(phiNe, V_CG)       # phi intracellular
        phi_G_e = interpolate(phiGe, V_CG)       # phi intracellular
        phi_E_e = interpolate(phiEe, V_CG)       # phi extracellular

        p_E_e = interpolate(pEe, V_CG)       # phi extracellular

        # get error L2
        alphaN_L2 = errornorm(alpha_N_e, alpha_N, "L2", degree_rise=4)
        alphaG_L2 = errornorm(alpha_G_e, alpha_G, "L2", degree_rise=4)
        NaN_L2 = errornorm(Na_N_e, Na_N, "L2", degree_rise=4)
        NaG_L2 = errornorm(Na_G_e, Na_G, "L2", degree_rise=4)
        NaE_L2 = errornorm(Na_E_e, Na_E, "L2", degree_rise=4)
        KN_L2 = errornorm(K_N_e, K_N, "L2", degree_rise=4)
        KG_L2 = errornorm(K_G_e, K_G, "L2", degree_rise=4)
        KE_L2 = errornorm(K_E_e, K_E, "L2", degree_rise=4)
        ClN_L2 = errornorm(Cl_N_e, Cl_N, "L2", degree_rise=4)
        ClG_L2 = errornorm(Cl_G_e, Cl_G, "L2", degree_rise=4)
        ClE_L2 = errornorm(Cl_E_e, Cl_E, "L2", degree_rise=4)
        phiN_L2 = errornorm(phi_N_e, phi_N, "L2", degree_rise=4)
        phiG_L2 = errornorm(phi_G_e, phi_G, "L2", degree_rise=4)
        phiE_L2 = errornorm(phi_E_e, phi_E, "L2", degree_rise=4)
        pE_L2 = errornorm(p_E_e, p_E, "L2", degree_rise=4)

        # get error H1
        alphaN_H1 = errornorm(alpha_N_e, alpha_N, "H1", degree_rise=4)
        alphaG_H1 = errornorm(alpha_G_e, alpha_G, "H1", degree_rise=4)
        NaN_H1 = errornorm(Na_N_e, Na_N, "H1", degree_rise=4)
        NaG_H1 = errornorm(Na_G_e, Na_G, "H1", degree_rise=4)
        NaE_H1 = errornorm(Na_E_e, Na_E, "H1", degree_rise=4)
        KN_H1 = errornorm(K_N_e, K_N, "H1", degree_rise=4)
        KG_H1 = errornorm(K_G_e, K_G, "H1", degree_rise=4)
        KE_H1 = errornorm(K_E_e, K_E, "H1", degree_rise=4)
        ClN_H1 = errornorm(Cl_N_e, Cl_N, "H1", degree_rise=4)
        ClG_H1 = errornorm(Cl_G_e, Cl_G, "H1", degree_rise=4)
        ClE_H1 = errornorm(Cl_E_e, Cl_E, "H1", degree_rise=4)
        phiN_H1 = errornorm(phi_N_e, phi_N, "H1", degree_rise=4)
        phiG_H1 = errornorm(phi_G_e, phi_G, "H1", degree_rise=4)
        phiE_H1 = errornorm(phi_E_e, phi_E, "H1", degree_rise=4)
        pE_H1 = errornorm(p_E_e, p_E, "H1", degree_rise=4)

        if i > 0:
            # L2 errors
            r_alphaN_L2 = np.log(alphaN_L2/alphaN_L2_0)/np.log(dt/dt0)
            r_alphaG_L2 = np.log(alphaG_L2/alphaG_L2_0)/np.log(dt/dt0)
            r_NaN_L2 = np.log(NaN_L2/NaN_L2_0)/np.log(dt/dt0)
            r_NaG_L2 = np.log(NaG_L2/NaG_L2_0)/np.log(dt/dt0)
            r_NaE_L2 = np.log(NaE_L2/NaE_L2_0)/np.log(dt/dt0)
            r_KN_L2 = np.log(KN_L2/KN_L2_0)/np.log(dt/dt0)
            r_KG_L2 = np.log(KG_L2/KG_L2_0)/np.log(dt/dt0)
            r_KE_L2 = np.log(KE_L2/KE_L2_0)/np.log(dt/dt0)
            r_ClN_L2 = np.log(ClN_L2/ClN_L2_0)/np.log(dt/dt0)
            r_ClG_L2 = np.log(ClG_L2/ClG_L2_0)/np.log(dt/dt0)
            r_ClE_L2 = np.log(ClE_L2/ClE_L2_0)/np.log(dt/dt0)
            r_phiN_L2 = np.log(phiN_L2/phiN_L2_0)/np.log(dt/dt0)
            r_phiG_L2 = np.log(phiG_L2/phiG_L2_0)/np.log(dt/dt0)
            r_phiE_L2 = np.log(phiE_L2/phiE_L2_0)/np.log(dt/dt0)
            r_pE_L2 = np.log(pE_L2/pE_L2_0)/np.log(dt/dt0)

            r_NaN_H1 = np.log(NaN_H1/NaN_H1_0)/np.log(dt/dt0)
            r_NaG_H1 = np.log(NaG_H1/NaG_H1_0)/np.log(dt/dt0)
            r_NaE_H1 = np.log(NaE_H1/NaE_H1_0)/np.log(dt/dt0)
            r_KN_H1 = np.log(KN_H1/KN_H1_0)/np.log(dt/dt0)
            r_KG_H1 = np.log(KG_H1/KG_H1_0)/np.log(dt/dt0)
            r_KE_H1 = np.log(KE_H1/KE_H1_0)/np.log(dt/dt0)
            r_ClN_H1 = np.log(ClN_H1/ClN_H1_0)/np.log(dt/dt0)
            r_ClG_H1 = np.log(ClG_H1/ClG_H1_0)/np.log(dt/dt0)
            r_ClE_H1 = np.log(ClE_H1/ClE_H1_0)/np.log(dt/dt0)
            r_phiN_H1 = np.log(phiN_H1/phiN_H1_0)/np.log(dt/dt0)
            r_phiG_H1 = np.log(phiG_H1/phiG_H1_0)/np.log(dt/dt0)
            r_phiE_H1 = np.log(phiE_H1/phiE_H1_0)/np.log(dt/dt0)
            r_pE_H1 = np.log(pE_H1/pE_H1_0)/np.log(dt/dt0)

            # write to file - L2/H1 err and rate - alpha
            f1.write('%g & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            alphaN_L2, r_alphaN_L2, alphaG_L2, r_alphaG_L2))
            # write to file - L2/H1 err and rate - Na
            f2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_L2, r_NaN_L2, NaG_L2, r_NaG_L2,\
                            NaE_L2, r_NaE_L2))
            # write to file - L2/H1 err and rate - K
            f3.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_L2, r_KN_L2, KG_L2, r_KG_L2,\
                            KE_L2, r_KE_L2))
            # write to file - L2/H1 err and rate - Cl
            f4.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_L2, r_ClN_L2, ClG_L2, r_ClG_L2,\
                            ClE_L2, r_ClE_L2))
             # write to file - L2/H1 err and rate - Cl
            f5.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)  \\\\' % (N,\
                            phiN_L2, r_phiN_L2, phiG_L2, r_phiG_L2,\
                            phiE_L2, r_phiE_L2))
            # write to file - L2/H1 err and rate - pressure
            f6.write('%g & %.2E(%.2f) \\\\' % (N, pE_L2, r_pE_L2))

            # write to file - L2/H1 err and rate - Na
            f7.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            NaN_H1, r_NaN_H1, NaG_H1, r_NaG_H1,\
                            NaE_H1, r_NaE_H1))
            # write to file - L2/H1 err and rate - K
            f8.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            KN_H1, r_KN_H1, KG_H1, r_KG_H1,\
                            KE_H1, r_KE_H1))
            # write to file - L2/H1 err and rate - Cl
            f9.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) \\\\' % (N,\
                            ClN_H1, r_ClN_H1, ClG_H1, r_ClG_H1,\
                            ClE_H1, r_ClE_H1))
             # write to file - L2/H1 err and rate - Cl
            f10.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)  \\\\' % (N,\
                            phiN_H1, r_phiN_H1, phiG_H1, r_phiG_H1,\
                            phiE_H1, r_phiE_H1))
            # write to file - L2/H1 err and rate - pressure
            f11.write('%g & %.2E(%.2f) \\\\' % (N, pE_H1, r_pE_H1))

                        # write to file - summary
            fsum_L2.write('%g & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_L2, r_NaE_L2, \
                            phiN_L2, r_phiN_L2, \
                            alphaN_L2, r_alphaN_L2, \
                            pE_L2, r_pE_L2))

            # write to file - summary
            fsum_H1.write('%g & %.2E(%.2f) & %.2E(%.2f)\\\\' % (N, \
                            NaE_H1, r_NaE_H1, \
                            phiN_H1, r_phiN_H1))

        # update prev h
        dt0 = dt
        # update prev L2
        alphaN_L2_0, alphaG_L2_0, NaN_L2_0, NaG_L2_0, NaE_L2_0, \
                KN_L2_0, KG_L2_0, KE_L2_0, ClN_L2_0, ClG_L2_0, ClE_L2_0,\
                phiN_L2_0, phiG_L2_0, phiE_L2_0, pE_L2_0 = alphaN_L2, \
                alphaG_L2, NaN_L2, NaG_L2, NaE_L2, KN_L2, KG_L2, \
                KE_L2, ClN_L2, ClG_L2, ClE_L2, phiN_L2, phiG_L2, phiE_L2, pE_L2 \
        # update prev H1
        NaN_H1_0, NaG_H1_0, NaE_H1_0, \
                KN_H1_0, KG_H1_0, KE_H1_0, ClN_H1_0, ClG_H1_0, ClE_H1_0,\
                phiN_H1_0, phiG_H1_0, phiE_H1_0, pE_H1_0 = NaN_H1, NaG_H1, NaE_H1, \
                KN_H1, KG_H1, KE_H1, ClN_H1, ClG_H1, ClE_H1, phiN_H1, phiG_H1,\
                phiE_H1, pE_H1

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
    space_time()
    space()
    time()
