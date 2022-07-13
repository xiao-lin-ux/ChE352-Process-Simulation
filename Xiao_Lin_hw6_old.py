# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:19:02 2022

@author: lunac
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of IVPs plus an adjustable k_therm parameter
def sys_fun(t, u, k_therm):
    C_A, C_B, C_C, T = u[0], u[1], u[2], u[3]  # Unpack variables

    # Define constant terms
    k_rxn = 30.0
    Ea = 7.05
    R = 1.987e-3  # in the unit of kcal/K−mol
    delH_rxn = 45.0
    # k_therm = 0.020
    T_set = 355

    # Define system of IVPs
    dCA_dt = -k_rxn * np.exp(-Ea / (R * T)) * C_A * C_B
    dCB_dt = dCA_dt
    dCC_dt = -dCA_dt
    dT_dt = delH_rxn * dCC_dt - k_therm * (T - T_set)

    # Define derivatives and return them in a numpy array
    return np.array([dCA_dt, dCB_dt, dCC_dt, dT_dt])

# Solve the system of IVPs using solve_ivp from scipy
def ivp_solver(k_therm, h):
    # Reference: Passing in extra parameters
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    sol = solve_ivp(sys_fun, [t0, t_final], u0, args=[k_therm], max_step=h)  # Using args to pass in k_therm to sys_fun
    C_A, C_B, C_C, T = sol.y

    max_T = max(T)
    C_C_final = C_C[-1]

    return sol, max_T, C_C_final

def plot_and_print(sol, max_T, C_C_final):
    # Reference: Plotting two axes in the same graph
    # https://pythonguides.com/matplotlib-two-y-axes/

    # Create the plot of Conc vs. t
    C_A, C_B, C_C, T = sol.y
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('t (s)')
    ax1.set_ylabel('Concentration (mol/m^3)')
    plot1 = ax1.plot(sol.t, C_A, label='C_A', linestyle='-.')
    plot2 = ax1.plot(sol.t, C_B, label='C_B', linestyle='--')
    plot3 = ax1.plot(sol.t, C_C, label='C_C', linestyle=':')

    # Add twin axes to plot Temp vs. t
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (K)')
    plot4 = ax2.plot(sol.t, T, label='T', linestyle='-', color='r')

    # Add the legend outside the figure to avoid blocking the curves
    plots = plot1 + plot2 + plot3 + plot4
    labels = [i.get_label() for i in plots]
    plt.legend(plots, labels, bbox_to_anchor=(1.12, 1), loc='upper left')
    #fig.savefig("IVP-system.png")
    plt.show()

    # Print the results from the calculation
    print('At the given initial conditions,')
    print(f'maximum temperature {max_T = :.3} K')
    print(f'final concentration of the product {C_C_final = :.3} mol/m^3''\n')
    return 0

# Find the lowest k_therm value without a thermal runaway
def find_k_thermal_runaway(k_therm, max_T, step):
    while max_T <= 473.15:                                # Less than 200C, which is 473.15K
        k_therm -= step                                   # Set the step of guesses
        sol, max_T, C_C_final = ivp_solver(k_therm, 0.1)  # Find the max_T for the new k_therm value
        print(f'At {k_therm = :.3}, {max_T = :.3} K')
    # Return k_therm + step because k_therm results in a T_max higher than 473.15K
    print('\n'f'When k_therm falls below {k_therm + step:.3}, there will be a thermal runaway')
    return 0

# Define initial values
t0 = 0; t_final = 300.0
# C_A0 = 5.00; C_B0 = 4.85; C_C0 = 0.00; T0 = 335
u0 = np.array([5.00, 4.85, 0.00, 355])
k_therm = 0.02

# Solve the system of IVPs, plot and print the results, and find the lowest k_therm value without a thermal runaway
sol, max_T, C_C_final = ivp_solver(k_therm, 0.1)  # Default h = 0.1
plot_and_print(sol, max_T, C_C_final)
find_k_thermal_runaway(k_therm, max_T, 5e-4) # Defualt step is set to 5e-4



