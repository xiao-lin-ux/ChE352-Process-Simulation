# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:19:02 2022

@author: lunac
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define reactor as an object
class reactor:
    def __init__(self, sol=None, max_T=None, C_C_final=None):
        self.sol = sol
        self.max_T = max_T
        self.C_C_final = C_C_final

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

# Solve the system of IVPs using solve_ivp from scipy and return a reactor object with calculated attributes
def ivp_solver(reactor, k_therm, h):
    # Reference: Passing in extra parameters
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    # Using args to pass in k_therm to sys_fun
    # Can use other methods, like BDF or Radau, to conduct sensitivity analysis
    reactor.sol = solve_ivp(sys_fun, [t0, t_final], u0, args=[k_therm], max_step=h, method='RK45')
    C_A, C_B, C_C, T = reactor.sol.y

    reactor.max_T = max(T)
    reactor.C_C_final = C_C[-1]

    return reactor

# plot concentration and temperature vs. reaction time, and print the calculation results
def plot_and_print(reactor):
    # Reference: Plotting two axes in the same graph
    # https://pythonguides.com/matplotlib-two-y-axes/

    # Create the plot of Conc vs. t
    C_A, C_B, C_C, T = reactor.sol.y
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('t (s)')
    ax1.set_ylabel('Concentration (mol/m^3)')
    plot1 = ax1.plot(reactor.sol.t, C_A, label='C_A', linestyle='-.')
    plot2 = ax1.plot(reactor.sol.t, C_B, label='C_B', linestyle='--')
    plot3 = ax1.plot(reactor.sol.t, C_C, label='C_C', linestyle=':')

    # Add twin axes to plot Temp vs. t
    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature (K)')
    plot4 = ax2.plot(reactor.sol.t, T, label='T', linestyle='-', color='r')

    # Add the legend outside the figure to avoid blocking the curves
    plots = plot1 + plot2 + plot3 + plot4
    labels = [i.get_label() for i in plots]
    plt.legend(plots, labels, bbox_to_anchor=(1.12, 1), loc='upper left')
    #fig.savefig("IVP-system.png")
    plt.show()

    # Print the results from the calculation
    print('At the given initial conditions,')
    print(f'maximum temperature: {reactor.max_T = :.3} K')
    print(f'final concentration of the product: {reactor.C_C_final = :.3} mol/m^3''\n')
    return 0

# Find the lowest k_therm value without a thermal runaway
def find_k_thermal_runaway(k_therm, max_T, h, k_step):
    reactor_test = reactor()
    reactor_test.max_T = max_T
    while reactor_test.max_T <= 473.15:                      # Less than 200C, which is 473.15K
        k_therm -= k_step                                    # Set the step of guesses
        ivp_solver(reactor_test, k_therm, h)                 # Find the max_T for the new k_therm value
        #print(f'At {k_therm = :.3}, {reactor_test.max_T = :.3} K')
    # Return k_therm + step because k_therm results in a T_max higher than 473.15K
    print(f'When k_therm falls below {k_therm + k_step:.3}, there will be a thermal runaway')
    return 0

# Define initial values
t0 = 0; t_final = 300.0
#C_A0 = 5.00; C_B0 = 4.85; C_C0 = 0.00; T0 = 355
u0 = np.array([5.00, 4.85, 0.00, 355])
k_therm = 0.02
reactor_init = reactor()
h = 3  # Set default h = 3 to get 100 steps

# Solve the system of IVPs, plot and print the results, and find the lowest k_therm value without a thermal runaway
ivp_solver(reactor_init, k_therm, h)
plot_and_print(reactor_init)
find_k_thermal_runaway(k_therm, reactor_init.max_T, h, 1e-4)  # Default k_step is set to 1e-4
