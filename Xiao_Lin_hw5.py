# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:35:35 2022

@author: Xiao Lin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the temperature function
def temp(t):
    lambda1 = 30.0 
    lambda2 = np.pi / 120
    return lambda1 * np.cos(lambda2 * t) + 350

# Define the desgin equation for this batch reactor
def fun(t, C_A):
    k0 = 4.0e3
    E0 = 5500 # Define E0 as E_act / R
    return -k0 * np.exp(-E0 / temp(t)) * C_A

# Use Euler's method to solve the IVP (only execute one step)
def euler_step(f, ti, wi, h):
    w = wi + h * f(ti, wi)
    #print(w)
    return w

# Use RK4 method to solve thr IVP (only execute one step)
def rk4_step(f, ti, wi, h):
    k1 = h * f(ti, wi)
    k2 = h * f(ti + h/2, wi + k1/2)
    k3 = h * f(ti + h/2, wi + k2/2)
    k4 = h * f(ti + h, wi + k3)
    w = wi + (k1 + 2*k2 +2*k3 +k4)/6
    #print(w)
    return w

# Use Euler and RK4 to solve for C_B and return the vector of guesses
def ivp_solver(t0, t_final, wi, C_Bi, h):
    ts = np.arange(t0, t_final, h)
    #print(ts)
    
    # Initialize the vector of guesses for both methods
    euler = [wi]
    rk = [wi]
    
    # use a loop to solve for C_A numerically
    for ti in ts:
        euler_i = euler[-1] # Set wi = the last element in the vector
        euler_next = euler_step(fun, ti, euler_i, h) # Get the next guess by calliing the step function
        euler.append(euler_next) # Attach the obtained value to the end of the vector
        rk_i = rk[-1]
        rk_next = rk4_step(fun, ti, rk_i, h)
        rk.append(rk_next)
        
    #print(euler[-1])
    #print(rk[-1])
    
    # Solve for C_B using the formula C_B = C_B0 + del(C_A)
    C_B_euler = C_B0 + (euler[0] - euler[-1])
    C_B_rk = C_B0 + (rk[0] - rk[-1])
    
    return euler, rk, C_B_euler, C_B_rk
    
# Solve and plot the results calculated from solve_ivp, RK4, and Euler
def solve_and_plot(h):   
    tt = np.arange(t0, t_final+1, h) # Use t_final+1 to match the number of output points
    euler, rk, C_B_euler, C_B_rk = ivp_solver(t0, t_final, C_A0, C_B0, h) # Get the results of Euler and RK4 by calling ivp_solver
    sol = solve_ivp(fun, (t0, t_final), [C_A0], max_step = h) # Get the results by calling solve_ivp from scipy
    #print(sol.y[0])
    C_B_solve_ivp = C_B0 + (sol.y[0, 0] - sol.y[0, -1]) # Solve for C_B for the solve_ivp method
    
    #print the results
    print(f'Using {h = }, C_B(30 min) for each method:')
    print(f'{C_B_solve_ivp = :.3} mol/m^3')
    print(f'{C_B_rk = :.3} mol/m^3')
    print(f'{C_B_euler = :.3} mol/m^3''\n')
    
    # Plot the C_A vs. t for the three methods
    plt.plot(sol.t, sol.y[0], label = 'solve_ivp h=%g'% h, linestyle='-')
    plt.plot(tt, rk, label = 'RK4 h=%g'% h, linestyle='--')
    plt.plot(tt, euler, label = 'Euler h=%g'% h, linestyle=':')  
    return 0

# Define initial values
t0 = 0; t_final = 1800.0 
C_A0 = C_B0 = 3.16

# Run the function using different h values
for h in [1.0, 10.0, 100.0]:
    solve_and_plot(h)

#solve_and_plot(100) # plot the graphs when h is set to 100 for demostration

# Generate the curves in one plot
plt.legend()
plt.xlabel('t (s)')
plt.ylabel('C_A (mol/m^3)')
plt.show()

'''
lines below are for testing purposes
fun = fun(t0, C_A0)
print(fun)
euler_step(fun, t0, C_A0, h)
rk4_step(fun, t0, C_A0, h)
'''