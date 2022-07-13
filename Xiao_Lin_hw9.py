import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Initial concentrations
C_A_0 = 3.0  # M
C_B_0 = 0.0  # M
C_C_0 = 0.0  # M


def solve_reactor(inputs, print_results=False, print_plots=False):
    '''
    Solve reactor IVP
    inputs[0] = temperature in Kelvin
    inputs[1] = reactor residence time in minutes
    returns: costs - benefits in $
    '''
    T = inputs[0]
    t_minutes = inputs[1]
    t_sec = t_minutes * 60  # Convert to seconds

    def fun(t, u):
        C_A, C_B, C_C = u

        # Define constant terms
        Ea1 = 6.0
        Ea2 = 8.0
        k1 = 30
        k2 = 90
        R = 1.987e-3  # in the unit of kcal/K−mol

        # Define system of IVPs
        dBdt = k1 * np.exp(-Ea1 / (R * T)) * C_A
        dCdt = k2 * np.exp(-Ea2 / (R * T)) * C_A
        dAdt = -(dBdt + dCdt)

        return np.array([dAdt, dBdt, dCdt])

    # Define initial values and constants
    t_final_seconds = t_sec
    u0 = np.array([C_A_0, C_B_0, C_C_0])
    reactant_cost = 50  # per M of [A]
    operating_cost = 10  # $ per minute
    product_benefit = 200  # M of [B]

    # solve with IVP solver
    sol = solve_ivp(fun, (0.0, t_final_seconds), u0, method='BDF')
    tt = sol.t
    C_A, C_B, C_C = sol.y

    # Use total_cost = cost - benefit in order to utilize scipy.optimize.minimize
    total_cost = reactant_cost * C_A_0 + operating_cost * tt[-1] / 60 - product_benefit * C_B[-1]

    if print_results:
        print(f'{sol.success = }, {len(tt) = }')
        print('[A] final = %.2f' % C_A[-1])
        print('[B] final = %.2f' % C_B[-1])
        print('[C] final = %.2f' % C_C[-1])

    # plot the species concentration curves
    if print_plots:
        plt.plot(tt, C_A, label='[A]', linestyle='-')
        plt.plot(tt, C_B, label='[B]', linestyle='-')
        plt.plot(tt, C_C, label='[C]', linestyle='-')

        plt.legend()
        plt.xlabel('t (s)')
        plt.ylabel('Concentration (M)')
        plt.show()

    return total_cost


# Define initial guess for the optimizer
T0 = 300  # K
t_res0 = 5  # minutes
init_guess = np.array([T0, t_res0])

'''
The lines here are for testing purposes
profit_init = -solve_reactor(init_guess, print_results=True, print_plots=True)
print(profit_init)
'''

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
result = minimize(solve_reactor, init_guess, method='Nelder-Mead', options={'xtol':1e-6, 'disp':True})

# Print the results
print('Optimization successful?', result.success)
T, t_res = result.x
print('T (K) = %d' % T)
print('t_res (minutes) = %.2f ' % t_res)
profit = -solve_reactor(result.x, print_results=True, print_plots=True)
# print(profit)
print('Profit/[A] ($/M): %.2f' % (profit / C_A_0))
