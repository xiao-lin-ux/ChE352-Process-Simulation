"""
A script to interpolate a small dataset of 
Hvap vs T for ethylene carbonate electrolyte

Polynomial regression
Lagrange polynomials
Cubic splines

Data source: NIST
https://webbook.nist.gov/cgi/cbook.cgi?ID=C96491&Units=SI&Mask=4#Thermo-Phase
Last point is from Table 7, Yaws, Carl L. Thermophysical properties of chemicals and hydrocarbons. William Andrew, 2008.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate

Ts = np.array([383, 396, 423, 433, 517.5])
Hvaps = np.array([60.3, 59.6, 56.3, 55.0, 45.827])

plt.scatter(Ts, Hvaps, label='Expt data')
plt.xlabel('T (K)')
plt.ylabel('Hvap (kJ/mol)')

xx = np.linspace(min(Ts)-10, max(Ts)+10, 100)

# Add polynomial regression to the graph
poly_n = 3
poly = np.polyfit(Ts, Hvaps, poly_n)
poly_fit = np.polyval(poly, xx)
plt.plot(xx, poly_fit, color='red', label='Poly %d' % poly_n)

# Add Lagrange polynomials to the graph
Lagrange = scipy.interpolate.lagrange(Ts, Hvaps)
Lagrange_fit = np.polyval(Lagrange, xx)
plt.plot(xx, Lagrange_fit, color='purple', label='Lagrange')

# Add cubic splines to the graph
cubic = scipy.interpolate.CubicSpline(Ts, Hvaps)
plt.plot(xx, cubic(xx), color='orange', label='CubicSpline')

# Find the maximum difference in model Hvap values and the corresponding T value
diff = Lagrange_fit - poly_fit # The maximum difference appears between the Lagrange and poly regressions
max_diff = max(diff) 
max_pos = np.argmax(diff) # Find the index of the maximum value
max_T = xx[max_pos] # Get the corresponding T value
# The code below is for testing purposes 
#max_pos = list(diff).index(max_diff)
#max_T = (min(Ts) - 10 + (max(Ts) - min(Ts) + 20) * max_pos / 99)
#max_poly = np.polyval(poly, max_T)
#max_lagrange = np.polyval(Lagrange, max_T)
#print(max_pos, max_T, max_diff, max_poly, max_lagrange)
print(f'The largest gap between the model Hvap values is {max_diff = } kJ/mol')
print(f'T value at this point is {max_T = } K')

# Use a for loop to plot each method with 10 different values of Gaussian noise in the data
for i in range (10):
    noisy_Hvaps = np.random.normal(Hvaps, scale=0.1) # Generate new data points with Gaussian noise
    #print(noisy_Hvaps)
    
    # Use generated data points to plot polynomial regression
    poly_test = np.polyfit(Ts, noisy_Hvaps, poly_n)
    poly_test_fit = np.polyval(poly_test, xx)
    plt.plot(xx, poly_test_fit, '--', color='red')

    # Use generated data points to plot Lagrange polynomials
    Lagrange_test = scipy.interpolate.lagrange(Ts, noisy_Hvaps)
    Lagrange_test_fit = np.polyval(Lagrange_test, xx)
    plt.plot(xx, Lagrange_test_fit, '--', color='purple')

    # Use generated data points to plot cubic splines
    cubic_test = scipy.interpolate.CubicSpline(Ts, noisy_Hvaps)
    plt.plot(xx, cubic_test(xx), '--', color='orange')

plt.legend()
plt.savefig('hvap_fit.png', dpi=300)
plt.show()
