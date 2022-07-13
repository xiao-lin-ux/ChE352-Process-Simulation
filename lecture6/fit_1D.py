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

Ts = np.array([383, 396, 423, 433, 517.5])
Hvaps = np.array([60.3, 59.6, 56.3, 55.0, 45.827])

plt.scatter(Ts, Hvaps, label='Expt data')
plt.xlabel('T (K)')
plt.ylabel('Hvap (kJ/mol)')

xx = np.linspace(min(Ts)-10, max(Ts)+10, 100)

poly_n = 3
poly = np.polyfit(Ts, Hvaps, poly_n)
poly_fit = np.polyval(poly, xx)
plt.plot(xx, poly_fit, color='red', label='Poly %d' % poly_n)

Lagrange_fit = scipy.interpolate.lagrange(Ts, Hvaps)
plt.plot(xx, Lagrange_fit, color= 'blue', label='')
plt.legend()
plt.savefig('hvap_fit.png', dpi=300)
plt.show()
