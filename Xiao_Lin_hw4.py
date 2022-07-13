"""
A script to test numerical derivative & integral methods
"""
import numpy as np

def dydx_midpoint(index, y_data, h):
    '''
    Numerical derivative of y_data at index,
    using the midpoint method
    3pt midpoint formula: (f(x0+h)-f(x0-h))/(2*h)
    '''
    #print(index, y_data, h)
    y_mid = (y_data[index + 1] - y_data[index - 1]) / (2 * h)
    #print(y_mid)
    return y_mid

def dydx_endpoint(index, y_data, h):
    '''
    Numerical derivative of y_data at index,
    using the midpoint method
    3pt endpoint formula: (-3*f(x0)+4*f(x0+h)-f(x0+2*h))/(2*h)
    '''
    y_end = (-3 * y_data[index] + 4 * y_data[index + 1] - y_data[index + 2]) / (2 * h)
    #print(y_end)
    return y_end

def test_midpoint_gradient():
    '''
    Test the midpoint formula on f = sin(k + x**2)
    '''
    xx = np.linspace(0.0, 5.0, 500)
    yy = np.sin(xx**2 + 1)
    dyy = 2 * xx * np.cos(xx**2 + 1)
    #print (xx[0], xx[1])
    dyy_approx = [dydx_midpoint(i, yy, xx[1] - xx[0]) for i in range(1, len(xx) - 1)]
    error = dyy_approx - dyy[1:len(xx) - 1]
    mean_abs_error = np.mean(np.abs(error))
    print(f'midpoint (skip 2) {mean_abs_error = :.3}')

def test_endpoint_gradient():
    '''
    Test the endpoint formula on f = sin(k + x**2)
    '''
    xx = np.linspace(0.0, 5.0, 500)
    yy = np.sin(xx**2 + 1)
    dyy = 2 * xx * np.cos(xx**2 + 1)
    dyy_approx = [dydx_endpoint(i, yy, xx[1] - xx[0]) for i in range(0, len(xx) - 2)]
    error = dyy_approx - dyy[0:len(xx) - 2]
    mean_abs_error = np.mean(np.abs(error))
    print(f'endpoint (skip 2) {mean_abs_error = :.3}')

def test_np_gradient():
    '''
    Test np.gradient() on f = sin(k + x**2)
    '''
    xx = np.linspace(0.0, 5.0, 500)
    yy = np.sin(xx**2 + 1)
    dyy = 2 * xx * np.cos(xx**2 + 1)
    dyy_approx = np.gradient(yy, xx)
    error = dyy_approx - dyy
    mean_abs_error = np.mean(np.abs(error))
    print(f'np.gradient (all) {mean_abs_error = :.3}')

def test_sympy_diff():
    '''
    Test sympy.diff() on f = sin(k + x**2)
    '''
    import sympy
    x, k = sympy.symbols('x k')
    f = sympy.sin(x**2 + k)
    df = sympy.diff(f, x)
    print(f'analytic {f = }')
    print(f'analytic {df = }')

def test_sympy_integrate():
    '''
    Test sympy.integrate() on f = sin(k + x**2)
    '''
    import sympy
    x, k = sympy.symbols('x k')
    f = sympy.sin(x**2 + k)
    intf = sympy.integrate(f, x) # Intergrated function f
    print(f'analytic {intf = }')
    
def test_integrate():
    '''
    Test numerical integration of f = sin(x**2 + 1)
    using scipy.integrate.quad as a reference
    '''
    from scipy.integrate import quad
    f = lambda x: np.sin(x**2 + 1)
    # test rectangle integration
    xx = np.linspace(0.0, np.pi, 100)
    h = xx[1] - xx[0]
    intf_approx = 0.0
    for i in range(len(xx)):
        if i < len(xx) - 1:
            #y = f(xx[i]) # Original rectangular formula
            #y = f((xx[i] + xx[i + 1]) / 2) # Use the midpoint rule for testing purposes 
            y = (f(xx[i]) + f(xx[i + 1])) / 2 # Use the trapezoidal rule
        else:
            y = 0.0  # last y is not used
        intf_approx += y * h
    # compare to scipy.integrate.quad
    intf_approx_quad = quad(f, 0.0, np.pi)[0]
    intf_error = intf_approx - intf_approx_quad
    print(f'{intf_error = :.3}')


test_midpoint_gradient()
test_endpoint_gradient()
test_np_gradient()

test_sympy_diff()
test_sympy_integrate()

test_integrate()
