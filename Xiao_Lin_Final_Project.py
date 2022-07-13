"""
Che352 final project Xiao Lin
Designs new experiments based on known data using neural network regressor and Bayesian optimization
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sklearn.neural_network as NN
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import sys
# Reference: How to print colored output in spyder console
# https://stackoverflow.com/questions/39473297/how-do-i-print-colored-output-with-python-3
from colorama import Fore, Style, Back
from sklearn.model_selection import train_test_split
import winsound

n_group_members = 3
n_experiments_per_batch = 12
n_experiments_to_return = n_experiments_per_batch // n_group_members  # Each team member is expected to find 4 points
n_dims = 4  # number of dimensions
known_concentrations = [  # [LiCl], [KCl], [KOH], [NaCl] in ppm
    [0.00, 0.00, 0.00, 0.00],
    [5.21, 6.04, 4.71, 2.03],
    [5.29, 1.91, 2.82, 7.54],
    [5.52, 8.64, 8.05, 2.48],
    [1.90, 9.84, 6.70, 2.80],
    [2.04, 6.25, 6.53, 8.99],
    [9.75, 1.54, 6.99, 4.47],
    [0.18, 2.91, 3.81, 3.21],
    [9.43, 7.03, 1.36, 3.43],
    [8.12, 1.48, 0.59, 3.14],
    [4.20, 8.08, 0.10, 4.54],
    [10.00, 10.00, 10.00, 10.00],
    [10.00, 0.00, 8.98, 4.55],
    [10.00, 0.00, 10.00, 7.48],
    [10.00, 0.00, 10.00, 0.00],
    [0.00, 10.00, 0.00, 10.00],
    [10.00, 10.00, 0.00, 10.00],
    [2.28, 10.00, 10.00, 0.00]
]
known_conductivities = [4.23e-03, 6.05e-04, 3.51e-03, 3.86e-03,
                        2.40e-03, 1.96e-03, 8.52e-03, 2.21e-03,
                        4.79e-03, 6.72e-03, 5.46e-03, 6.81e-03,
                        1.55e-03, 1.39e-03, 1.40e-03, 1.98e-03,
                        1.45e-03, 1.90e-03]  # in S/m

# Reference: Create a dynamic progress bar
# https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
def printProgressBar(i, max, postText):
    n_bar = 20  # size of progress bar
    j = i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

# Remove one point from the dataset and use the rest of the points as the training set
def train_case_separation(x, y, testNum):
    trainX = np.delete(x, testNum, 0)
    trainY = np.delete(y, testNum, 0)
    # print(trainX, trainY)
    return trainX, trainY

# Get the removed point and set it as the testing set
def test_case_separation(x, y, testNum):
    testX = x[testNum, :]
    testY = y[testNum]
    # print(testX, testY)
    return testX, testY

# Calculate the mean predicted value with standard deviation using the ensemble estimators
def estimate(x, estimators):
    # print(x)
    ys = [nn.predict([x]) for nn in estimators]
    # print(ys)
    return np.mean(ys), np.std(ys)

def build_NN(hidden_layer_i, print_results=False, print_plots=False):
    """
    Build ensemble estimators from neural networks using the given concentration and conductivities
    hidden_layer_i: For sensitivity_analysis() to change the hidden_layer_size in MLPRegressor
    print_results: If set to true, print the expected values, predicted values and some metrics
    print_plots: If set to true, print the conductivity vs. impurity concentration plots, include the real data points
                 and NN model predictions
    """

    # Initialization
    x = np.array(known_concentrations)
    real_y = np.array(known_conductivities)
    testNum = 18  # 12 initial points + 6 non-duplicate points from round 1
    estimators = []

    # Create empty array to store the generated values
    fit_y = np.zeros(testNum)
    rel_err = np.zeros(testNum)

    # Generate NNs using training sets and add them into the estimator list
    for nn_i in range(testNum):
        trainX, trainY = train_case_separation(x, real_y, nn_i)
        # The following line can be used when 80/20 train/test splits are needed
        # trainX, testX, trainY, testY = train_test_split(x, real_y, test_size=0.2, random_state=nn_i)

        # Reference: Parameters of MLPRegressor()
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlpregressor#sklearn.neural_network.MLPRegressor
        # Alpha is set to 0.00149 to generate the best R2 value (from manual sensitivity analysis)
        # For small datasets, solver ‘lbfgs’ can converge faster and perform better
        # Set early_stopping to False since the dataset is small
        nn = NN.MLPRegressor(random_state=nn_i, hidden_layer_sizes=(hidden_layer_i,), alpha=0.00149, max_iter=1000,
                             activation='relu', solver='lbfgs', early_stopping=False)
        nn.fit(trainX, trainY)
        estimators.append(nn)

    # Test NNs using testing sets and store the predicted value and calculated relative errors
    for test_i in range(testNum):
        testX, testY = test_case_separation(x, real_y, test_i)
        mean, std = estimate(testX, estimators)
        # print(x[test_i, :])
        fit_y[test_i] = mean
        rel_err[test_i] = np.abs(mean - testY) / testY

    # Calculate the metrics for the model ( Good model = minimum mean absolute error & R-square value close to 1)
    mean_absolute_err = mean_absolute_error(real_y, fit_y)
    r2 = r2_score(real_y, fit_y)

    if print_results:
        print(f'Expected: \n {real_y} \n')
        print(f'Prediction: \n {fit_y} \n')
        print(f'Relative error: \n {rel_err} \n')
        print(f'Mean absolute error: {Fore.CYAN}{mean_absolute_err:2f}{Fore.RESET} \n')
        print(f'r2 score: {Fore.CYAN}{r2:2f}{Fore.RESET} \n')
        # print(f'estimators: \n {estimators} \n')

    if print_plots:
        impurity = np.array(["[LiCl] (ppm)", "[KCl] (ppm)", "[KOH] (ppm)", "[NaCl] (ppm)"])
        # Generate 4 scatter plots with real data and NN predictions
        for plot_i in range(0, 4):
            plt.scatter(x[:, plot_i], real_y, label='Real Data')
            plt.scatter(x[:, plot_i], fit_y, label='NN Model')
            plt.xlabel(impurity[plot_i])
            plt.ylabel("conductivity (S/m)")
            plt.legend()
            plt.show()

    # return r2 as a metric for function sensitivity_analysis(), and return estimators for further optimization
    return r2, estimators

# Use sensitivity analysis to find the best hidden layer sizes at the given alpha
def sensitivity_analysis():
    r2_list = np.zeros(101)  # Create empty array to store the generated R-square values

    # Pass in parameter hidden_layer_size(SA_i,) in the MLPRegressor, where SA_i is an integer from 1 to 100
    print(f'{Back.RED}Conducting sensitivity analysis on hidden_layer_size, please wait ...{Back.RESET}')
    for SA_i in range(1, 101):
        printProgressBar(SA_i, 100, "Finding best hidden_layer_size")
        r2, estimators = build_NN(SA_i, print_results=False, print_plots=False)
        r2_list[SA_i] = r2

    # Find the max R-square values with the corresponding optimized hidden layer size
    print(f'\n{Style.DIM}{Back.CYAN}Sensitivity analysis completed, generating results ...{Style.RESET_ALL}')
    max_i = np.argmax(r2_list)
    max_r2 = np.max(r2_list)
    print(f'\nMax at index: {Fore.CYAN}{max_i}{Fore.RESET}')
    print(f'Max R2: {Fore.CYAN}{max_r2}{Fore.RESET}\n')
    return max_i

# Combine mean and standard deviations (with an explore-weight) and use it in the scipy function minimize()
# explore_weight is chosen to be 0.1 after manual sensitivity analysis
def acquisition_function(X, estimators, explore_weight=0.1):
    mean, std = estimate(X, estimators)
    return -(mean + std * explore_weight)

def bayesian_optimization(estimators, opti_iter=100):
    """
    Conduct Bayesian optimizations and find the optimized conductivity and corresponding impurity concentrations,
    and also find the top four recommended future experiment points where the mean is the highest after optimizations
    estimators: the estimator calculated from function build_NN(), needed as an argument to pass into scipy minimize()
                and evaluate acquisition_function()
    opti_iter: The number of optimized points that will be generated, the default is set to 100
    """

    # Create a structured array to store the concentrations, the mean and standard deviation for each point
    opti_mat = np.zeros(opti_iter,
                        dtype=[('x1', 'f4'), ('x2', 'f4'), ('x3', 'f4'), ('x4', 'f4'), ('mean', 'f4'), ('std', 'f4'), ('adjusted_mean', 'f4')])
    # print(opti_mat)

    print(f'{Back.RED}Conducting Bayesian optimizations, please wait ...{Back.RESET}')
    np.random.seed(42)  # Set a random seed to generate deterministic results

    # Conduct Bayesian optimizations on each point and store the results to the structured array
    for attempt_i in range(opti_iter):
        printProgressBar(attempt_i+1,100,"Bayesian optimizations")
        init_guess = 10 * np.random.random(n_dims)  # Generate the set of four random impurity concentrations
        # print(init_guess)

        # Reference: Parameters of minimize()
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        result = minimize(acquisition_function, init_guess, args=estimators, method='L-BFGS-B',
                          bounds=[(0, 10), (0, 10), (0, 10), (0, 10)], options={'disp': False})
        mean, std = estimate(result.x, estimators)  # Use the optimized result to get the mean and standard deviation
        x1, x2, x3, x4 = result.x  # Unpack variables

        # Store the results to the structured array opti_mat
        opti_mat[attempt_i]['x1'] = x1
        opti_mat[attempt_i]['x2'] = x2
        opti_mat[attempt_i]['x3'] = x3
        opti_mat[attempt_i]['x4'] = x4
        opti_mat[attempt_i]['mean'] = np.round(mean, decimals=4)
        opti_mat[attempt_i]['std'] = np.round(std, decimals=4)
        opti_mat[attempt_i]['adjusted_mean'] = np.round(mean + 0.1 * std, decimals=2)
        
    print(f'\n{Style.DIM}{Back.GREEN}Bayesian optimizations completed, generating results ...{Style.RESET_ALL}')
    print(opti_mat)

    # Reference: Sort a structured numpy array in descending order and drop duplicates
    # https://stackoverflow.com/questions/46390376/drop-duplicates-from-structured-numpy-array-python3-x
    # Sort opti_mat based on the mean value in descending order
    helper1, helper2 = np.unique(opti_mat['adjusted_mean'][::-1], return_index=True)
    sorted_mat = opti_mat[::-1][helper2][::-1]
    # print(sorted_mat)

    # Transfer the top four concentration data from the sorted matrix to a normal numpy array
    experiments = np.zeros((4, 4))
    for exp_i in range(0, 4):
        experiments[exp_i] = np.array([np.round(sorted_mat[exp_i]['x1'], 2),
                                       np.round(sorted_mat[exp_i]['x2'], 2),
                                       np.round(sorted_mat[exp_i]['x3'], 2),
                                       np.round(sorted_mat[exp_i]['x4'], 2)])
    # print(experiments)

    # Find the maximum conductivity and corresponding concentrations (which is the first row in the sorted_mat)
    max_mean = sorted_mat[0]['mean']
    max_std = sorted_mat[0]['std']
    max_x = experiments[0, :]
    # print(max_x)
    print(f'\nMax conductivity found at concentrations:{Style.BRIGHT}{Fore.GREEN}{max_x}{Style.RESET_ALL}')
    print(f'  --Concentration of each impurity: {Style.BRIGHT}{Fore.GREEN}'
          f'[LiCl]={max_x[0]:.2f}ppm, '
          f'[KCl]={max_x[1]:.2f}ppm, '
          f'[KOH]={max_x[2]:.2f}ppm, '
          f'[NaCl]={max_x[3]:.2f}ppm{Style.RESET_ALL}')
    print(f'Max conductivity: {Style.BRIGHT}{Fore.GREEN}{max_mean:.2E} S/m{Style.RESET_ALL}')
    print(f'Uncertainty at this point: {Style.BRIGHT}{Fore.GREEN}{max_std:.2E} S/m{Style.RESET_ALL}\n')
    return experiments

def new_experiments():
    # Reference: Set array floating point precision
    # https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
    np.set_printoptions(precision=4)

    # Uncomment the line below to conduct the sensitivity analysis
    # hidden_layer_size = sensitivity_analysis()

    # Get the estimators and use it for Bayesian optimization
    # hidden_layer_size(40,) is determined to have the best R-square value from the previous sensitivity_analysis()
    r2, estimators = build_NN(40, print_results=True, print_plots=True)
    experiments = bayesian_optimization(estimators, 100)

    assert experiments.shape == (n_experiments_to_return, n_dims)
    return experiments

# Calculate suggested new experimental points from the NN prediction and Bayesian optimization
if __name__ == '__main__':  # when this script is run
    experiments = new_experiments()
    print('New experiments should be conducted at the following points:')
    print(f'{Style.BRIGHT}{Fore.GREEN}{experiments}{Style.RESET_ALL}')
    # Uncomment the line below to enable notification sound when the process is finished (Only works on Windows)
    # winsound.MessageBeep(1)
