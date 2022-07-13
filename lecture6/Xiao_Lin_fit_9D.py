"""
A script for approximating the CO-NOx dataset DOI: doi:10.3906/elk-1807-87

Polynomial regression
Kernel regression
Neural networks
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn

def load_dataset():
    description_file = 'co_nox_dataset/description.csv'
    # load headers
    with open(description_file) as f:
        csv_header = next(f).strip().split(',')
        abbreviations = {}
        for line in f:
            columns = line.strip().split(',')
            name, abbrev, units = columns[0:3]
            abbreviations[abbrev] = name, units
            print(f'{abbrev}: {name} ({units})')
    # load data
    data = {}
    data_series = {}
    for year in range(2013, 2016):
        data[year] = []
        with open(f'co_nox_dataset/gt_{year}.csv') as f:
            csv_header = next(f).strip().split(',')
            for line in f:
                columns = line.strip().split(',')
                data[year].append(columns)
            data[year] = np.array(data[year], dtype=float)
        print('Read', data[year].shape, 'values from year', year)
        data_series[year] = {}
        for col_i, header in enumerate(csv_header):
            column = data[year][:, col_i]
            data_series[year][header] = column
    return abbreviations, csv_header, data, data_series

abbreviations, col_names, data, data_series = load_dataset()
fit_year, test_year = 2014, 2015
output_variables = ['CO', 'NOX']
fit_y = output_variables[1]  # which output to fit

# get fit and test inputs as one matrix each
fit_inputs = data[fit_year][:,:-len(output_variables)].T.copy()
test_inputs = data[test_year][:,:-len(output_variables)].T.copy()
# scale fitting inputs such that mean = 0, stddev = 1
fit_inputs -= np.mean(fit_inputs, axis=1).reshape(-1,1)
fit_inputs /= np.std(fit_inputs, axis=1).reshape(-1,1)
# scale test inputs same as fitting inputs
test_inputs -= np.mean(test_inputs, axis=1).reshape(-1,1)
test_inputs /= np.std(test_inputs, axis=1).reshape(-1,1)

# fit a constant to the whole dataset, ignoring inputs
fit_data_mean = np.mean(data_series[fit_year][fit_y])
test_data_mean = np.mean(data_series[test_year][fit_y])
fit_err = np.mean((fit_data_mean - data_series[fit_year][fit_y])**2)**0.5 / fit_data_mean * 100
test_err = np.mean((fit_data_mean - data_series[test_year][fit_y])**2)**0.5 / test_data_mean * 100
print(f'Constant: {fit_err = :.1f}% {test_err = :.1f}%')

# fit an N-D linear function
def linear_Nd(x, *params):
    x = np.exp(-x)
    return np.dot(params[:len(x)], x) + params[len(x)]
linpopt, pcov = scipy.optimize.curve_fit(linear_Nd, fit_inputs, data_series[fit_year][fit_y], p0=np.zeros(len(fit_inputs) + 1))
fit_err = np.mean((linear_Nd(fit_inputs, *linpopt) - data_series[fit_year][fit_y])**2)**0.5 / fit_data_mean * 100
test_err = np.mean((linear_Nd(test_inputs, *linpopt) - data_series[test_year][fit_y])**2)**0.5 / test_data_mean * 100
print(f'Linear: {fit_err = :.1f}% {test_err = :.1f}%')

# fit an N-D quadratic function
def quadratic_Nd(x, *params):
    return np.dot(params[:len(x)], x) + np.dot(params[len(x):len(x)*2], x**2) + params[len(x)*2]
quadpopt, pcov = scipy.optimize.curve_fit(quadratic_Nd, fit_inputs, data_series[fit_year][fit_y], p0=np.zeros(len(fit_inputs) * 2 + 1))
fit_err = np.mean((quadratic_Nd(fit_inputs, *quadpopt) - data_series[fit_year][fit_y])**2)**0.5 / fit_data_mean * 100
test_err = np.mean((quadratic_Nd(test_inputs, *quadpopt) - data_series[test_year][fit_y])**2)**0.5 / test_data_mean * 100
print(f'Quadratic: {fit_err = :.1f}% {test_err = :.1f}%')

# fit an N-D cubic function
def cubic_Nd(x, *params):
    return np.dot(params[:len(x)], x) + np.dot(params[len(x):len(x)*2], x**2) + np.dot(params[len(x)*2:len(x)*3], x**3) + params[len(x)*3]
cubpopt, pcov = scipy.optimize.curve_fit(cubic_Nd, fit_inputs, data_series[fit_year][fit_y], p0=np.zeros(len(fit_inputs) * 3 + 1))
fit_err = np.mean((cubic_Nd(fit_inputs, *cubpopt) - data_series[fit_year][fit_y])**2)**0.5 / fit_data_mean * 100
test_err = np.mean((cubic_Nd(test_inputs, *cubpopt) - data_series[test_year][fit_y])**2)**0.5 / test_data_mean * 100
print(f'Cubic: {fit_err = :.1f}% {test_err = :.1f}%')

# fit a small neural network
import sklearn.neural_network
NN = sklearn.neural_network.MLPRegressor(random_state=1, hidden_layer_sizes=(10,), max_iter=1000, early_stopping=True)
model = NN.fit(fit_inputs.T, data_series[fit_year][fit_y])
fit_err = np.mean((model.predict(np.array(fit_inputs.T)) - data_series[fit_year][fit_y])**2)**0.5 / fit_data_mean * 100
test_err = np.mean((model.predict(np.array(test_inputs.T)) - data_series[test_year][fit_y])**2)**0.5 / test_data_mean * 100
print(f'NN: {fit_err = :.1f}% {test_err = :.1f}%')

# plot the NN fit compared to the linear fit and the real data
NN_test_preds = model.predict(test_inputs.T)
linear_test_preds = linear_Nd(test_inputs, *linpopt)
for abbrev in abbreviations:
    if abbrev in output_variables:
        continue
    # plot real data
    plt.scatter(data_series[test_year][abbrev], data_series[test_year][fit_y],
                label=fit_y + ' %d real' % test_year, marker='x', s=3)
    plt.xlabel(abbreviations[abbrev])
    plt.ylabel(abbreviations[fit_y])
    # plot model outputs
    plt.scatter(data_series[test_year][abbrev], linear_test_preds,
                label=fit_y + ' %d linear' % test_year, s=3)
    plt.scatter(data_series[test_year][abbrev], NN_test_preds,
                label=fit_y + ' %d NN' % test_year, s=3)
    plt.legend()
    plt.show()

# print linear coefficients of the inputs
print('')
print('Input   Linear coeff')
for abbrev, param in zip(col_names[:-len(output_variables)], linpopt):
    print(f'{abbrev:>4s}      {param:>6.2f}')
