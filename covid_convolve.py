import scipy as sp
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import joblib
from functools import partial

plt.style.use('seaborn-darkgrid')
import sub_units.load_data as load_data
from scipy.integrate import odeint
from time import time as get_time
import datetime
from scipy.optimize import Bounds
import matplotlib.dates as mdates
from os import path
import itertools
from scipy.special import logsumexp


import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm


n_bootstraps = 10
state = 'total' #'total'#'New York'#'California'

def norm(x, mu=0, std=0):
    return np.exp(-((x - mu) / std) ** 2) / (np.sqrt(2 * np.pi) * std)

def sigmoid_interp(t, loc, level1, level2, width=1.0):
    '''
    helper function for interpolation
    :return: float
    '''
    sigmoid_val = 1 / (1 + np.exp(-(t - loc) / width))
    return sigmoid_val * (level2 - level1) + level1

# Unit test for sigmoid_interp
for t in range(40):
    print(t / 10, sigmoid_interp(t / 10, 2.5, 1, 2))
    
def ODE_system(y, t, *p):
    '''
    system of ODEs to simulate
    '''
    di = sigmoid_interp(t, p[2], p[0], p[1]) * y[0]
    # NB: rates will "flatten" things out, but they have limited ability to represent delay
    #     for that you may need delay ODE solvers
    return [di]

def run_simulation(params, t_vals, SIP_date_in_days=None):
    '''
    run combined ODE and convolution simulation
    :param params: dictionary of relevant parameters
    :return: N
    '''
    if type(params) != dict:
        print('why is params not a dict in run_simulation?')
    # First we simulate how the growth rate results into total # of contagious
    param_tuple = tuple(params[x] for x in ['alpha_1', 'alpha_2']) + (SIP_date_in_days, )
    contagious = odeint(ODE_system,
                        [params['I_0']],
                        t_vals,
                        args=param_tuple)
    
    # then use convolution to simulate transition to positive
    convolution_kernel = norm(np.linspace(0, 100, 100), mu=params['contagious_to_positive_delay'],
                              std=params['contagious_to_positive_width'])
    convolution_kernel /= sum(convolution_kernel)
    positive = np.convolve(np.squeeze(contagious), convolution_kernel) * 0.1  # params['contagious_to_positive_mult']
    
    # then use convolution to simulate transition to positive
    convolution_kernel = norm(np.linspace(0, 100, 100), mu=params['contagious_to_deceased_delay'],
                              std=params['contagious_to_deceased_width'])
    convolution_kernel /= sum(convolution_kernel)
    deceased = np.convolve(np.squeeze(contagious), convolution_kernel) * params['contagious_to_deceased_mult']
    
    return np.vstack([np.squeeze(contagious), positive[:contagious.size], deceased[:contagious.size]])


def test_errfunc(in_params, 
                 SIP_date_in_days=None, 
                 burn_in=None, 
                 opt_return_sol=True, 
                 map_name_to_sorted_ind=None,
                 data_new_tested=None,
                 data_new_dead=None,
                 cases_bootstrap_indices=None,
                 deaths_bootstrap_indices=None):  # returns value to be squared for error
    
    # convert from list to dictionary (for compatibility with the least-sq solver
    if type(in_params) != dict and map_name_to_sorted_ind is not None:
        params = {key: in_params[ind] for key, ind in map_name_to_sorted_ind.items()}
    else:
        params = in_params.copy()
    
    sol = run_simulation(params, t_vals, SIP_date_in_days=SIP_date_in_days)
    
    # this allows us to compare the sum of infections to data
    new_tested_from_sol = sol[1]
    new_deceased_from_sol = sol[2]
    
    if cases_bootstrap_indices is None:
        print('not using bootstraps in loss function')
        cases_bootstrap_indices = list(range(len(data_new_tested)))
    if deaths_bootstrap_indices is None:
        print('not using bootstraps in loss function')
        deaths_bootstrap_indices = list(range(len(data_new_dead)))
    
    # compile each component of the error separately, then add        
    new_tested_err = [
        (np.log(data_new_tested[i]) - np.log(new_tested_from_sol[i + burn_in])) / np.log(np.sqrt(data_new_tested[i])) \
        for i in cases_bootstrap_indices]
    new_dead_err = [
        (np.log(data_new_dead[i]) - np.log(new_deceased_from_sol[i + burn_in])) / np.log(np.sqrt(data_new_dead[i])) \
        for i in deaths_bootstrap_indices]
    
    final_err = new_tested_err + new_dead_err
    
    if opt_return_sol:
        return final_err, sol
    else:
        return final_err

def get_log_likelihood(p, 
                     SIP_date_in_days=None, 
                     burn_in=None,
                     data_new_tested=None,
                     data_new_dead=None,
                     cases_bootstrap_indices=None,
                     deaths_bootstrap_indices=None,
                     opt_return_sol=False):
    
    err, sol = test_errfunc(p, 
                     SIP_date_in_days=SIP_date_in_days, 
                     burn_in=burn_in,
                     data_new_tested=data_new_tested,
                     data_new_dead=data_new_dead,
                     cases_bootstrap_indices=cases_bootstrap_indices,
                     deaths_bootstrap_indices=deaths_bootstrap_indices)
    
    return_val = -sum([max(-11, np.log(np.power(x, 2))) for x in err])
    
    if opt_return_sol:
        return return_val, sol
    else:
        return return_val
    

def solve_and_plot_solution(in_params, title=None, SIP_date_in_days=0, min_date=None, burn_in=0):
    '''
    Solve ODEs and plot relavant parts
    :param test_params: dictionary of parameters 
    :return: None
    '''
    
    params = in_params.copy()
    
    time0 = get_time()
    for i in range(100):
        # note the extra comma after the args list is to ensure we pass the whole shebang as a tuple otherwise errors!
        sol = run_simulation(params, t_vals, SIP_date_in_days=SIP_date_in_days)
    time1 = get_time()
    print(f'time to simulate (ms): {(time1 - time0) / 100 * 1000}')
    
    new_positive = sol[1]
    new_deceased = sol[2]
    
    min_plot_pt = burn_in
    max_plot_pt = min(len(sol[0]), len(series_data) + 14 + burn_in)
    data_plot_date_range = [min_date + datetime.timedelta(days=1) * i for i in range(len(series_data))]
    sol_plot_date_range = [min_date - datetime.timedelta(days=burn_in) + datetime.timedelta(days=1) * i for i in
                           range(len(sol[0]))][min_plot_pt:max_plot_pt]
    
    fig, ax = plt.subplots()
    ax.plot(sol_plot_date_range, [sol[0][i] for i in range(min_plot_pt, max_plot_pt)], 'blue',
            label='contagious')
    ax.plot(sol_plot_date_range, new_positive[min_plot_pt: max_plot_pt], 'green', label='positive')
    ax.plot(sol_plot_date_range, new_deceased[min_plot_pt: max_plot_pt], 'red', label='deceased')
    ax.plot(data_plot_date_range, data_new_tested, 'g.', label='confirmed cases')
    ax.plot(data_plot_date_range, data_new_dead, 'r.', label='confirmed deaths')
    
    # this removes the year from the x-axis ticks
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.yscale('log')
    plt.ylabel('cumulative numbers')
    plt.xlabel('day')
    plt.ylabel('new people each day')
    plt.ylim((0.5, None))
    plt.xlim((min_date + datetime.timedelta(days=day_of_100th_case - 10), None))
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()
    
    # for i in range(len(sol)):
    #     print(f'index: {i}, odeint_value: {sol[i]}, real_value: {[None, series_data[i]]}')

####
# Load Data
####

bootstrap_filename = f"{state.lower().replace(' ', '_')}_{n_bootstraps}_bootstraps.joblib"
state_data = load_data.get_state_data(state)
series_data = state_data['series_data']
SIP_date = state_data['sip_date']
min_date = state_data['min_date']
population = state_data['population']
n_count_data = series_data[:, 1].size
SIP_date_in_days = (SIP_date - min_date).days

#####
# Initialize and set params
#####

burn_in = 20
t_vals = np.linspace(-burn_in, 100, burn_in + 100 + 1)
bootstrap_sols = list()
bootstrap_params = list()
bootstrap_cases_jitter_magnitude = 0.1
bootstrap_deaths_jitter_magnitude = 0.05
n_prediction_pts = 100
day_of_100th_case = [i for i, x in enumerate(series_data[:, 1]) if x >= 100][0]
day_of_100th_death = [i for i, x in enumerate(series_data[:, 2]) if x >= 100][0]

cases_indices = list(range(day_of_100th_case, len(series_data)))
deaths_indices = list(range(day_of_100th_death, len(series_data)))

delta_t = 17
data_cum_tested = series_data[:, 1].copy()
data_new_tested = [data_cum_tested[0]] + [data_cum_tested[i] - data_cum_tested[i - 1] for i in
                                          range(1, len(data_cum_tested))]
data_cum_dead = series_data[:, 2].copy()
data_new_dead = [data_cum_dead[0]] + [data_cum_dead[i] - data_cum_dead[i - 1] for i in range(1, len(data_cum_dead))]
data_new_recovered = [0] * delta_t + data_new_tested
data_cum_recovered = np.cumsum(data_new_recovered)


#####
# Define Model
#####


sorted_init_condit_names = ['I_0']
sorted_param_names = [ 'alpha_1',
                       'alpha_2',
                       'contagious_to_positive_delay',
                       'contagious_to_positive_width',
                       #'contagious_to_positive_mult',
                       'contagious_to_deceased_delay',
                       'contagious_to_deceased_width',
                       'contagious_to_deceased_mult'
                      ]

bounds = {'I_0': (-100.0, 100.0),  # starting infections
           'alpha_1': (-1, 2),
           'alpha_2': (-1, 2),
           'contagious_to_positive_delay': (0, 100),
           'contagious_to_positive_width': (0, 20),
           #'contagious_to_positive_mult': (0, 2),
           'contagious_to_deceased_delay': (0, 100),
           'contagious_to_deceased_width': (0, 20),
           'contagious_to_deceased_mult': (0, 1),
           }


test_params = {'I_0': 2e-5,  # starting infections
               'alpha_1': 0.3,
               'alpha_2': 0.01,
               'contagious_to_positive_delay': 10,
               'contagious_to_positive_width': 5,
               #'contagious_to_positive_mult': 0.1,
               'contagious_to_deceased_delay': 17,
               'contagious_to_deceased_width': 8,
               'contagious_to_deceased_mult': 0.01,
               }

sorted_names = sorted_init_condit_names + sorted_param_names
map_name_to_sorted_ind = {val: ind for ind, val in enumerate(sorted_names)}

solve_and_plot_solution(test_params, 
                        title='Test plot just to check',
                        SIP_date_in_days=SIP_date_in_days, 
                        min_date=min_date, 
                        burn_in=burn_in)

#####
#
# Fit Model Params
#
# Let's do dumb, frequentist curve-fitting using the mean sq. error over the series
#   then bootstrap on the training data (sampling with replacement)
#   and add sqrt(N) jitter to the training data for each bootstrap data point
#     to reflect inherent ambiguity in our measurements
#
#####

if not path.exists(bootstrap_filename):
    print('Calculating bootstraps...')
    
    orig_test_params = [test_params[key] for key in sorted_names]
    for bootstrap_ind in range(n_bootstraps):
        # get bootstrap indices by concatenating cases and deaths
        bootstrap_tuples = [('cases', x) for x in cases_indices] + [('deaths', x) for x in deaths_indices]
        bootstrap_indices_tuples = np.random.choice(list(range(len(bootstrap_tuples))), len(bootstrap_tuples), replace=True)
        cases_bootstrap_indices = [bootstrap_tuples[i][1] for i in bootstrap_indices_tuples if
                                   bootstrap_tuples[i][0] == 'cases']
        deaths_bootstrap_indices = [bootstrap_tuples[i][1] for i in bootstrap_indices_tuples if
                                    bootstrap_tuples[i][0] == 'deaths']
        
        # Add normal-distributed jitter with sigma=sqrt(N)
        tested_jitter = [max(0.01, data_new_tested[i] + np.random.normal(0, np.sqrt(data_new_tested[i]))) for i in
                         range(len(data_new_tested))]
        dead_jitter = [max(0.01, data_new_dead[i] + np.random.normal(0, np.sqrt(data_new_dead[i]))) for i in
                       range(len(data_new_dead))]
        
        # error_weights = [1 if i > split_date - 5 else 0 for i in range(n_prediction_pts)]
        error_weights = [1] * n_prediction_pts
        
        passed_params = [test_params[key] for key in sorted_names]
        
        # NB: define the model constraints (mainly, positive values)
        optimize_test_errfunc = partial(test_errfunc,
                                        SIP_date_in_days=SIP_date_in_days,
                                        burn_in=burn_in,
                                        opt_return_sol=False,
                                        map_name_to_sorted_ind=map_name_to_sorted_ind,
                                        data_new_tested=tested_jitter,
                                        data_new_dead=dead_jitter,
                                        cases_bootstrap_indices=cases_bootstrap_indices,
                                        deaths_bootstrap_indices=deaths_bootstrap_indices)
        params = sp.optimize.least_squares(optimize_test_errfunc,
                                           passed_params,
                                           bounds=([bounds[name][0] for name in sorted_names],
                                                   [bounds[name][1] for name in sorted_names]))
        
        params_as_dict = {key: params.x[i] for i, key in enumerate(sorted_names)}
        sol = run_simulation(params_as_dict, t_vals, SIP_date_in_days=SIP_date_in_days)
        bootstrap_sols.append(sol)
        bootstrap_params.append(params_as_dict)
        print(f'bootstrap #{bootstrap_ind} of {n_bootstraps} ({bootstrap_ind / n_bootstraps * 100})%\n params: {params_as_dict}')
    
    joblib.dump({'bootstrap_sols': bootstrap_sols, 'bootstrap_params': bootstrap_params}, bootstrap_filename)
    
else:
    
    print(f'Loading bootstraps from {bootstrap_filename}...')
    bootstrap_dict = joblib.load(bootstrap_filename)
    bootstrap_sols = bootstrap_dict['bootstrap_sols']
    bootstrap_params = bootstrap_dict['bootstrap_params']


####
# Plot the figures
####

sol = bootstrap_sols[0]
fig, ax = plt.subplots()
min_plot_pt = burn_in
max_plot_pt = min(len(sol[0]), len(series_data) + 14 + burn_in)
data_plot_date_range = [min_date + datetime.timedelta(days=1) * i for i in range(len(series_data))]

for i in range(len(bootstrap_sols)):
    sol = bootstrap_sols[i]
    params = bootstrap_params[i]

    new_tested = sol[1]
    cum_tested = np.cumsum(new_tested)

    new_dead = sol[2]
    cum_dead = np.cumsum(new_dead)

    sol_plot_date_range = [min_date - datetime.timedelta(days=burn_in) + datetime.timedelta(days=1) * i for i in range(len(sol[0]))][min_plot_pt:max_plot_pt]

    # ax.plot(plot_date_range[min_plot_pt:], [(sol[i][0]) for i in range(min_plot_pt, len(sol[0))], 'b', alpha=0.1)
    # ax.plot(plot_date_range[min_plot_pt:max_plot_pt], [(sol[i][1]) for i in range(min_plot_pt, max_plot_pt)], 'g', alpha=0.1)

    ax.plot(sol_plot_date_range, [new_tested[i] for i in range(min_plot_pt, max_plot_pt)], 'g',
            alpha=5 / n_bootstraps)
    ax.plot(sol_plot_date_range, [new_dead[i] for i in range(min_plot_pt, max_plot_pt)], 'r',
            alpha=5 / n_bootstraps)

ax.plot(data_plot_date_range, data_new_tested, 'g.', label='cases')
ax.plot(data_plot_date_range, data_new_dead, 'r.', label='deaths')
fig.autofmt_xdate()

# this removes the year from the x-axis ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
plt.yscale('log')
plt.ylabel('new people each day')
plt.ylim((0.5, None))
plt.xlim((min_date + datetime.timedelta(days=day_of_100th_case - 10), None))
plt.legend()
# plt.title(f'{state} Data (points) and Model Predictions (lines)')
plt.show()

map_name_to_distro = dict()
n_params = len(bootstrap_params[0])
for param_name in sorted_names:
    param_distro = [bootstrap_params[i][param_name] for i in range(len(bootstrap_params))]
    map_name_to_distro[param_name] = np.array(param_distro)

for name in sorted_names:
    print(f'{name}: {map_name_to_distro[name].mean()}')

bootstrap_selection = np.random.choice(len(bootstrap_params))
solve_and_plot_solution(bootstrap_params[bootstrap_selection],
                        title='Sample solution plot just to check',
                        SIP_date_in_days=SIP_date_in_days, 
                        min_date=min_date, 
                        burn_in=burn_in)

#######
# Plot the parameter univariate distributions
#######

data = az.convert_to_inference_data(map_name_to_distro)
az.plot_posterior(data, round_to=3, credible_interval=0.9, show=True, group='posterior',
                  var_names=sorted_param_names)  # show=True allows for plotting within IDE

# Apply uniform priors with bounds:

bounds = {'I_0': (-100.0, 100.0),  # starting infections
          'alpha_1': (0, 1),
          'alpha_2': (-0.5, 0.5),
          'contagious_to_positive_delay': (0, 20),
          'contagious_to_positive_width': (0, 10),
          # 'contagious_to_positive_mult': (0, 2),
          'contagious_to_deceased_delay': (10, 40),
          'contagious_to_deceased_width': (1, 10),
          'contagious_to_deceased_mult': (0, 0.05),
          }

bootstrap_weights = [1] * len(bootstrap_params)
for bootstrap_ind in range(len(bootstrap_params)):
    for param_name, (lower, upper) in bounds.items():
        if lower is not None and bootstrap_params[bootstrap_ind][param_name] < lower:
            bootstrap_weights[bootstrap_ind] = 0
        if upper is not None and bootstrap_params[bootstrap_ind][param_name] > upper:
            bootstrap_weights[bootstrap_ind] = 0

map_name_to_distro = dict()
n_params = len(bootstrap_params[0])
for param_name in sorted_names:
    param_distro = [bootstrap_params[i][param_name] for i in range(len(bootstrap_params)) if bootstrap_weights[i] > 0.5]
    map_name_to_distro[param_name] = np.array(param_distro)

for name in sorted_names:
    print(f'{name}: {map_name_to_distro[name].mean()}')

# bootstrap_selection = np.random.choice(len(bootstrap_params))
# solve_and_plot_solution(bootstrap_params[bootstrap_selection])

# if we want floating-point weights, have to find the common multiple and do discrete sampling
# for example, this commented line doesn't change the HDP values, since it just duplicates all values
# data = az.convert_to_inference_data({key: np.array(list(val) + list(val)) for key,val in map_name_to_distro.items()})

data = az.convert_to_inference_data(map_name_to_distro)
az.plot_posterior(data, round_to=3, credible_interval=0.9, show=True, group='posterior',
                  var_names=sorted_param_names)  # show=True allows for plotting within IDE
