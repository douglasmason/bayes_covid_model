import scipy as sp
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import sub_units.load_data as load_data
from scipy.integrate import odeint
from time import time as get_time
import datetime
from scipy.optimize import Bounds
import matplotlib.dates as mdates
import time


####
# Load Data
####

state = 'total'#'California'#New York
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

bootstrap_sols = list()
bootstrap_params = list()
bootstrap_cases_jitter_magnitude = 0.1
bootstrap_deaths_jitter_magnitude = 0.05
n_bootstraps = 100
n_prediction_pts = 100
day_of_100th_case = [i for i, x in enumerate(series_data[:, 1]) if x >= 100][0]
day_of_100th_death = [i for i, x in enumerate(series_data[:, 2]) if x >= 100][0]

#####
#
# Add fall-off infected counts
#   the total number of infected individuals is not just the cumulative cases at time t
#   but the cumulative cases at time t minus the cumulative cases at time t - delta_t
#   where delta_t ~ 2 wks
#   Start with simple fall-off, can add a normal distribution of fall-offs later
#
#####

delta_t = 17
data_cum_tested = series_data[:, 1].copy()
data_new_tested = [data_cum_tested[0]] + [data_cum_tested[i] - data_cum_tested[i - 1] for i in range(1, len(data_cum_tested))]
data_cum_dead = series_data[:, 2].copy()
data_new_dead = [data_cum_dead[0]] + [data_cum_dead[i] - data_cum_dead[i - 1] for i in range(1, len(data_cum_dead))]
data_new_recovered = [0] * delta_t + data_new_tested
data_cum_recovered = np.cumsum(data_new_recovered)

#####
# Define Model
#####

def sigmoid_interp(t, loc, level1, level2, width=1.0):
    '''
    helper function for interpolation
    :return: float
    '''
    sigmoid_val = 1/(1 + np.exp(-(t - loc) / width))
    return sigmoid_val * (level2 - level1) + level1

for t in range(40):
    print(t/10, sigmoid_interp(t/10, 2.5, 1, 2))

dead_ind = 3
confirmed_ind = 2
infected_ind = 0
symptomatic_ind = 1
n_init_condits = 4
new_tested_param_name = 'delta'
new_tested_input_ind = 1

sorted_init_condit_names = ['I_0']
sorted_param_names =  ['delta_t',
                       'alpha_1',
                       'alpha_2',
                       'beta',
                       'delta',
                       'gamma',
                       'eta',
                       ]

bounds = {'I_0': (0, 10000),
          'delta_t': (0, 15),
          'alpha_1': (0, 2),
          'alpha_2': (0, 2),
          'beta': (0, 1),
          'delta': (0, 0.1),
          'gamma': (0, 1),
          'eta': (0, 1),}

test_params = {'I_0': 1.0, # starting infections
               'alpha_1': 0.4, # infection rate 1
               'alpha_2': 0.2, # infection rate 2
               'beta': 0.1, # infectious-to-confirmed rate
               'delta': 0.002,
               'gamma': 0.004, # confirmed-to-dead
               'eta': 0.1, # recovery rate
               'delta_t': 9, # delta_t (days after SIP date)
               }

sorted_names = sorted_init_condit_names + sorted_param_names
map_name_to_sorted_ind = {val: ind for ind, val in enumerate(sorted_names)}

def ODE_system(y, t, *p): 
    '''
    system of ODEs to simulate
    '''
    
    alpha = sigmoid_interp(t, p[0] + SIP_date_in_days, p[1], p[2])
    
    di = alpha * y[0] - (p[3] + p[6]) * y[0] # infectious
    ds = p[3] * y[0] - (p[4] + p[6]) * y[1]  # symptomatic
    dc = p[4] * y[1] - (p[5] + p[6]) * y[2]  # confirmed
    #dc = p[3] * y[0] - (p[4] + p[5]) * y[1]  # confirmed
    dd = p[5] * y[2] # dead
    # NB: rates will "flatten" things out, but they have limited ability to represent delay
    #     for that you need delay ODE solvers
    
    return [di, ds, dc, dd]


def solve_and_plot_solution(test_params):
    '''
    Solve ODEs and plot relavant parts
    :param test_params: dictionary of parameters
    :return: None
    '''
    time0 = get_time()
    for i in range(100):
        # note the extra comma after the args list is to ensure we pass the whole shebang as a tuple otherwise errors!
        sol = odeint(ODE_system,
                     [test_params['I_0']] + [0] * (n_init_condits - 1),
                     np.linspace(0, 100, 100),
                     args=tuple(test_params[x] for x in sorted_param_names))
    
    time1 = get_time()
    print(f'time to integrate (ms): {(time1 - time0) / 100 * 1000}')
    
    new_confirmed = test_params[new_tested_param_name] * sol[:, new_tested_input_ind]
    cum_confirmed = np.cumsum(new_confirmed)
    
    cum_dead = sol[:, dead_ind]
    new_dead = [sol[0, dead_ind]] + [sol[i, dead_ind] - sol[i - 1, dead_ind] for i in range(1, len(sol))]
    #print(new_dead)
    
    min_plot_pt = 0
    max_plot_pt = min(len(sol), len(series_data) + 14)
    data_plot_date_range = [min_date + datetime.timedelta(days=1) * i for i in range(len(series_data))][min_plot_pt:]
    sol_plot_date_range = [min_date + datetime.timedelta(days=1) * i for i in range(len(sol))][min_plot_pt:max_plot_pt]
    
    print(len(data_plot_date_range), len(sol_plot_date_range))
    
    fig, ax = plt.subplots()
    ax.plot(sol_plot_date_range, [sol[i][infected_ind] for i in range(min_plot_pt, max_plot_pt)], 'blue', label='contagious (current)')
    ax.plot(sol_plot_date_range, [sol[i][symptomatic_ind] for i in range(min_plot_pt, max_plot_pt)], 'cyan', label='symptomatic (current)')
    ax.plot(sol_plot_date_range, new_confirmed[min_plot_pt: max_plot_pt], 'green', label='positive (new)')
    ax.plot(sol_plot_date_range, new_dead[min_plot_pt: max_plot_pt], 'red', label='deceased (new)')
    # ax.plot(sol_plot_date_range, new_recovered[min_plot_pt: max_plot_pt], 'grey', label='confirmed and recovered (new)')
    ax.plot(data_plot_date_range, data_new_tested, 'g.', label='confirmed cases (new)')
    # ax.plot(data_plot_date_range, series_data[:, 1], 'g.', label='confirmed cases (cumulative)')
    ax.plot(data_plot_date_range, data_new_dead, 'r.', label='confirmed deaths (new)')
    # ax.plot(data_plot_date_range, series_data[:, 2], 'r.', label='confirmed deaths (cumulative)')
    # ax.plot(data_plot_date_range, data_new_recovered, '.', color='grey', label='hypothesized recoveries (new)')
    fig.autofmt_xdate()
    # this removes the year from the x-axis ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.yscale('log')
    plt.ylabel('cumulative numbers')
    plt.xlabel('day')
    plt.ylabel('new people each day')
    plt.ylim((0.5, None))
    plt.xlim((min_date + datetime.timedelta(days=day_of_100th_case - 10), None))
    plt.legend()
    plt.show()
    
    # for i in range(len(sol)):
    #     print(f'index: {i}, odeint_value: {sol[i]}, real_value: {[None, series_data[i]]}')

bootstrap_selection = np.random.choice(len(bootstrap_params))
solve_and_plot_solution(bootstrap_params[bootstrap_selection])

solve_and_plot_solution(test_params)

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

orig_test_params = [test_params[key] for key in sorted_names]
for bootstrap_ind in range(n_bootstraps):
    
    bootstrap_indices = list(range(len(series_data)))
    bootstrap_indices = np.random.choice(bootstrap_indices, len(bootstrap_indices), replace=True)
    
    # Add normal-distributed jitter with sigma=sqrt(N)
    tested_jitter = [max(0.01, data_new_tested[i] + np.random.normal(0, np.sqrt(data_new_tested[i]))) for i in
                     range(len(data_new_tested))]
    dead_jitter = [max(0.01, data_new_dead[i] + np.random.normal(0, np.sqrt(data_new_dead[i]))) for i in
                   range(len(data_new_dead))]
    
    #error_weights = [1 if i > split_date - 5 else 0 for i in range(n_prediction_pts)]
    error_weights = [1] * n_prediction_pts    
    
    def test_errfunc(p):  # returns value to be squared for error
        sol = odeint(ODE_system,
                     [p[map_name_to_sorted_ind['I_0']]] + [0] * (n_init_condits - 1),
                     np.linspace(0, 100, 100),
                     args=tuple(p[map_name_to_sorted_ind[key]] for key in sorted_param_names))
        
        # this allows us to compare the sum of infections to data
        new_tested_from_sol = p[map_name_to_sorted_ind[new_tested_param_name]] * sol[:, new_tested_input_ind] # need to isolate the new infections term, which comes from the exposed counts
        
        cum_dead_from_sol = sol[:, dead_ind] # need to isolate the new infections term, which comes from the exposed counts
        new_dead_from_sol = cum_dead_from_sol[0] + [cum_dead_from_sol[i] - cum_dead_from_sol[i - 1] for i in range(1, len(cum_dead_from_sol))]
        
        # compile each component of the error separately, then add        
        new_tested_err = [
            np.log(tested_jitter[i]) - np.log(new_tested_from_sol[i]) * error_weights[
                i] for i in bootstrap_indices if i > day_of_100th_case]
        new_dead_err = [
            np.log(dead_jitter[i]) - np.log(new_dead_from_sol[i]) * error_weights[
                i] for i in bootstrap_indices if i > day_of_100th_death]
        
        final_err = new_tested_err + new_dead_err
                
        return final_err
    
    # after the first fit, use the previous solution as our initial condition
    if bootstrap_ind == 0:
        passed_params = [test_params[key] for key in sorted_names]
    else:
        passed_params = sorted_params.copy()
    
    # NB: define the model constraints (mainly, positive values)
    params = sp.optimize.least_squares(test_errfunc,
                                       passed_params,
                                       bounds=([bounds[name][0] for name in sorted_names],
                                               [bounds[name][1] for name in sorted_names]))
    # except:
    #     print(f'attempt #{n_tries} failed, trying again...')
    #     n_tries += 1
    
    p = {key: params.x[i] for i, key in enumerate(sorted_names)}
    sorted_params = [p[key] for key in sorted_names]
    sol = odeint(ODE_system,
                 [p['I_0']] + [0] * (n_init_condits - 1),
                 np.linspace(0, 100, 100),
                 args=tuple(p[x] for x in sorted_param_names))
    bootstrap_sols.append(sol)
    bootstrap_params.append(p)
    print(f'bootstrap #{bootstrap_ind} of {n_bootstraps} ({bootstrap_ind/n_bootstraps*100})%\n params: {p}')

####
# Plot the figures
####

fig, ax = plt.subplots()
min_plot_pt = 0
max_plot_pt = min(len(sol), len(series_data) + 14)
data_plot_date_range = [min_date + datetime.timedelta(days=1) * i for i in range(len(series_data))][min_plot_pt:]

for i in range(len(bootstrap_sols)):
    
    sol = bootstrap_sols[i]
    params = bootstrap_params[i]
    
    new_tested = params[new_tested_param_name] * sol[:, new_tested_input_ind]
    cum_tested = np.cumsum(new_tested)
    
    cum_dead = sol[:, dead_ind]
    new_dead = [cum_dead[0]] + [cum_dead[i] - cum_dead[i - 1] for i in range(1, len(sol))]
    
    sol_plot_date_range = [min_date + datetime.timedelta(days=1) * i for i in range(len(sol))][min_plot_pt:max_plot_pt]
    
    #ax.plot(plot_date_range[min_plot_pt:], [(sol[i][0]) for i in range(min_plot_pt, len(sol))], 'b', alpha=0.1)
    #ax.plot(plot_date_range[min_plot_pt:max_plot_pt], [(sol[i][1]) for i in range(min_plot_pt, max_plot_pt)], 'g', alpha=0.1)
    
    ax.plot(sol_plot_date_range, [new_tested[i] for i in range(min_plot_pt, max_plot_pt)], 'g',
            alpha=5/n_bootstraps)
    ax.plot(sol_plot_date_range, [new_dead[i] for i in range(min_plot_pt, max_plot_pt)], 'r', 
            alpha=5/n_bootstraps)

ax.plot(data_plot_date_range data_new_tested, 'g.', label='cases')
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
#plt.title(f'{state} Data (points) and Model Predictions (lines)')
plt.show()

map_name_to_distro = dict()
n_params = len(bootstrap_params[0])
for param_name in sorted_names:
    param_distro = [bootstrap_params[i][param_name] for i in range(len(bootstrap_params))]
    map_name_to_distro[param_name] = np.array(param_distro)

for name in sorted_names:
    print(f'{name}: {map_name_to_distro[name].mean()}')

bootstrap_selection = np.random.choice(len(bootstrap_params))
solve_and_plot_solution(bootstrap_params[bootstrap_selection])

data = az.convert_to_inference_data(map_name_to_distro)
az.plot_posterior(data, round_to=2, credible_interval=0.9, show=True, group='posterior', 
                  var_names=sorted_param_names) # show=True allows for plotting within IDE

