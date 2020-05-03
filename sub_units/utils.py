import numpy as np
# from sampyl import np as sampyl_np # for autograd
# from jax import numpy as np
# from jax.experimental.ode import odeint as odeint
import matplotlib.pyplot as plt
import scipy as sp
import joblib
from os import path
from functools import partial
from tqdm import tqdm
import os

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

plt.style.use('seaborn-darkgrid')
from scipy.integrate import odeint
from time import time as get_time
import datetime
import matplotlib.dates as mdates
import arviz as az


class ConvolutionModel:

    def __init__(self,
                 state_name,
                 max_date_str,
                 n_bootstraps=1000,
                 n_likelihood_samples=1000,
                 load_data_obj=None,
                 burn_in=20,
                 sorted_param_names=None,
                 sorted_init_condit_names=None,
                 curve_fit_bounds=None,
                 tight_bounds=None,
                 medium_bounds=None,
                 loose_bounds=None,
                 priors=None,
                 test_params=None,
                 static_params=None):

        self.state_name = state_name
        self.max_date_str = max_date_str
        self.n_bootstraps = n_bootstraps
        self.n_likelihood_samples = n_likelihood_samples
        self.burn_in = burn_in
        self.max_date = datetime.datetime.strptime(max_date_str, '%Y-%m-%d')
        self.static_params = static_params

        self.bootstrap_filename = path.join('state_bootstraps',
                                            f"{state_name.lower().replace(' ', '_')}_{n_bootstraps}_bootstraps_max_date_{max_date_str.replace('-', '_')}.joblib")
        self.likelihood_samples_filename_format_str = path.join('state_likelihood_samples',
                                                                f"{state_name.lower().replace(' ', '_')}_{{}}_{n_likelihood_samples}_samples_max_date_{max_date_str.replace('-', '_')}.joblib")
        self.plot_filename_base = path.join('state_plots',
                                            f"{state_name.lower().replace(' ', '_')}_{n_bootstraps}_bootstraps_max_date_{max_date_str.replace('-', '_')}")

        if not os.path.exists(self.plot_filename_base):
            os.mkdir(self.plot_filename_base)

        if load_data_obj is None:
            from sub_units import load_data as load_data_obj

        state_data = load_data_obj.get_state_data(state_name)

        # I replaced this with the U.S. total so everyone's on the same playing field, otherwise: state_data['sip_date']
        self.SIP_date = datetime.datetime.strptime('2020-03-20', '%Y-%m-%d')

        self.min_date = state_data['min_date']
        self.population = state_data['population']
        self.n_count_data = state_data['series_data'][:, 1].size
        self.SIP_date_in_days = (self.SIP_date - self.min_date).days
        self.max_date_in_days = (self.max_date - self.min_date).days
        self.series_data = state_data['series_data'][:self.max_date_in_days, :]  # cut off recent days if desired

        self.t_vals = np.linspace(-burn_in, 100, burn_in + 100 + 1)
        self.day_of_100th_case = [i for i, x in enumerate(self.series_data[:, 1]) if x >= 100][0]
        self.day_of_100th_death = [i for i, x in enumerate(self.series_data[:, 2]) if x >= 100][0]

        self.cases_indices = list(range(self.day_of_100th_case, len(self.series_data)))
        self.deaths_indices = list(range(self.day_of_100th_death, len(self.series_data)))

        data_cum_tested = self.series_data[:, 1].copy()
        self.data_new_tested = [data_cum_tested[0]] + [data_cum_tested[i] - data_cum_tested[i - 1] for i in
                                                       range(1, len(data_cum_tested))]

        data_cum_dead = self.series_data[:, 2].copy()
        self.data_new_dead = [data_cum_dead[0]] + [data_cum_dead[i] - data_cum_dead[i - 1] for i in
                                                   range(1, len(data_cum_dead))]

        delta_t = 17
        self.data_new_recovered = [0.0] * delta_t + self.data_new_tested

        self.curve_fit_bounds = curve_fit_bounds
        self.tight_bounds = tight_bounds
        self.medium_bounds = medium_bounds
        self.loose_bounds = loose_bounds
        self.priors = priors
        self.test_params = test_params

        self.sorted_param_names = [name for name in sorted_param_names if name not in static_params]
        self.sorted_init_condit_names = sorted_init_condit_names
        self.sorted_names = self.sorted_init_condit_names + self.sorted_param_names

        self.map_name_to_sorted_ind = {val: ind for ind, val in enumerate(self.sorted_names)}

    @staticmethod
    def norm(x, mu=0, std=0):
        return np.exp(-((x - mu) / std) ** 2) / (np.sqrt(2 * np.pi) * std)

    @staticmethod
    def ODE_system(y, t, *p):
        '''
        system of ODEs to simulate
        '''

        def sigmoid_interp(t, loc, level1, level2, width=1.0):
            '''
            helper function for interpolation
            :return: float
            '''
            sigmoid_val = 1 / (1 + np.exp(-(t - loc) / width))
            return sigmoid_val * (level2 - level1) + level1

        di = sigmoid_interp(t, p[2], p[0], p[1]) * y[0]
        # NB: rates will "flatten" things out, but they have limited ability to represent delay
        #     for that you may need delay ODE solvers
        return [di]

    def run_simulation(self, in_params):
        '''
        run combined ODE and convolution simulation
        :param params: dictionary of relevant parameters
        :return: N
        '''
        if type(in_params) != dict:
            if self.map_name_to_sorted_ind is None:
                print("Whoa! I have to create a dictionary but I don't have the proper mapping object")
            params = {key: in_params[ind] for key, ind in self.map_name_to_sorted_ind.items()}
        else:
            params = in_params.copy()

        params.update(self.static_params)

        # First we simulate how the growth rate results into total # of contagious
        param_tuple = tuple(params[x] for x in ['alpha_1', 'alpha_2']) + (self.SIP_date_in_days,)
        contagious = odeint(self.ODE_system,
                            [params['I_0']],
                            self.t_vals,
                            args=param_tuple)
        contagious = np.array(contagious)

        # then use convolution to simulate transition to positive
        convolution_kernel = self.norm(np.linspace(0, 100, 100), mu=params['contagious_to_positive_delay'],
                                       std=params['contagious_to_positive_width'])
        convolution_kernel /= sum(convolution_kernel)
        convolution_kernel = np.array(convolution_kernel)
        positive = np.convolve(np.squeeze(contagious),
                               convolution_kernel) * 0.1  # params['contagious_to_positive_mult']

        # then use convolution to simulate transition to positive
        convolution_kernel = self.norm(np.linspace(0, 100, 100), mu=params['contagious_to_deceased_delay'],
                                       std=params['contagious_to_deceased_width'])
        convolution_kernel /= sum(convolution_kernel)
        convolution_kernel = np.array(convolution_kernel)
        deceased = np.convolve(np.squeeze(contagious), convolution_kernel) * params['contagious_to_deceased_mult']

        return np.vstack([np.squeeze(contagious), positive[:contagious.size], deceased[:contagious.size]])

    # returns value to be squared for error
    def test_errfunc(self,
                     in_params,
                     data_new_tested=None,
                     data_new_dead=None,
                     opt_return_sol=True,
                     cases_bootstrap_indices=None,
                     deaths_bootstrap_indices=None):
        '''
        Returns the distance in log-space between simulation and observables, plus adds two loss terms for
          unphysical results (negative solutions and later confirmed than deceased delays)
        :param in_params: dictionary or list of parameters to obtain errors for
        :param data_new_tested: list of observables (passable since we may want to add jitter)
        :param data_new_dead: list of observables (passable since we may want to add jitter)
        :param opt_return_sol: boolean for whether to return the solution from the simulation
        :param cases_bootstrap_indices: bootstrap indices when applicable
        :param deaths_bootstrap_indices: bootstrap indices when applicable
        :return: list of error values (one for each observabe, plus two we added)
        '''

        if data_new_tested is None:
            data_new_tested = self.data_new_tested
            data_new_dead = self.data_new_dead

        # convert from list to dictionary (for compatibility with the least-sq solver
        if type(in_params) != dict and self.map_name_to_sorted_ind is not None:
            params = {key: in_params[ind] for key, ind in self.map_name_to_sorted_ind.items()}
        else:
            params = in_params.copy()

        sol = self.run_simulation(params)

        # this allows us to compare the sum of infections to data
        new_tested_from_sol = sol[1]
        new_deceased_from_sol = sol[2]

        if cases_bootstrap_indices is None:
            cases_bootstrap_indices = self.cases_indices
        if deaths_bootstrap_indices is None:
            deaths_bootstrap_indices = self.deaths_indices

        # print(f'sorted_cases_bootstrap_indices: {sorted(cases_bootstrap_indices)}')
        # print(f'deaths_bootstrap_indices: {sorted(deaths_bootstrap_indices)}')

        # compile each component of the error separately, then concatenate
        # NB: I don't think we want to divide by the expected std-dev (sqrt(N)) here
        revised_cases_indices = [i for i in cases_bootstrap_indices if data_new_tested[i] > 0]# and new_tested_from_sol[i + self.burn_in] > 0]
        revised_deaths_indices = [i for i in deaths_bootstrap_indices if data_new_dead[i] > 0]# and new_deceased_from_sol[i + self.burn_in] > 0]
        
        # rule out solutions with negative values
        err_from_negative_sols = sum([new_tested_from_sol[i + self.burn_in] for i in revised_cases_indices if
                                      new_tested_from_sol[i + self.burn_in] < 0])
        err_from_negative_sols += sum([new_deceased_from_sol[i + self.burn_in] for i in revised_cases_indices if
             new_deceased_from_sol[i + self.burn_in] < 0])
        err_from_negative_sols *= -1

        val1 = params['contagious_to_positive_delay']
        val2 = params['contagious_to_deceased_delay']
        err_from_reversed_delays = val1 - val2 if val1 > val2 else 0
        
        new_tested_err = [
            (np.log(data_new_tested[i]) - np.log(new_tested_from_sol[i + self.burn_in]))
            for i in revised_cases_indices]
        new_dead_err = [
            (np.log(data_new_dead[i]) - np.log(new_deceased_from_sol[i + self.burn_in]))
            for i in revised_deaths_indices]

        final_err = new_tested_err + new_dead_err + [err_from_negative_sols, err_from_reversed_delays]

        use_data = [data_new_tested[i] for i in revised_cases_indices] + \
                   [data_new_dead[i] for i in revised_deaths_indices]
        sigmas = [np.log(np.sqrt(val)) for val in use_data]

        if opt_return_sol:
            return final_err, sol, sigmas + [1, 1]
        else:
            return final_err

    def fit_curve_exactly_with_jitter(self,
                                      p0,
                                      data_tested=None,
                                      data_dead=None,
                                      tested_indices=None,
                                      deaths_indices=None):
        '''
        Given initial parameters, fit the curve with MSE
        :param p0: initial parameters
        :param data_tested: list of observables (passable since we may want to add jitter)
        :param data_dead: list of observables (passable since we may want to add jitter)
        :param tested_indices: bootstrap indices when applicable
        :param deaths_indices: bootstrap indices when applicable
        :return: optimized parameters as dictionary
        '''
        optimize_test_errfunc = partial(self.test_errfunc,
                                        opt_return_sol=False,
                                        data_new_tested=data_tested,
                                        data_new_dead=data_dead,
                                        cases_bootstrap_indices=tested_indices,
                                        deaths_bootstrap_indices=deaths_indices)
        results = sp.optimize.least_squares(optimize_test_errfunc,
                                           p0,
                                           bounds=(
                                               [self.curve_fit_bounds[name][0] for name in self.sorted_names],
                                               [self.curve_fit_bounds[name][1] for name in self.sorted_names]))
        
        params_as_list = results.x
        params_as_dict = {key: params_as_list[i] for i, key in enumerate(self.sorted_names)}
        return params_as_dict

    def fit_curve_via_likelihood(self,
                                 p0,
                                 tested_indices=None,
                                 deaths_indices=None
                                 ):
        '''
        Given initial parameters, fit the curve by minimizing log likelihood using measure error Gaussian PDFs
        :param p0: initial parameters
        :param data_tested: list of observables (passable since we may want to add jitter)
        :param data_dead: list of observables (passable since we may want to add jitter)
        :param tested_indices: bootstrap indices when applicable
        :param deaths_indices: bootstrap indices when applicable
        :return: optimized parameters as dictionary
        '''

        def get_neg_log_likelihood(p):
            return -self.get_log_likelihood(p,
                                            cases_bootstrap_indices=tested_indices,
                                            deaths_bootstrap_indices=deaths_indices
                                            )
        
        bounds_to_use = [self.curve_fit_bounds[name] for name in self.sorted_names]
        print(f'val to minimize at start: {get_neg_log_likelihood(p0)}')
        results = sp.optimize.minimize(get_neg_log_likelihood, p0, method='bfgs')#, bounds=bounds_to_use)
        params_as_list = results.x
        params_as_dict = {key: params_as_list[i] for i, key in enumerate(self.sorted_names)}

        return params_as_dict

    def get_log_likelihood(self,
                           p,
                           cases_bootstrap_indices=None,
                           deaths_bootstrap_indices=None,
                           opt_return_sol=False):
        '''
        Obtains the log likelihood for a given parameter dictionary
        :param p: dictionary or list of parameters
        :param cases_bootstrap_indices: bootstrap indices when applicable
        :param deaths_bootstrap_indices:  bootstrap indices when applicable
        :param opt_return_sol: boolean for returning solution fromm simulation
        :return: float of log likelihood, or tuple adding solution
        '''

        # err gives us the distance in the logs of the data and the simulation
        err, sol, sigmas = self.test_errfunc(p,
                                             cases_bootstrap_indices=cases_bootstrap_indices,
                                             deaths_bootstrap_indices=deaths_bootstrap_indices)

        if len(err) != len(sigmas):
            raise ValueError("error values not same length as sigma values")

        all_probs = list()
        for i in range(len(err)):
            get_prob = sp.stats.norm(0, sigmas[i]).pdf
            all_probs.append(get_prob(err[i]))

        return_val = sum(np.log(x) for x in all_probs)

        # return_val = -sum([max(-100, np.log(np.power(err[i], 2) / sigmas[i]**2)) for i in range(len(err))])

        if opt_return_sol:
            return return_val, sol
        else:
            return return_val

    def solve_and_plot_solution(self,
                                in_params=None,
                                title=None,
                                plot_filename_filename='test_plot'):
        '''
        Solve ODEs and plot relavant parts
        :param in_params: dictionary of parameters
        :param t_vals: values in time to plot against
        :param plot_filename_filename: string to add to plot filename
        :return: None
        '''

        if in_params is None:
            if hasattr(self, 'all_data_params'):
                print('Using params from model trained on all data')
                params = self.all_data_params
            else:
                print('Using default test_params')
                params = self.test_params.copy()
        else:
            params = in_params.copy()

        time0 = get_time()
        for i in range(100):
            # note the extra comma after the args list is to ensure we pass the whole shebang as a tuple otherwise errors!
            sol = self.run_simulation(params)
        time1 = get_time()
        print(f'time to simulate (ms): {(time1 - time0) / 100 * 1000}')

        new_positive = sol[1]
        new_deceased = sol[2]

        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + 14 + self.burn_in)
        data_plot_date_range = [self.min_date + datetime.timedelta(days=1) * i for i in range(len(self.series_data))]
        sol_plot_date_range = [self.min_date - datetime.timedelta(days=self.burn_in) + datetime.timedelta(days=1) * i
                               for i in
                               range(len(sol[0]))][min_plot_pt:max_plot_pt]

        fig, ax = plt.subplots()
        ax.plot(sol_plot_date_range, [sol[0][i] for i in range(min_plot_pt, max_plot_pt)], 'blue',
                label='contagious')
        ax.plot(sol_plot_date_range, new_positive[min_plot_pt: max_plot_pt], 'green', label='positive')
        ax.plot(sol_plot_date_range, new_deceased[min_plot_pt: max_plot_pt], 'red', label='deceased')
        ax.plot(data_plot_date_range, self.data_new_tested, 'g.', label='confirmed cases')
        ax.plot(data_plot_date_range, self.data_new_dead, 'r.', label='confirmed deaths')

        # this removes the year from the x-axis ticks
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.yscale('log')
        plt.ylabel('cumulative numbers')
        plt.xlabel('day')
        plt.ylabel('new people each day')
        plt.ylim((0.5, None))
        plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_100th_case - 10), None))
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.savefig(path.join(self.plot_filename_base, plot_filename_filename))

        # for i in range(len(sol)):
        #     print(f'index: {i}, odeint_value: {sol[i]}, real_value: {[None, series_data[i]]}')

    def plot_all_solutions(self, n_sols_to_plot=None):
        '''
        Plot all the bootstrap simulation solutions
        :param n_sols_to_plot: how many simulations should we sample for the plot?
        :return: None
        '''

        if n_sols_to_plot is None:
            n_sols_to_plot = len(self.bootstrap_sols)
        sols_to_plot = np.random.choice(len(self.bootstrap_sols), n_sols_to_plot, replace=False)

        self.plot_all_solutions_sub(sols_to_plot, plot_filename_filename='bootstrap_solutions.png')

        # In the future we can do float weights, for now it's just binary ones
        sols_to_plot = [i for i, sol in enumerate(self.bootstrap_sols) if round(self.bootstrap_weights[i])]

        self.plot_all_solutions_sub(sols_to_plot, plot_filename_filename='bootstrap_solutions_with_priors.png')

    def plot_all_solutions_sub(self,
                               sols_to_plot,
                               plot_filename_filename='bootstrap_solutions.png'):
        '''
        Helper function to plot_all_solutions
        :param n_sols_to_plot: how many simulations should we sample for the plot?
        :param plot_filename_filename: string to add to the plot filename
        :return: None
        '''

        sol = self.bootstrap_sols[0]
        n_bootstraps = len(self.bootstrap_sols)
        fig, ax = plt.subplots()
        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + 14 + self.burn_in)
        data_plot_date_range = [self.min_date + datetime.timedelta(days=1) * i for i in
                                range(len(self.data_new_tested))]

        for i in sols_to_plot:
            sol = self.bootstrap_sols[i]

            new_tested = sol[1]
            # cum_tested = np.cumsum(new_tested)

            new_dead = sol[2]
            # cum_dead = np.cumsum(new_dead)

            sol_plot_date_range = [self.min_date - datetime.timedelta(days=self.burn_in) + datetime.timedelta(
                days=1) * i for i in
                                   range(len(sol[0]))][min_plot_pt:max_plot_pt]

            # ax.plot(plot_date_range[min_plot_pt:], [(sol[i][0]) for i in range(min_plot_pt, len(sol[0))], 'b', alpha=0.1)
            # ax.plot(plot_date_range[min_plot_pt:max_plot_pt], [(sol[i][1]) for i in range(min_plot_pt, max_plot_pt)], 'g', alpha=0.1)

            ax.plot(sol_plot_date_range, [new_tested[i] for i in range(min_plot_pt, max_plot_pt)], 'g',
                    alpha=5 / n_bootstraps)
            ax.plot(sol_plot_date_range, [new_dead[i] for i in range(min_plot_pt, max_plot_pt)], 'r',
                    alpha=5 / n_bootstraps)

        ax.plot(data_plot_date_range, self.data_new_tested, 'g.', label='cases')
        ax.plot(data_plot_date_range, self.data_new_dead, 'r.', label='deaths')
        fig.autofmt_xdate()

        # this removes the year from the x-axis ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        plt.yscale('log')
        plt.ylabel('new people each day')
        plt.ylim((0.5, None))
        plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_100th_case - 10), None))
        plt.ylim((1, max(self.data_new_tested) * 100))
        plt.legend()
        # plt.title(f'{state} Data (points) and Model Predictions (lines)')
        plt.savefig(path.join(self.plot_filename_base, plot_filename_filename))

    @staticmethod
    def norm_2d(xv, yv, mu=(0, 0), sigma=(1, 1)):
        arg = -((xv - mu[0]) ** 2 / sigma[0] + (yv - mu[1]) ** 2 / sigma[1])
        vals = np.exp(arg)
        return vals

    def render_bootstraps(self):
        '''
        Compute the bootstrap solutions
        :return: None
        '''

        bootstrap_sols = list()
        bootstrap_params = list()
        successful_load = False

        if path.exists(self.bootstrap_filename):

            print(f'Loading bootstraps from {self.bootstrap_filename}...')
            try:
                bootstrap_dict = joblib.load(self.bootstrap_filename)
                bootstrap_sols = bootstrap_dict['bootstrap_sols']
                bootstrap_params = bootstrap_dict['bootstrap_params']
                all_data_params = bootstrap_dict['all_data_params']
                all_data_sol = bootstrap_dict['all_data_sol']
                successful_load = True
            except:
                print('Error loading bootstrap file!')
                successful_load = False
        else:
            print(f'No bootstraps file {self.bootstrap_filename} found.')

        if not successful_load:

            print('\n----\nRendering bootstrap model fits... starting with the all-data one...\n----')

            all_data_test_errfunc = partial(self.test_errfunc,
                                            opt_return_sol=False,
                                            data_new_tested=self.data_new_tested,
                                            data_new_dead=self.data_new_dead,
                                            cases_bootstrap_indices=self.cases_indices,
                                            deaths_bootstrap_indices=self.deaths_indices)

            test_params_as_list = [self.test_params[key] for key in self.sorted_names]
            all_data_params_as_list = sp.optimize.least_squares(all_data_test_errfunc,
                                                                test_params_as_list,
                                                                bounds=([self.curve_fit_bounds[name][0] for name in
                                                                         self.sorted_names],
                                                                        [self.curve_fit_bounds[name][1] for name in
                                                                         self.sorted_names]))
            all_data_params = {key: all_data_params_as_list.x[i] for i, key in enumerate(self.sorted_names)}
            all_data_sol = self.run_simulation(all_data_params)

            print('\nParameters when trained on all data (this is our starting point for optimization):')
            [print(f'{key}: {val:.4g}') for key, val in all_data_params.items()]

            print('\n----\nRendering bootstrap model fits... now going through bootstraps...\n----')
            for bootstrap_ind in tqdm(range(self.n_bootstraps)):
                # get bootstrap indices by concatenating cases and deaths
                bootstrap_tuples = [('cases', x) for x in self.cases_indices] + [('deaths', x) for x in
                                                                                 self.deaths_indices]
                bootstrap_indices_tuples = np.random.choice(list(range(len(bootstrap_tuples))), len(bootstrap_tuples),
                                                            replace=True)
                cases_bootstrap_indices = [bootstrap_tuples[i][1] for i in bootstrap_indices_tuples if
                                           bootstrap_tuples[i][0] == 'cases']
                deaths_bootstrap_indices = [bootstrap_tuples[i][1] for i in bootstrap_indices_tuples if
                                            bootstrap_tuples[i][0] == 'deaths']
                
                # Add normal-distributed jitter with sigma=sqrt(N)
                tested_jitter = [
                    max(0.01, self.data_new_tested[i] + np.random.normal(0, np.sqrt(self.data_new_tested[i]))) for i in
                    range(len(self.data_new_tested))]
                dead_jitter = [max(0.01, self.data_new_dead[i] + np.random.normal(0, np.sqrt(self.data_new_dead[i])))
                               for i in
                               range(len(self.data_new_dead))]

                # here is where we select the all-data parameters as our starting point
                starting_point_as_list = [all_data_params[key] for key in self.sorted_names]

                # NB: define the model constraints (mainly, positive values)
                # This is the old version in which it still attempts to fit exactly on jittered data
                params_as_dict = self.fit_curve_exactly_with_jitter(starting_point_as_list,
                                                            data_tested=tested_jitter,
                                                            data_dead=dead_jitter,
                                                            tested_indices=cases_bootstrap_indices,
                                                            deaths_indices=deaths_bootstrap_indices
                                                            )

                # This is the new version which just caps the likelihood
                # params_as_dict = self.fit_curve_via_likelihood(starting_point_as_list,
                #                                        tested_indices=cases_bootstrap_indices,
                #                                        deaths_indices=deaths_bootstrap_indices)
                # print(params_as_dict)
                
                sol = self.run_simulation(params_as_dict)
                bootstrap_sols.append(sol)
                bootstrap_params.append(params_as_dict)

            print(f'saving bootstraps to {self.bootstrap_filename}...')
            joblib.dump({'bootstrap_sols': bootstrap_sols, 'bootstrap_params': bootstrap_params,
                         'all_data_params': all_data_params, 'all_data_sol': all_data_sol},
                        self.bootstrap_filename)
            print('...done!')

        self.bootstrap_sols = bootstrap_sols
        self.bootstrap_params = bootstrap_params
        self.all_data_params = all_data_params
        self.all_data_sol = all_data_sol

        bootstrap_weights = [1] * len(self.bootstrap_params)
        for bootstrap_ind in range(len(self.bootstrap_params)):
            for param_name, (lower, upper) in self.priors.items():
                if param_name in self.static_params:
                    continue
                if lower is not None and self.bootstrap_params[bootstrap_ind][param_name] < lower:
                    bootstrap_weights[bootstrap_ind] = 0
                if upper is not None and self.bootstrap_params[bootstrap_ind][param_name] > upper:
                    bootstrap_weights[bootstrap_ind] = 0

        self.bootstrap_weights = bootstrap_weights

    def render_likelihood_samples(self,
                                  n_samples=None,
                                  bounds_to_use_str='medium',  # medium / loose / tight
                                  ):
        '''
        Obtain likelihood samples
        :param n_samples: how many samples to obtain
        :param bounds_to_use_str: string for which bounds to use
        :return: 
        '''

        if not path.exists(self.likelihood_samples_filename_format_str.format(bounds_to_use_str)):

            if bounds_to_use_str == 'tight':
                print('Using tight bounds for sampling')
                bounds_to_use = self.tight_bounds
            if bounds_to_use_str == 'medium':
                print('Using medium bounds for sampling')
                bounds_to_use = self.medium_bounds
            if bounds_to_use_str == 'loose':
                print('Using loose bounds for sampling')
                bounds_to_use = self.loosebounds

            if n_samples is None:
                n_samples = self.n_likelihood_samples

            logp = partial(self.get_log_likelihood,
                           opt_return_sol=False,
                           cases_bootstrap_indices=self.cases_indices,
                           deaths_bootstrap_indices=self.deaths_indices)

            all_samples = list()
            all_vals = list()
            print('\n----\nRendering likelihood samples...\n----')
            for _ in tqdm(range(n_samples)):
                indiv_sample_dict = dict()
                for param_name in self.sorted_names:
                    indiv_sample_dict[param_name] = \
                        np.random.uniform(bounds_to_use[param_name][0], bounds_to_use[param_name][1], 1)[0]
                all_samples.append(indiv_sample_dict)
                all_vals.append(
                    logp(indiv_sample_dict))  # this is the part that takes a while? Still surprised this takes so long

            all_samples_as_list = list()
            all_vals_as_list = list()
            for i in tqdm(range(n_samples)):
                if not np.isfinite(all_vals[i]):
                    continue
                else:
                    sample_as_list = np.array([float(all_samples[i][name]) for name in self.map_name_to_sorted_ind])
                    all_samples_as_list.append(sample_as_list)
                    all_vals_as_list.append(all_vals[i])

            print(f'saving samples to {self.likelihood_samples_filename_format_str.format(bounds_to_use_str)}...')
            joblib.dump({'all_samples_as_list': all_samples_as_list, 'all_vals_as_list': all_vals_as_list},
                        self.likelihood_samples_filename_format_str.format(bounds_to_use_str))
            print('...done!')

        else:

            print(f'\n----\nLoading likelihood samples from {self.likelihood_samples_filename_format_str}...\n----')
            samples_dict = joblib.load(self.likelihood_samples_filename_format_str.format(bounds_to_use_str))
            all_samples_as_list = samples_dict['all_samples_as_list']
            all_vals_as_list = samples_dict['all_vals_as_list']

        self.all_samples_as_list = all_samples_as_list
        self.all_vals_as_list = all_vals_as_list

    def fit_GMM_to_likelihood(self,
                              cov_type='diag',  # diag or full
                              n_components=1
                              ):
        '''
        Once you've run render_likelihood_samples, you can fit a Gaussian mixture model (GMM) to it!
        :param cov_type: what type of covariance matrix should we use? diag or diagonal and full for full
        :param n_components: how many components? Please leave this to one
        :return: None, it adds attributes to the object
        '''

        all_samples_as_array = np.vstack(self.all_samples_as_list).T
        n_samples_from_vals = int(1e5)
        normalized_probability = [np.exp(x) for x in self.all_vals_as_list]
        normalized_probability /= np.sum(normalized_probability)

        sorted_val_inds = np.argsort(self.all_vals_as_list)

        random_sampling = np.random.choice(len(self.all_samples_as_list), size=n_samples_from_vals,
                                           p=normalized_probability)
        random_sampling = [self.all_samples_as_list[i] for i in random_sampling]
        model = GaussianMixture(n_components=n_components, covariance_type=cov_type, reg_covar=1e-6)
        model.fit(random_sampling)

        conf_int = dict()
        std_devs = [np.sqrt(x) for x in model.covariances_[0]]
        for i, std_dev in enumerate(std_devs):
            mu = model.means_[0][i]
            lower = float(mu - std_dev * 1.645)
            upper = float(mu + std_dev * 1.645)
            conf_int[self.sorted_names[i]] = (lower, upper)
            print(f'Param {self.sorted_names[i]} 90% conf. int.: ({lower:.4g}, {upper:.4g})')
        if cov_type == 'full':
            plt.imshow(model.covariances_[0], cmap='coolwarm')
            plt.colorbar()
            plt.savefig(path.join(self.plot_filename_base, 'GMM_covariance_matrix.png'))

        predicted_vals = model.score_samples(all_samples_as_array.T)
        plt.clf()
        plt.plot([predicted_vals[i] for i in sorted_val_inds], [self.all_vals_as_list[i] for i in sorted_val_inds], '.',
                 alpha=100 / self.n_likelihood_samples)
        plt.xlabel('predicted values')
        plt.ylabel('actual values')
        plt.savefig(path.join(self.plot_filename_base, 'GMM_actual_vs_predicted_vals.png'))

        self.GMM_model = model
        self.GMM_confidence_intervals = conf_int
        self.GMM_means = {self.sorted_names[i]: x for i, x in enumerate(model.means_[0])}
        self.GMM_std_devs = std_devs

    def render_and_plot_cred_int(self):
        '''
        Use arviz to plot the credible intervals
        :return: None, just adds attributes to the object
        '''

        map_name_to_distro = dict()
        n_params = len(self.bootstrap_params[0])
        for param_name in self.sorted_names:
            param_distro = [self.bootstrap_params[i][param_name] for i in range(len(self.bootstrap_params))]
            map_name_to_distro[param_name] = np.array(param_distro)

        print('\n----\nBootstrap Means withOUT Priors Applied\n----')
        self.map_param_name_to_bootstrap_distro_without_prior = map_name_to_distro.copy()
        map_name_to_mean_without_prior = dict()
        for name in self.sorted_names:
            print(f'{name}: {map_name_to_distro[name].mean()}')
            map_name_to_mean_without_prior[name] = map_name_to_distro[name].mean()

        data = az.convert_to_inference_data(map_name_to_distro)
        az.plot_posterior(data, round_to=3, credible_interval=0.9, group='posterior',
                          var_names=self.sorted_param_names)  # show=True allows for plotting within IDE
        plt.savefig(path.join(self.plot_filename_base, 'param_distro_without_priors.png'))
        plt.close()

        cred_int_without_priors = dict()
        print('\n----\nHighest Probability Density Intervals withOUT Priors Applied\n----')
        for name in self.sorted_names:
            cred_int = tuple(az.hpd(data.posterior[name].T, credible_interval=0.9)[0])
            cred_int_without_priors[name] = cred_int
            print(f'Param {name} 90% HPD: ({cred_int[0]:.4g}, {cred_int[1]:.4g})')

        ####
        # Now apply priors
        ####

        print('\n----\nWhat % of bootstraps do we use after applying priors?')
        print(f'{sum(self.bootstrap_weights) / len(self.bootstrap_weights) * 100:.4g}%')
        print('----')

        map_name_to_mean_with_prior = None
        cred_int_with_priors = None
        if sum(self.bootstrap_weights) > 0:

            map_name_to_distro = dict()
            for param_name in self.sorted_names:
                param_distro = [self.bootstrap_params[i][param_name] for i in range(len(self.bootstrap_params)) if
                                self.bootstrap_weights[i] > 0.5]
                map_name_to_distro[param_name] = np.array(param_distro)

            print('\n----\nBootstrap Means with Priors Applied\n----')
            self.map_param_name_to_bootstrap_distro_with_prior = map_name_to_distro.copy()
            map_name_to_mean_with_prior = dict()
            for name in self.sorted_names:
                print(f'{name}: {map_name_to_distro[name].mean()}')
                map_name_to_mean_with_prior[name] = map_name_to_distro[name].mean()

            # if we want floating-point weights, have to find the common multiple and do discrete sampling
            # for example, this commented line doesn't change the HDP values, since it just duplicates all values
            # data = az.convert_to_inference_data({key: np.array(list(val) + list(val)) for key,val in map_name_to_distro.items()})

            data = az.convert_to_inference_data(map_name_to_distro)
            az.plot_posterior(data, round_to=3, credible_interval=0.9, group='posterior',
                              var_names=self.sorted_param_names)  # show=True allows for plotting within IDE
            plt.savefig(path.join(self.plot_filename_base, 'param_distro_with_priors.png'))
            plt.close()

            cred_int_with_priors = dict()
            print('\n----\nHighest Probability Density Intervals with Priors Applied\n----')
            for name in self.sorted_names:
                cred_int = tuple(az.hpd(data.posterior[name].T, credible_interval=0.9)[0])
                cred_int_with_priors[name] = cred_int
                print(f'Param {name} 90% HPD: ({cred_int[0]:.4g}, {cred_int[1]:.4g})')

        self.bootstrap_cred_int_without_priors = cred_int_without_priors
        self.bootstrap_cred_int_with_priors = cred_int_with_priors
        self.bootstrap_means_without_priors = map_name_to_mean_without_prior
        self.bootstrap_means_with_priors = map_name_to_mean_with_prior

    def run_fits(self):
        '''
        Builder that goes through each method in its proper sequence
        :return: None
        '''

        # Sample plot just to check
        self.solve_and_plot_solution(title='Test Plot with Default Parameters',
                                     plot_filename_filename='test_plot.png')

        # Training Data Bootstraps
        self.render_bootstraps()

        # Plot example solutions from bootstrap
        bootstrap_selection = np.random.choice(self.bootstrap_params)
        self.solve_and_plot_solution(in_params=bootstrap_selection,
                                     title='Random Bootstrap Selection',
                                     plot_filename_filename='random_bootstrap_selection.png')

        # Plot all-data solution 
        self.solve_and_plot_solution(in_params=self.all_data_params,
                                     title='All-Data Solution',
                                     plot_filename_filename='all_data_solution.png')

        # Plot all bootstraps
        self.plot_all_solutions()

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int()

        # Fit GMM to likelihood -- Start with prepping data for fit
        # state_model.n_likelihood_samples = 1000
        self.render_likelihood_samples()

        # Next define GMM model on likelihood and fit
        self.fit_GMM_to_likelihood()

    def pretty_print_params(self, in_dict):
        '''
        Helper function for printing our parameter values and bounds consistently
        :param in_dict: dictionary of parameters to pretty print
        :return: None, just prints
        '''
        if in_dict is None:
            print('None')
        else:
            for name in self.sorted_names:
                val = in_dict[name]
                if type(val) == tuple and len(val) == 2:
                    val_str = f'({val[0]:.4g}, {val[1]:.4g})'
                else:
                    val_str = f'{val:.4g}'
                print(f'{name}: {val_str}')
