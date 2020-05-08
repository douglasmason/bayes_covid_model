import numpy as np
import pandas as pd
# from sampyl import np as sampyl_np # for autograd
# from jax import numpy as np
# from jax.experimental.ode import odeint as odeint
from nuts_local import nuts6, NutsSampler_fn_wrapper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy as sp
import joblib
from os import path
from functools import partial
from tqdm import tqdm
import os
from scipy.optimize import approx_fprime
from functools import lru_cache, partial

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KernelDensity

plt.style.use('seaborn-darkgrid')
matplotlib.use('Agg')

from scipy.integrate import odeint
from time import time as get_time
import datetime
import matplotlib.dates as mdates
import arviz as az
import seaborn as sns


class Stopwatch:

    def __init__(self):
        self.time0 = get_time()

    def elapsed_time(self):
        return get_time() - self.time0

    def reset(self):
        self.time0 = get_time()


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
                 priors=None,
                 test_params=None,
                 static_params=None,
                 logarithmic_params=None,
                 extra_params=None,
                 plot_dpi=300,
                 opt_force_plot=True,
                 opt_calc=True):

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
        self.likelihood_samples_from_bootstraps_filename = path.join('state_likelihood_samples',
                                                                     f"{state_name.lower().replace(' ', '_')}_{n_bootstraps}_bootstraps_likelihoods_max_date_{max_date_str.replace('-', '_')}.joblib")
        self.plot_filename_base = path.join('state_plots',
                                            f"{state_name.lower().replace(' ', '_')}_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_opt_walk_True_max_date_{max_date_str.replace('-', '_')}")

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

        self.t_vals = np.linspace(-burn_in, 120, burn_in + 120 + 1)

        # misnomer here
        try:
            self.day_of_threshold_met_case = [i for i, x in enumerate(self.series_data[:, 1]) if x >= 20][0]
        except:
            self.day_of_threshold_met_case = len(self.series_data) - 1
        try:
            self.day_of_threshold_met_death = [i for i, x in enumerate(self.series_data[:, 2]) if x >= 20][0]
        except:
            self.day_of_threshold_met_death = len(self.series_data) - 1

        data_cum_tested = self.series_data[:, 1].copy()
        self.data_new_tested = [data_cum_tested[0]] + [data_cum_tested[i] - data_cum_tested[i - 1] for i in
                                                       range(1, len(data_cum_tested))]

        data_cum_dead = self.series_data[:, 2].copy()
        self.data_new_dead = [data_cum_dead[0]] + [data_cum_dead[i] - data_cum_dead[i - 1] for i in
                                                   range(1, len(data_cum_dead))]

        delta_t = 17
        self.data_new_recovered = [0.0] * delta_t + self.data_new_tested

        cases_indices = list(range(self.day_of_threshold_met_case, len(self.series_data)))
        deaths_indices = list(range(self.day_of_threshold_met_death, len(self.series_data)))
        self.cases_indices = [i for i in cases_indices if self.data_new_tested[i] > 0]
        self.deaths_indices = [i for i in deaths_indices if self.data_new_dead[i] > 0]

        self.curve_fit_bounds = curve_fit_bounds
        self.priors = priors
        self.test_params = test_params

        self.sorted_param_names = [name for name in sorted_param_names if name not in static_params]
        self.sorted_init_condit_names = sorted_init_condit_names
        self.sorted_names = self.sorted_init_condit_names + self.sorted_param_names
        self.logarithmic_params = logarithmic_params

        self.map_name_to_sorted_ind = {val: ind for ind, val in enumerate(self.sorted_names)}

        self.all_samples_as_list = list()
        self.all_log_probs_as_list = list()
        self.all_propensities_as_list = list()

        self.all_random_walk_samples_as_list = list()
        self.all_random_walk_log_probs_as_list = list()

        self.plot_dpi = plot_dpi
        self.opt_force_plot = opt_force_plot
        self.opt_calc = opt_calc

        self.loaded_bootstraps = False
        self.loaded_likelihood_samples = list()
        self.loaded_MCMC = list()
        
        self.extra_params = {key: partial(val, map_name_to_sorted_ind=self.map_name_to_sorted_ind) for key, val in extra_params.items()}

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

        # compile each component of the error separately, then concatenate
        revised_cases_indices = cases_bootstrap_indices  # and new_tested_from_sol[i + self.burn_in] > 0]
        revised_deaths_indices = deaths_bootstrap_indices  # and new_deceased_from_sol[i + self.burn_in] > 0]

        # revised_cases_indices = [i for i in cases_bootstrap_indices if data_new_tested[i] > 0 and new_tested_from_sol[i + self.burn_in] > 0]
        # revised_deaths_indices = [i for i in deaths_bootstrap_indices if
        #                          data_new_dead[i] > 0 and new_deceased_from_sol[i + self.burn_in] > 0]

        # rule out solutions with negative values
        # err_from_negative_sols = sum([new_tested_from_sol[i + self.burn_in] for i in revised_cases_indices if
        #                               new_tested_from_sol[i + self.burn_in] < 0])
        # err_from_negative_sols += sum([new_deceased_from_sol[i + self.burn_in] for i in revised_cases_indices if
        #                                new_deceased_from_sol[i + self.burn_in] < 0])
        # err_from_negative_sols *= -1

        val1 = params['contagious_to_positive_delay']
        val2 = params['contagious_to_deceased_delay']
        err_from_reversed_delays = val1 - val2 if val1 > val2 else 0

        # TODO: if you see errors here, then self.t_vals isn't long enough
        #       go to the __init__ method and adjust the maximum values
        new_tested_err = [
            (np.log(data_new_tested[i]) - np.log(new_tested_from_sol[i + self.burn_in]))
            for i in revised_cases_indices]
        new_dead_err = [
            (np.log(data_new_dead[i]) - np.log(new_deceased_from_sol[i + self.burn_in]))
            for i in revised_deaths_indices]

        final_err = new_tested_err + new_dead_err + [
            err_from_reversed_delays]  # + [err_from_negative_sols, err_from_reversed_delays]

        if opt_return_sol:
            return final_err, sol, None  # + [1, 1]
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
                                 in_params,
                                 tested_indices=None,
                                 deaths_indices=None,
                                 method='SLSQP'
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

        if type(in_params) == dict:
            p0 = [in_params[name] for name in self.sorted_names]
        else:
            p0 = in_params.copy()

        def get_neg_log_likelihood(p):
            return -self.get_log_likelihood(p,
                                            cases_bootstrap_indices=tested_indices,
                                            deaths_bootstrap_indices=deaths_indices
                                            )

        bounds_to_use = [self.curve_fit_bounds[name] for name in self.sorted_names]
        results = sp.optimize.minimize(get_neg_log_likelihood, p0, bounds=bounds_to_use, method=method)
        print(f'success? {results.success}')
        params_as_list = results.x
        params_as_dict = {key: params_as_list[i] for i, key in enumerate(self.sorted_names)}

        return params_as_dict

    def get_log_likelihood(self,
                           in_params,
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

        # convert from list to dictionary (for compatibility with the least-sq solver
        if type(in_params) != dict and self.map_name_to_sorted_ind is not None:
            params = {key: in_params[ind] for key, ind in self.map_name_to_sorted_ind.items()}
        else:
            params = in_params.copy()

        if cases_bootstrap_indices is None:
            cases_bootstrap_indices = self.cases_indices
        if deaths_bootstrap_indices is None:
            deaths_bootstrap_indices = self.deaths_indices

        # timer = Stopwatch()
        sol = self.run_simulation(params)
        new_tested_from_sol = sol[1]
        new_deceased_from_sol = sol[2]
        # print(f'Simulation took {timer.elapsed_time() * 100} ms')
        # timer = Stopwatch()

        new_tested_dists = [ \
            (np.log(self.data_new_tested[i]) - np.log(new_tested_from_sol[i + self.burn_in]))
            for i in cases_bootstrap_indices]
        new_dead_dists = [ \
            (np.log(self.data_new_dead[i]) - np.log(new_deceased_from_sol[i + self.burn_in]))
            for i in deaths_bootstrap_indices]

        dists = new_tested_dists + new_dead_dists
        return_val = sum(-x ** 2 for x in dists)

        # ensure the two delays are physical
        val1 = params['contagious_to_positive_delay']
        val2 = params['contagious_to_deceased_delay']
        err_from_reversed_delays = val1 - val2 if val1 > val2 else 0
        return_val += -err_from_reversed_delays ** 2

        # print(f'Other stuff took {timer.elapsed_time()} ms')

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

        timer = Stopwatch()
        for i in range(100):
            # note the extra comma after the args list is to ensure we pass the whole shebang as a tuple otherwise errors!
            sol = self.run_simulation(params)
        print(f'time to simulate (ms): {(timer.elapsed_time()) / 100 * 1000}')

        new_positive = sol[1]
        new_deceased = sol[2]

        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + 14 + self.burn_in)
        data_plot_date_range = [self.min_date + datetime.timedelta(days=1) * i for i in range(len(self.series_data))]
        sol_plot_date_range = [self.min_date - datetime.timedelta(days=self.burn_in) + datetime.timedelta(days=1) * i
                               for i in
                               range(len(sol[0]))][min_plot_pt:max_plot_pt]

        full_output_filename = path.join(self.plot_filename_base, plot_filename_filename)
        if not path.exists(full_output_filename) or self.opt_force_plot:
            plt.clf()
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
            plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_threshold_met_case - 10), None))
            plt.legend()
            if title is not None:
                plt.title(title)
            plt.savefig(full_output_filename, dpi=self.plot_dpi)
    
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

        self._plot_all_solutions_sub(sols_to_plot, plot_filename_filename='bootstrap_solutions.png')

        # In the future we can do float weights, for now it's just binary ones
        sols_to_plot = [i for i, sol in enumerate(self.bootstrap_sols) if round(self.bootstrap_weights[i])]

        self._plot_all_solutions_sub(sols_to_plot, plot_filename_filename='bootstrap_solutions_with_priors.png')

    def _plot_all_solutions_sub(self,
                                sols_to_plot,
                                plot_filename_filename='bootstrap_solutions.png'):
        '''
        Helper function to plot_all_solutions
        :param n_sols_to_plot: how many simulations should we sample for the plot?
        :param plot_filename_filename: string to add to the plot filename
        :return: None
        '''
        full_output_filename = path.join(self.plot_filename_base, plot_filename_filename)
        if path.exists(full_output_filename) and not self.opt_force_plot:
            return        
        
        print('Printing...', path.join(self.plot_filename_base, plot_filename_filename))
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
        plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_threshold_met_case - 10), None))
        plt.ylim((1, max(self.data_new_tested) * 100))
        plt.legend()
        # plt.title(f'{state} Data (points) and Model Predictions (lines)')
        plt.savefig(full_output_filename, dpi=self.plot_dpi)

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
        success = False

        try:
            bootstrap_dict = joblib.load(self.bootstrap_filename)
            bootstrap_sols = bootstrap_dict['bootstrap_sols']
            bootstrap_params = bootstrap_dict['bootstrap_params']
            all_data_params = bootstrap_dict['all_data_params']
            all_data_sol = bootstrap_dict['all_data_sol']
            success = True
            self.loaded_bootstraps = True or self.opt_force_plot
        except:
            self.loaded_bootstraps = False or self.opt_force_plot

        if not success and self.opt_calc:

            print('\n----\nRendering bootstrap model fits... starting with the all-data one...\n----')

            test_params_as_list = [self.test_params[key] for key in self.sorted_names]
            all_data_params = self.fit_curve_exactly_with_jitter(test_params_as_list)  # fit_curve_via_likelihood
            all_data_sol = self.run_simulation(all_data_params)

            print('\nParameters when trained on all data (this is our starting point for optimization):')
            [print(f'{key}: {val:.4g}') for key, val in all_data_params.items()]

            methods = [
                'Nelder-Mead',  # ok results, but claims it failed
                # 'Powell', #warnings
                # 'CG', #warnings
                # 'BFGS',
                # 'Newton-CG', #can't do bounds
                # 'L-BFGS-B', #bad results
                # 'TNC', #ok, but still weird results
                # 'COBYLA', #bad results
                'SLSQP',  # ok, good results, and says it succeeded
                # 'trust-constr', #warnings
                # 'dogleg', #can't do bounds
                # 'trust-ncg', #can't do bounds
                # 'trust-exact',
                # 'trust-krylov'
            ]

            for method in methods:
                print('trying method', method)
                try:
                    test_params_as_list = [self.test_params[key] for key in self.sorted_names]
                    all_data_params2 = self.fit_curve_via_likelihood(test_params_as_list,
                                                                     method=method)  # fit_curve_via_likelihood
                    all_data_sol2 = self.run_simulation(all_data_params)

                    print('\nParameters when trained on all data (this is our starting point for optimization):')
                    [print(f'{key}: {val:.4g}') for key, val in all_data_params2.items()]
                except:
                    print(f'method {method} failed!')

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

                sol = self.run_simulation(params_as_dict)
                bootstrap_sols.append(sol)
                bootstrap_params.append(params_as_dict)

            print(f'saving bootstraps to {self.bootstrap_filename}...')
            joblib.dump({'bootstrap_sols': bootstrap_sols, 'bootstrap_params': bootstrap_params,
                         'all_data_params': all_data_params, 'all_data_sol': all_data_sol},
                        self.bootstrap_filename)
            print('...done!')

        print('\nParameters when trained on all data (this is our starting point for optimization):')
        [print(f'{key}: {val:.4g}') for key, val in all_data_params.items()]

        # Add deterministic parameters to bootstraps
        for params in bootstrap_params:
            for extra_param, extra_param_func in self.extra_params.items():
                params[extra_param] = extra_param_func([params[name] for name in self.sorted_names])

        # Add deterministic parameters to all-data solution
        for extra_param, extra_param_func in self.extra_params.items():
            all_data_params[extra_param] = extra_param_func([all_data_params[name] for name in self.sorted_names])

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
                                  n_samples=None
                                  ):
        '''
        Obtain likelihood samples
        :param n_samples: how many samples to obtain
        :param bounds_to_use_str: string for which bounds to use
        :return: None, saves results to object attributes
        '''

        success = False

        try:
            print(
                f'\n----\nLoading likelihood samples from {self.likelihood_samples_filename_format_str.format(bounds_to_use_str)}...\n----')
            samples_dict = joblib.load(self.likelihood_samples_filename_format_str.format(bounds_to_use_str))
            print('...done!')
            all_samples_as_list = samples_dict['all_samples_as_list']
            all_log_probs_as_list = samples_dict['all_log_probs_as_list']
            all_propensities_as_list = samples_dict['all_propensities_as_list']
            success = True
            self.loaded_likelihood_samples.append(bounds_to_use_str)
        except:
            pass

        if not success and self.opt_calc:

            bounds_to_use = self.curve_fit_bounds

            if n_samples is None:
                n_samples = self.n_likelihood_samples

            all_samples = list()
            all_log_probs = list()
            volume = np.prod([x[1] - x[0] for x in bounds_to_use.values()])
            print('\n----\nRendering likelihood samples...\n----')
            for _ in tqdm(range(n_samples)):
                indiv_sample_dict = dict()
                for param_name in self.sorted_names:
                    indiv_sample_dict[param_name] = \
                        np.random.uniform(bounds_to_use[param_name][0], bounds_to_use[param_name][1], 1)[0]
                all_samples.append(indiv_sample_dict)
                all_log_probs.append(
                    self.get_log_likelihood(
                        indiv_sample_dict))  # this is the part that takes a while? Still surprised this takes so long

            all_samples_as_list = list()
            all_log_probs_as_list = list()
            for i in tqdm(range(n_samples)):
                if not np.isfinite(all_log_probs[i]):
                    continue
                else:
                    sample_as_list = np.array([float(all_samples[i][name]) for name in self.map_name_to_sorted_ind])
                    all_samples_as_list.append(sample_as_list)
                    all_log_probs_as_list.append(all_log_probs[i])

            all_propensities_as_list = [len(all_samples_as_list) / volume] * len(all_samples_as_list)
            print(f'saving samples to {self.likelihood_samples_filename_format_str.format("medium")}...')
            joblib.dump({'all_samples_as_list': all_samples_as_list,
                         'all_log_probs_as_list': all_log_probs_as_list,
                         'all_propensities_as_list': all_propensities_as_list
                         },
                        self.likelihood_samples_filename_format_str.format("medium"))
            print('...done!')

        self.random_likelihood_samples = all_samples_as_list
        self.random_likelihood_vals = all_log_probs_as_list
        self.random_likelihood_propensities = all_propensities_as_list
        self._add_samples(all_samples_as_list, all_log_probs_as_list, all_propensities_as_list)

        # we add the results from our bootstrap fits to fill in unknown regions in parameter space
        # if not path.exists(self.likelihood_samples_from_bootstraps_filename):
        #     # add in bootstrap fits as additional samples (huh, nice connection to synthetic data techniques...)
        #     print('Adding bootstrap fitted params to likelihood samples...')
        #     bootstrap_samples_as_list = list()
        #     bootstrap_vals_as_list = list()
        #     for sample in tqdm(self.bootstrap_params):
        #         val = logp(sample)
        #         sample_as_list = [float(sample[name]) for name in self.map_name_to_sorted_ind]
        #         bootstrap_samples_as_list.append(sample_as_list)
        #         bootstrap_vals_as_list.append(val)
        #     print(f'Saving to {self.likelihood_samples_from_bootstraps_filename}...')
        #     joblib.dump({'bootstrap_samples_as_list': bootstrap_samples_as_list,
        #                  'bootstrap_vals_as_list': bootstrap_vals_as_list},
        #                 self.likelihood_samples_from_bootstraps_filename)
        # else:
        #     print(
        #         f'\n----\nLoading likelihood samples from {self.likelihood_samples_from_bootstraps_filename}...\n----')
        #     tmp_dict = joblib.load(self.likelihood_samples_from_bootstraps_filename)
        #     print('...done!')
        #     bootstrap_samples_as_list = tmp_dict['bootstrap_samples_as_list']
        #     bootstrap_vals_as_list = tmp_dict['bootstrap_vals_as_list']
        # 
        # self.bootstrap_vals = bootstrap_vals_as_list
        # self._add_samples(bootstrap_samples_as_list, bootstrap_vals_as_list)

    def _add_samples(self, samples, vals, propensities, opt_walk=False):

        print('adding samples...')
        print('checking sizes...')

        if not opt_walk:
            print(f'samples: {len(samples)}, vals: {len(vals)}, propensities: {len(propensities)}')
            self.all_samples_as_list += samples
            self.all_log_probs_as_list += vals
            self.all_propensities_as_list += propensities

            shuffled_ind = list(range(len(self.all_samples_as_list)))
            np.random.shuffle(shuffled_ind)
            self.all_samples_as_list = [self.all_samples_as_list[i] for i in shuffled_ind]
            self.all_log_probs_as_list = [self.all_log_probs_as_list[i] for i in shuffled_ind]
            self.all_propensities_as_list = [self.all_propensities_as_list[i] for i in shuffled_ind]
            print('...done!')
        else:
            print(f'samples: {len(samples)}, vals: {len(vals)}, propensities: {len(propensities)}')
            self.all_random_walk_samples_as_list += samples
            self.all_random_walk_log_probs_as_list += vals

            shuffled_ind = list(range(len(self.all_random_walk_samples_as_list)))
            np.random.shuffle(shuffled_ind)
            self.all_random_walk_samples_as_list = [self.all_random_walk_samples_as_list[i] for i in shuffled_ind]
            self.all_random_walk_log_probs_as_list = [self.all_random_walk_log_probs_as_list[i] for i in shuffled_ind]
            print('...done!')


    def MCMC(self, p0, opt_walk=True,
             bounds_range_to_sigma_denom=None,
             retry_after_n_seconds=10):

        n_samples = self.n_likelihood_samples
        if opt_walk:
            MCMC_burn_in_frac = 0.2
        else:
            MCMC_burn_in_frac = 0

        # take smaller steps for a random walk, larger ones for a gaussian sampling
        if bounds_range_to_sigma_denom is None:
            if opt_walk:
                bounds_range_to_sigma_denom = 100
            else:
                bounds_range_to_sigma_denom = 10
        use_bounds_range_to_sigma_denom = bounds_range_to_sigma_denom

        bounds_to_use = self.curve_fit_bounds
        
        def get_bunched_up_on_bounds(input_params, new_val=None):
            if new_val is None:
                new_val = np.array([input_params[name] for name in self.sorted_names])
            bunched_up_on_bounds = input_params.copy()
            offending_params = list()
            for param_ind, param_name in enumerate(self.sorted_names):
                lower, upper = bounds_to_use[param_name]
                if lower is not None and new_val[param_ind] < lower:
                    # print(f'Failed on {param_name} lower: {new_val[param_ind]} < {lower}')
                    bunched_up_on_bounds[param_name] = lower
                    offending_params.append((param_name, 'lower'))
                if upper is not None and new_val[param_ind] > upper:
                    # print(f'Failed on {param_name} upper: {new_val[param_ind]} > {upper}')
                    bunched_up_on_bounds[param_name] = upper
                    offending_params.append((param_name, 'upper'))
                    
            return bunched_up_on_bounds, offending_params
        
        # check that initial conditions are valid
        _, offending_params = get_bunched_up_on_bounds(p0)
        if len(offending_params) > 0:
            print('Starting point outside of bounds, MCMC won\'t work!')
            return
        
        @lru_cache(maxsize=10)
        def get_propensity_model(bounds_range_to_sigma_denom):
            sigma = {key: (val[1] - val[0]) / bounds_range_to_sigma_denom for key, val in bounds_to_use.items()}

            # overwrite sigma for values that are strictly positive multipliers of unknown scale
            for param_name in self.logarithmic_params:
                sigma[param_name] = 1 / bounds_range_to_sigma_denom

            sigma_as_list = [sigma[name] for name in self.sorted_names]
            cov = np.diag([max(1e-8, x ** 2) for x in sigma_as_list])
            propensity_model = sp.stats.multivariate_normal(cov=cov)
            return propensity_model

        def acquisition_function(input_params, bounds_range_to_sigma_denom):
            output_params = input_params
            jitter_propensity = 1
            accepted = False
            n_attempts = 0
            propensity_model = get_propensity_model(bounds_range_to_sigma_denom)
            while n_attempts < 100 and not accepted:  # this limits endless searches with wide sigmas
                n_attempts += 1

                # acq_timer=Stopwatch()
                jitter = propensity_model.rvs()
                # sample_time = acq_timer.elapsed_time()
                # print(f'Sample time: {sample_time * 1000:.4g} ms')
                # acq_timer.reset()
                jitter_propensity = propensity_model.pdf(jitter)
                # pdf_time = acq_timer.elapsed_time()
                # print(f'PDF time: {pdf_time * 1000:.4g} ms')
                # print(f'n_attempts: {n_attempts}')
                for param_name in self.logarithmic_params:
                    jitter[self.map_name_to_sorted_ind[param_name]] = jitter[self.map_name_to_sorted_ind[param_name]] * \
                                                                      p0[param_name]

                new_val = np.array([input_params[name] for name in self.sorted_names])
                new_val += jitter
                bunched_up_on_bounds, offending_params = get_bunched_up_on_bounds(input_params, new_val=new_val)
                if len(offending_params) == 0:
                    output_params = {self.sorted_names[i]: new_val[i] for i in range(len(new_val))}
                    accepted = True

            if not accepted:
                print('Warning: Setting next sample point bunched up on the bounds...')
                print('Offending parameters: ', offending_params)
                output_params = bunched_up_on_bounds
                jitter_propensity = 1

            return output_params, jitter_propensity

        # update user on the starting pt
        prev_p = p0.copy()
        print('Starting from...')
        self.pretty_print_params(p0)

        ll = self.get_log_likelihood(p0)
        samples = list()
        log_probs = list()
        propensities = list()

        if opt_walk:
            filename_str = f'MCMC_bounds_range_to_sigma_denom_{int(bounds_range_to_sigma_denom)}'
        else:
            filename_str = f'MCMC_fixed_bounds_range_to_sigma_denom_{int(bounds_range_to_sigma_denom)}'

        success = False

        try:
            print(f'loading from {self.likelihood_samples_filename_format_str.format(filename_str)}...')
            tmp_dict = joblib.load(self.likelihood_samples_filename_format_str.format(filename_str))
            samples = tmp_dict['samples']
            log_probs = tmp_dict['vals']
            propensities = tmp_dict['propensities']
            success = True
            print('...done!')
            self.loaded_MCMC.append({'opt_walk': opt_walk, 'bounds_range_to_sigma_denom': bounds_range_to_sigma_denom})
        except:
            print('...load failed!... doing calculations...')

        timer = Stopwatch()
        prev_test_ind = -1
        n_accepted = 0
        n_accepted_turn = 0
        if not success and self.opt_calc:

            for test_ind in tqdm(range(n_samples)):

                # sub_timer = Stopwatch()
                proposed_p, proposed_propensity = acquisition_function(prev_p, use_bounds_range_to_sigma_denom)
                # print(f'acquisition time {sub_timer.elapsed_time() * 1000} ms')
                # sub_timer = Stopwatch
                proposed_ll = self.get_log_likelihood(proposed_p)
                # print(f'likelihood time {sub_timer.elapsed_time()} ms')

                if opt_walk:

                    acceptance_ratio = np.exp(proposed_ll - ll)
                    rand_num = np.random.uniform()

                    if timer.elapsed_time() > 3:
                        n_test_ind_turn = test_ind - prev_test_ind
                        print(f'\n NO acceptances in past two seconds! Diagnose:'
                              f'\n how many samples accepted overall? {n_accepted} ({n_accepted / test_ind * 100:.4g}%)' + \
                              f'\n how many samples accepted since last update? {n_accepted_turn} ({n_accepted_turn / n_test_ind_turn * 100:.4g}%)' + \
                              f'\n prev. log likelihood: {ll:.4g}' + \
                              f'\n       log likelihood: {proposed_ll:.4g}' + \
                              f'\n use_bounds_range_to_sigma_denom {use_bounds_range_to_sigma_denom}' + \
                              f'\n acceptance ratio: {acceptance_ratio:.4g}' + \
                              ''.join(f'\n  {key}: {proposed_p[key]:.4g}' for key in self.sorted_names))
                        timer.reset()
                        if opt_walk:
                            use_bounds_range_to_sigma_denom = bounds_range_to_sigma_denom * 100

                    if rand_num <= acceptance_ratio and np.isfinite(acceptance_ratio):

                        # it's important to do this as soon as we get a good fit
                        if opt_walk:
                            use_bounds_range_to_sigma_denom = bounds_range_to_sigma_denom

                        n_accepted += 1
                        n_accepted_turn += 1
                        if timer.elapsed_time() > 1:
                            n_test_ind_turn = test_ind - prev_test_ind
                            print(
                                f'\n how many samples accepted overall? {n_accepted} of {test_ind} ({n_accepted / test_ind * 100:.4g}%)' + \
                                f'\n how many samples accepted since last update? {n_accepted_turn} of {n_test_ind_turn} ({n_accepted_turn / n_test_ind_turn * 100:.4g}%)' + \
                                f'\n prev. log likelihood: {ll:.4g}' + \
                                f'\n       log likelihood: {proposed_ll:.4g}' + \
                                f'\n use_bounds_range_to_sigma_denom {use_bounds_range_to_sigma_denom}' + \
                                f'\n acceptance ratio: {acceptance_ratio:.4g}' + \
                                ''.join(f'\n  {key}: {proposed_p[key]:.4g}' for key in self.sorted_names))
                            timer.reset()
                            prev_test_ind = test_ind
                            n_accepted_turn = 0

                        prev_p = proposed_p.copy()
                        ll = proposed_ll
                        samples.append(proposed_p)
                        log_probs.append(proposed_ll)
                        propensities.append(1)

                else:

                    samples.append(proposed_p)
                    log_probs.append(proposed_ll)
                    propensities.append(proposed_propensity)

            propensities = [x * len(samples) for x in propensities]
            joblib.dump({'samples': samples, 'vals': log_probs, 'propensities': propensities},
                        self.likelihood_samples_filename_format_str.format(filename_str))

        samples_as_list = [[sample[key] for key in self.sorted_names] for sample in samples]
        MCMC_burn_in = int(MCMC_burn_in_frac * len(samples_as_list))
        self._add_samples(samples_as_list[MCMC_burn_in:], log_probs[MCMC_burn_in:], propensities[MCMC_burn_in:],
                          opt_walk=opt_walk)

    def fit_MVN_to_likelihood(self,
                              cov_type='full',
                              opt_walk=False,
                              opt_plot=True):

        if opt_walk:
            print('Since this is a random walk, ignoring weights when fitting NVM...')
            weight_sampled_params = self.all_random_walk_samples_as_list
            params = self.all_random_walk_samples_as_list
            log_probs = self.all_random_walk_log_probs_as_list
            weights = [1] * len(params)
            filename_str = 'MVN_random_walk'
        else:
            weight_sampled_params, params, weights, log_probs = self._get_weighted_samples()
            filename_str = 'MVN_samples'

        # If there are no weights, then just abort
        if sum(weights) == 0:
            return

        means = dict()
        std_devs = dict()
        for param_name in self.sorted_names:
            param_ind = self.map_name_to_sorted_ind[param_name]
            locs = [params[i][param_ind] for i in range(len(params))]
            means[param_name] = np.average(locs, weights=weights)
            std_devs[param_name] = np.sqrt(np.average((locs - means[param_name]) ** 2, weights=weights))

        means_as_list = [means[name] for name in self.sorted_names]
        std_devs_as_list = [std_devs[name] for name in self.sorted_names]
        print('means:', means_as_list)
        print('std_devs:', std_devs_as_list)

        self.solve_and_plot_solution(in_params=means,
                                     title=f'Mean of {filename_str.replace("_", " ")} Fit Solution',
                                     plot_filename_filename=f'mean_of_{filename_str}_solution.png')

        if sum(std_devs_as_list) < 1e-6:
            print('No variance in our samples... skipping MVN calcs!')

            if opt_walk:
                self.MVN_random_walk_model = None
                self.MVN_random_walk_confidence_intervals = None
                self.MVN_random_walk_means = means
                self.MVN_random_walk_std_devs = std_devs
                self.MVN_random_walk_cov = None
                self.MVN_random_walk_corr = None
            else:
                self.MVN_sample_model = None
                self.MVN_sample_confidence_intervals = None
                self.MVN_sample_means = means
                self.MVN_sample_std_devs = std_devs
                self.MVN_sample_cov = None
                self.MVN_sample_corr = None

            return

        if cov_type == 'diag':
            cov = np.diag([x ** 2 for x in std_devs_as_list])
        else:
            cov = np.cov(np.vstack(params).T,
                         aweights=weights)

        # convert covariance to correlation matrix
        std_devs_mat = np.diag(np.sqrt(np.diagonal(cov)))
        cov_inv = np.linalg.inv(std_devs_mat)
        corr = cov_inv @ cov @ cov_inv

        print('cov:', cov)
        model = sp.stats.multivariate_normal(mean=means_as_list, cov=cov, allow_singular=True)

        conf_ints = dict()
        for param_name, std_dev in std_devs.items():
            mu = means[param_name]
            lower = float(mu - std_dev * 1.645)
            upper = float(mu + std_dev * 1.645)
            conf_ints[param_name] = (lower, upper)
            print(f'Param {param_name} mean and 90% conf. int.: {mu:.4g} ({lower:.4g}, {upper:.4g})')

        predicted_vals = model.pdf(np.vstack(params))

        full_output_filename = path.join(self.plot_filename_base, f'{filename_str}_correlation_matrix.png')
        if not path.exists(full_output_filename) or self.opt_force_plot:
            
            plt.clf()
            ax = sns.heatmap(corr, xticklabels=self.sorted_names, yticklabels=self.sorted_names, cmap='coolwarm',
                             center=0.0)
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                horizontalalignment='right'
            )
            plt.savefig(full_output_filename, dpi=self.plot_dpi)

        full_output_filename = path.join(self.plot_filename_base, f'{filename_str}_actual_vs_predicted_vals.png')
        if not path.exists(full_output_filename) or self.opt_force_plot:
                
            plt.clf()
            plt.plot(np.exp(log_probs), predicted_vals, '.',
                     alpha=max(100 / len(predicted_vals), 0.01))
            plt.xlabel('actual values')
            plt.ylabel('predicted values')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig(full_output_filename, dpi=self.plot_dpi)
            
            # 
            # plt.clf()
            # plt.plot(weights, [np.exp(x) for x in predicted_vals], '.',
            #          alpha=100 / len(predicted_vals))
            # plt.xlabel('actual values')
            # plt.ylabel('predicted values')
            # plt.xlim((np.percentile(weights, 75), np.percentile(weights, 99)))
            # plt.ylim((np.percentile([np.exp(x) for x in predicted_vals], 75), np.percentile([np.exp(x) for x in predicted_vals], 99)))
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.savefig(path.join(self.plot_filename_base, 'MVN_actual_vs_predicted_vals_zoom.png'))

        if opt_walk:
            self.MVN_random_walk_model = model
            self.MVN_random_walk_confidence_intervals = conf_ints
            self.MVN_random_walk_means = means
            self.MVN_random_walk_std_devs = std_devs
            self.MVN_random_walk_cov = cov
            self.MVN_random_walk_corr = corr
        else:
            self.MVN_sample_model = model
            self.MVN_sample_confidence_intervals = conf_ints
            self.MVN_sample_means = means
            self.MVN_sample_std_devs = std_devs
            self.MVN_sample_cov = cov
            self.MVN_sample_corr = corr

    def _get_weighted_samples(self):

        time0 = get_time()
        valid_ind = [i for i, x in enumerate(self.all_log_probs_as_list) if np.isfinite(np.exp(x)) and \
                     np.isfinite(self.all_propensities_as_list[i]) and \
                     self.all_propensities_as_list[i] > 0]
        propensities = np.array(self.all_propensities_as_list)[valid_ind]
        print('filtered propensities')
        log_probs = np.array(self.all_log_probs_as_list)[valid_ind]
        print('filtered log_probs')
        params = np.array(self.all_samples_as_list)[valid_ind]
        print('filtered params')
        weights = np.array([np.exp(x) / propensities[i] for i, x in enumerate(log_probs)])
        weights /= sum(weights)
        print(f'max weight: {max(weights)}')
        max_weight_ind = np.argmax(weights)
        print(f'max log-prob: {max(log_probs)}')
        max_log_prob_ind = np.argmax(log_probs)

        print(f'took {get_time() - time0:.2g} seconds to process resampling')
        print(f'propensity at max log-prob: {propensities[max_log_prob_ind]:.4g}')
        print(f'params at max log-prob: {params[max_log_prob_ind]}')
        print(f'propensity at max weight: {propensities[max_weight_ind]:.4g}')
        print(f'log_prob at max weight: {log_probs[max_weight_ind]:.4g}')
        print(f'log_prob at all-data fit: {self.get_log_likelihood(self.all_data_params):.4g}')
        print(f'params at max weight: {params[max_weight_ind]}')
        print(f'params at max weight: {self.get_log_likelihood(params[max_weight_ind])}')

        param_inds = np.random.choice(len(params), len(params), p=weights, replace=True)
        weight_sampled_params = [params[i] for i in param_inds]
        return weight_sampled_params, params, weights, log_probs

    def render_and_plot_cred_int(self,
                                 param_type=None,  # bootstrap, likelihood_sample, random_walk
                                 opt_plot=True
                                 ):
        '''
        Use arviz to plot the credible intervals
        :return: None, just adds attributes to the object
        '''

        plot_param_names = self.sorted_names + list(self.extra_params.keys())
        plot_param_names = ['alpha_1', 'alpha_2', 'contagious_to_positive_delay', 'positive_to_deceased_delay',
                            'positive_to_deceased_mult']
        calc_param_names = self.sorted_names + list(self.extra_params.keys())

        if param_type == 'bootstrap':
            params = [[x[param_name] for param_name in self.sorted_names] for x in self.bootstrap_params]
            prior_weights = self.bootstrap_weights
        elif param_type == 'likelihood_sample':
            params, _, weights, _ = self._get_weighted_samples()
            # from scipy.interpolate import Rbf
            # rbfi = Rbf(*all_samples_as_array.T, [np.exp(x) for x in all_log_probs])
            # bandwidth = [(x[1] - x[0]) / 1000 for x in bounds_to_use.values()]
            # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(all_samples_as_array)
            prior_weights = [1] * len(params)
        elif param_type == 'random_walk':
            print('Since this is a random walk, ignoring weights when fitting NVM...')
            params = self.all_random_walk_samples_as_list
            prior_weights = [1] * len(params)
        else:
            raise ValueError

        map_name_to_distro_without_prior = dict()
        for param_name in self.sorted_names:
            param_ind = self.map_name_to_sorted_ind[param_name]
            param_distro = [param_list[param_ind] for param_list in params if
                            np.isfinite(param_list[param_ind]) and not np.isnan(param_list[param_ind])]
            map_name_to_distro_without_prior[param_name] = np.array(param_distro)
        for param_name, param_func in self.extra_params.items():
            param_distro = [param_func(param_list) for param_list in params if
                            np.isfinite(param_func(param_list)) and not np.isnan(param_func(param_list))]
            map_name_to_distro_without_prior[param_name] = np.array(param_distro)

        print(f'\n----\n{param_type} Means withOUT Priors Applied\n----')
        map_name_to_mean_without_prior = dict()
        map_name_to_median_without_prior = dict()
        for name in calc_param_names:
            if len(map_name_to_distro_without_prior[name]) == 0:
                print(f'Skipping distribution calcs for param_type {param_type}')
                return
            else:
                print(f'{name}: {map_name_to_distro_without_prior[name].mean()}')
                map_name_to_mean_without_prior[name] = map_name_to_distro_without_prior[name].mean()
                map_name_to_median_without_prior[name] = np.percentile(map_name_to_distro_without_prior[name], 50)

        data = az.convert_to_inference_data(map_name_to_distro_without_prior)

        full_output_filename = path.join(self.plot_filename_base, '{}_param_distro_without_priors.png'.format(param_type))
        if not path.exists(full_output_filename) or self.opt_force_plot:
            try:
                print('Printing...',
                      path.join(self.plot_filename_base, '{}_param_distro_without_priors.png'.format(param_type)))
                az.plot_posterior(data,
                                  round_to=3,
                                  credible_interval=0.9,
                                  group='posterior',
                                  var_names=plot_param_names)  # show=True allows for plotting within IDE
                plt.savefig(full_output_filename, dpi=self.plot_dpi)
                plt.close()
            except:
                print(f"Error printing", full_output_filename)

        cred_int_without_priors = dict()
        inner_cred_int_without_priors = dict()
        print('\n----\nHighest Probability Density Intervals withOUT Priors Applied\n----')
        for name in calc_param_names:
            cred_int = tuple(az.hpd(data.posterior[name].T, credible_interval=0.9)[0])
            cred_int_without_priors[name] = cred_int
            print(f'Param {name} 90% HPD: ({cred_int[0]:.4g}, {cred_int[1]:.4g})')
            inner_cred_int = tuple(az.hpd(data.posterior[name].T, credible_interval=0.5)[0])
            inner_cred_int_without_priors[name] = cred_int

        ####
        # Now apply priors
        ####

        print('\n----\nWhat % of samples do we use after applying priors?')
        print(f'{sum(prior_weights) / len(prior_weights) * 100:.4g}%')
        print('----')

        map_name_to_mean_with_prior = None
        map_name_to_distro_with_prior = None
        cred_int_with_priors = None
        inner_cred_int_with_priors = None
        map_name_to_mean_with_prior = None
        map_name_to_median_with_prior = None

        if sum(prior_weights) > 0:

            map_name_to_distro_with_prior = dict()
            for param_ind, param_name in enumerate(self.sorted_names):
                param_distro = [params[i][param_ind] for i in range(len(params)) if
                                prior_weights[i] > 0.5]
                map_name_to_distro_with_prior[param_name] = np.array(
                    [x for x in param_distro if np.isfinite(x) and not np.isnan(x)])
            for param_name, param_func in self.extra_params.items():
                param_distro = [param_func(param_list) for param_list in params if
                                np.isfinite(param_func(param_list)) and not np.isnan(param_func(param_list))]
                map_name_to_distro_with_prior[param_name] = np.array(param_distro)

            print('\n----\nMeans/Medians with Priors Applied\n----')
            map_name_to_mean_with_prior = dict()
            map_name_to_median_with_prior = dict()
            for name in calc_param_names:
                print(f'{name}: {map_name_to_distro_with_prior[name].mean()}')
                map_name_to_mean_with_prior[name] = map_name_to_distro_with_prior[name].mean()
                map_name_to_median_with_prior[name] = np.percentile(map_name_to_distro_with_prior[name], 50)

            # if we want floating-point weights, have to find the common multiple and do discrete sampling
            # for example, this commented line doesn't change the HDP values, since it just duplicates all values
            # data = az.convert_to_inference_data({key: np.array(list(val) + list(val)) for key,val in map_name_to_distro.items()})

            data = az.convert_to_inference_data(map_name_to_distro_with_prior)

            full_output_filename = path.join(self.plot_filename_base, '{}_param_distro_with_priors.png'.format(param_type))
            if not path.exists(full_output_filename) or self.opt_force_plot:
                try:
                    print('Printing...',
                          path.join(self.plot_filename_base, '{}_param_distro_with_priors.png'.format(param_type)))
                    az.plot_posterior(data,
                                      round_to=3,
                                      credible_interval=0.9,
                                      group='posterior',
                                      var_names=plot_param_names)  # show=True allows for plotting within IDE
                    plt.savefig(full_output_filename, dpi=self.plot_dpi)
                    plt.close()
                except:
                    print(f'Error plotting {param_type} for ', full_output_filename)

            cred_int_with_priors = dict()
            inner_cred_int_with_priors = dict()
            print('\n----\nHighest Probability Density Intervals with Priors Applied\n----')
            for name in calc_param_names:
                cred_int = tuple(az.hpd(data.posterior[name].T, credible_interval=0.9)[0])
                cred_int_with_priors[name] = cred_int
                print(f'Param {name} 90% HPD: ({cred_int[0]:.4g}, {cred_int[1]:.4g})')
                inner_cred_int = tuple(az.hpd(data.posterior[name].T, credible_interval=0.5)[0])
                inner_cred_int_with_priors[name] = inner_cred_int

        if param_type == 'bootstrap':
            self.map_param_name_to_bootstrap_distro_without_prior = map_name_to_distro_without_prior
            self.map_param_name_to_bootstrap_distro_with_prior = map_name_to_distro_with_prior
            self.bootstrap_cred_int_without_priors = cred_int_without_priors
            self.bootstrap_cred_int_with_priors = cred_int_with_priors
            self.bootstrap_inner_cred_int_without_priors = inner_cred_int_without_priors
            self.bootstrap_inner_cred_int_with_priors = inner_cred_int_with_priors
            self.bootstrap_means_without_priors = map_name_to_mean_without_prior
            self.bootstrap_means_with_priors = map_name_to_mean_with_prior
            self.bootstrap_medians_without_priors = map_name_to_median_without_prior
            self.bootstrap_medians_with_priors = map_name_to_median_with_prior
        if param_type == 'likelihood_sample':
            self.map_param_name_to_likelihood_sample_distro_without_prior = map_name_to_distro_without_prior
            self.map_param_name_to_likelihood_sample_distro_with_prior = map_name_to_distro_with_prior
            self.likelihood_sample_cred_int_without_priors = cred_int_without_priors
            self.likelihood_sample_cred_int_with_priors = cred_int_with_priors
            self.likelihood_sample_inner_cred_int_without_priors = inner_cred_int_without_priors
            self.likelihood_sample_inner_cred_int_with_priors = inner_cred_int_with_priors
            self.likelihood_sample_means_without_priors = map_name_to_mean_without_prior
            self.likelihood_sample_means_with_priors = map_name_to_mean_with_prior
            self.likelihood_sample_medians_without_priors = map_name_to_median_without_prior
            self.likelihood_sample_medians_with_priors = map_name_to_median_with_prior
        if param_type == 'random_walk':
            self.map_param_name_to_random_walk_distro_without_prior = map_name_to_distro_without_prior
            self.map_param_name_to_random_walk_distro_with_prior = map_name_to_distro_with_prior
            self.random_walk_cred_int_without_priors = cred_int_without_priors
            self.random_walk_cred_int_with_priors = cred_int_with_priors
            self.random_walk_inner_cred_int_without_priors = inner_cred_int_without_priors
            self.random_walk_inner_cred_int_with_priors = inner_cred_int_with_priors
            self.random_walk_means_without_priors = map_name_to_mean_without_prior
            self.random_walk_means_with_priors = map_name_to_mean_with_prior
            self.random_walk_medians_without_priors = map_name_to_median_without_prior
            self.random_walk_medians_with_priors = map_name_to_median_with_prior

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

        # Plot all-data solution 
        if self.opt_plot_bootstraps:
            self.solve_and_plot_solution(in_params=self.all_data_params,
                                         title='All-Data Solution',
                                         plot_filename_filename='all_data_solution.png')

            # Plot example solutions from bootstrap
            bootstrap_selection = np.random.choice(self.bootstrap_params)
            self.solve_and_plot_solution(in_params=bootstrap_selection,
                                         title='Random Bootstrap Selection',
                                         plot_filename_filename='random_bootstrap_selection.png')

            # Plot all bootstraps
            self.plot_all_solutions()

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='bootstrap', opt_plot=self.opt_plot_bootstraps)

        # Do random walks around the overall fit
        # print('\nSampling around MLE with wide sigma')
        # self.MCMC(self.all_data_params, opt_walk=False,
        #           bounds_range_to_sigma_denom=1)
        print('Sampling around MLE with medium sigma')
        self.MCMC(self.all_data_params, opt_walk=False,
                  bounds_range_to_sigma_denom=10)
        print('Sampling around MLE with narrow sigma')
        self.MCMC(self.all_data_params, opt_walk=False,
                  bounds_range_to_sigma_denom=100)

        # Next define MVN model on likelihood and fit        
        self.fit_MVN_to_likelihood(opt_walk=False,
                                   opt_plot=self.opt_plot_likelihood_samples)

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='likelihood_sample',
                                      opt_plot=self.opt_plot_likelihood_samples)

        print('Sampling via random walk MCMC, starting with MLE')  # a random bootstrap selection')
        # bootstrap_selection = np.random.choice(self.n_bootstraps)
        # starting_point = self.bootstrap_params[bootstrap_selection]
        self.MCMC(self.all_data_params, opt_walk=True)

        # print('Sampling via random walk NUTS, starting with MLE')
        # self.NUTS(self.all_data_params)

        # Next define MVN model on likelihood and fit
        self.fit_MVN_to_likelihood(opt_walk=True,
                                   opt_plot=self.opt_plot_random_walk)
        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='random_walk',
                                      opt_plot=self.opt_plot_random_walk)

        # Get extra likelihood samples
        # print('Just doing random sampling')
        # self.render_likelihood_samples()

        # Next define GMM model on likelihood and fit
        # self.fit_GMM_to_likelihood()

    @property
    def opt_plot_bootstraps(self):
        return (not self.loaded_bootstraps) or self.opt_force_plot

    @property
    def opt_plot_likelihood_samples(self):
        opt_plot_likelihood_samples = True
        for x in self.loaded_MCMC:
            if not x['opt_walk']:
                opt_plot_likelihood_samples = False
        return opt_plot_likelihood_samples or self.opt_force_plot

    @property
    def opt_plot_random_walk(self):
        opt_plot_random_walk = True
        for x in self.loaded_MCMC:
            if x['opt_walk']:
                opt_plot_random_walk = False
        return opt_plot_random_walk or self.opt_force_plot

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


def render_whisker_plot(state_report,
                        param_name='alpha_2',
                        output_filename_format_str='test_boxplot_for_{}_{}.png'):
    tmp_ind = [i for i, x in state_report.iterrows() if x['param'] == param_name]
    tmp_ind = sorted(tmp_ind, key=lambda x: state_report.iloc[x]['BS_p50'])

    small_state_report = state_report.iloc[tmp_ind]
    small_state_report.to_csv('state_report_{}.csv'.format(param_name))

    latex_str = small_state_report[['BS_p5', 'BS_p50', 'BS_p95']].to_latex(index=False, float_format="{:0.4f}".format)
    print(latex_str)

    BS_boxes = list()
    for i in range(len(small_state_report)):
        row = pd.DataFrame([small_state_report.iloc[i]])
        new_box = \
            {
                'label': 'Bootstrap',
                'whislo': row['BS_p5'].values[0],  # Bottom whisker position
                'q1': row['BS_p25'].values[0],  # First quartile (25th percentile)
                'med': row['BS_p50'].values[0],  # Median         (50th percentile)
                'q3': row['BS_p75'].values[0],  # Third quartile (75th percentile)
                'whishi': row['BS_p95'].values[0],  # Top whisker position
                'fliers': []  # Outliers
            }
        BS_boxes.append(new_box)
    LS_boxes = list()
    for i in range(len(small_state_report)):
        row = pd.DataFrame([small_state_report.iloc[i]])
        new_box = \
            {
                'label': 'Direct Likelihood Sampling',
                'whislo': row['LS_p5'].values[0],  # Bottom whisker position
                'q1': row['LS_p25'].values[0],  # First quartile (25th percentile)
                'med': row['LS_p50'].values[0],  # Median         (50th percentile)
                'q3': row['LS_p75'].values[0],  # Third quartile (75th percentile)
                'whishi': row['LS_p95'].values[0],  # Top whisker position
                'fliers': []  # Outliers
            }
        LS_boxes.append(new_box)
    MCMC_boxes = list()
    for i in range(len(small_state_report)):
        row = pd.DataFrame([small_state_report.iloc[i]])
        new_box = \
            {
                'label': 'MCMC',
                'whislo': row['MCMC_p5'].values[0],  # Bottom whisker position
                'q1': row['MCMC_p25'].values[0],  # First quartile (25th percentile)
                'med': row['MCMC_p50'].values[0],  # Median         (50th percentile)
                'q3': row['MCMC_p75'].values[0],  # Third quartile (75th percentile)
                'whishi': row['MCMC_p95'].values[0],  # Top whisker position
                'fliers': []  # Outliers
            }
        MCMC_boxes.append(new_box)

    plt.close()
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 10.5)

    n_groups = 3
    ax1 = ax.bxp(BS_boxes, showfliers=False, positions=range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=1.2 / n_groups, patch_artist=True, vert=False)
    ax2 = ax.bxp(LS_boxes, showfliers=False, positions=range(2, len(LS_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=1.2 / n_groups, patch_artist=True, vert=False)
    ax3 = ax.bxp(MCMC_boxes, showfliers=False, positions=range(3, len(MCMC_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=1.2 / n_groups, patch_artist=True, vert=False)

    # plt.yticks([x + 0.5 for x in range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1))], small_state_report['state'])
    plt.yticks(range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)), small_state_report['state'])

    # fill with colors
    colors = ['red', 'green', 'blue']
    for ax, color in zip((ax1, ax2, ax3), colors):
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for patch in ax[item]:
                try:
                    patch.set_facecolor(color)
                except:
                    pass
                patch.set_color(color)
                # patch.set_markeredgecolor(color)

    # add legend
    custom_lines = [
        Line2D([0], [0], color="red", lw=4),
        Line2D([0], [0], color="green", lw=4),
        Line2D([0], [0], color="blue", lw=4),
    ]
    plt.legend(custom_lines, ('Bootstraps', 'MCMC'))

    # increase left margin
    output_filename = output_filename_format_str.format(param_name, 'with_direct_samples')
    plt.subplots_adjust(left=0.2)
    plt.savefig(output_filename, dpi=300)
    # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])

    plt.close()
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 10.5)

    n_groups = 2
    ax1 = ax.bxp(BS_boxes, showfliers=False, positions=range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=1.2 / n_groups, patch_artist=True, vert=False)
    # ax2 = ax.bxp(LS_boxes, showfliers=False, positions=range(2, len(LS_boxes) * (n_groups + 1), (n_groups + 1)),
    #              widths=1.2 / n_groups, patch_artist=True, vert=False)
    ax3 = ax.bxp(MCMC_boxes, showfliers=False, positions=range(2, len(MCMC_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=1.2 / n_groups, patch_artist=True, vert=False)

    # plt.yticks([x + 0.5 for x in range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1))], small_state_report['state'])
    plt.yticks(range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)), small_state_report['state'])

    # fill with colors
    colors = ['red', 'blue', 'green']
    for ax, color in zip((ax1, ax3), colors):
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for patch in ax[item]:
                try:
                    patch.set_facecolor(color)
                except:
                    pass
                patch.set_color(color)
                # patch.set_markeredgecolor(color)

    # add legend
    custom_lines = [
        Line2D([0], [0], color="red", lw=4),
        Line2D([0], [0], color="blue", lw=4),
        # Line2D([0], [0], color="green", lw=4),
    ]
    plt.legend(custom_lines, ('Bootstraps', 'MCMC'))

    # increase left margin
    output_filename = output_filename_format_str.format(param_name, 'without_direct_samples')
    plt.subplots_adjust(left=0.2)
    plt.savefig(output_filename, dpi=300)
    # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])
