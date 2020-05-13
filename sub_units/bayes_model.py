from sub_units.utils import Stopwatch
import numpy as np
import pandas as pd
from enum import Enum

pd.plotting.register_matplotlib_converters()  # addresses complaints about Timestamp instead of float for plotting x-values
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import joblib
from os import path
from tqdm import tqdm
import os
from scipy.optimize import approx_fprime
from functools import lru_cache, partial
from abc import ABC, abstractmethod

plt.style.use('seaborn-darkgrid')
matplotlib.use('Agg')
import datetime
import matplotlib.dates as mdates
import arviz as az
import seaborn as sns


class ApproxType(Enum):
    bootstrap = 'bootstrap'
    bootstrap_with_priors = 'bootstrap_with_priors'
    likelihood_sample = 'likelihood_sample'
    MVN_fit = 'MVN_fit'
    MCMC = 'MCMC'

    def __str__(self):
        return str(self.value)


class WhichDistro(Enum):
    norm = 'norm'
    laplace = 'laplace'

    def __str__(self):
        return str(self.value)


class BayesModel(ABC):

    # this fella isn't necessary like other abstractmethods, but optional in a subclass that supports statsmodels solutions
    def render_statsmodels_fit(self):
        pass

    @abstractmethod
    def run_simulation(self, in_params):
        pass

    @abstractmethod
    def _get_log_likelihood_precursor(self,
                                      in_params,
                                      data_new_tested=None,
                                      data_new_dead=None,
                                      cases_bootstrap_indices=None,
                                      deaths_bootstrap_indices=None,
                                      ):
        pass

    def convert_params_as_list_to_dict(self, in_params, map_name_to_sorted_ind=None):
        '''
        Helper function to convert params as a list to the dictionary form
        :param in_params: params as list
        :return: params as dict
        '''

        if map_name_to_sorted_ind is None:
            map_name_to_sorted_ind = self.map_name_to_sorted_ind

        # convert from list to dictionary (for compatibility with the least-sq solver
        if type(in_params) != dict and self.map_name_to_sorted_ind is not None:
            params = {key: in_params[ind] for key, ind in map_name_to_sorted_ind.items()}
        else:
            params = in_params.copy()
        return params

    def convert_params_as_dict_to_list(self, in_params):
        '''
        Helper function to convert params as a dict to the list form
        :param in_params: params as dict
        :return: params as list
        '''

        if type(in_params) == dict:
            p0 = [in_params[name] for name in self.sorted_names]
        else:
            p0 = in_params.copy()

        return p0

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
                 opt_calc=True,
                 model_type_name=None,
                 plot_param_names=None,
                 **kwargs
                 ):

        for key, val in kwargs.items():
            print(f'Adding extra params to attributes... {key}: {val}')
            setattr(self, key, val)

        self.model_type_name = model_type_name
        self.state_name = state_name
        self.max_date_str = max_date_str
        self.n_bootstraps = n_bootstraps
        self.n_likelihood_samples = n_likelihood_samples
        self.burn_in = burn_in
        self.max_date = datetime.datetime.strptime(max_date_str, '%Y-%m-%d')
        self.static_params = static_params

        self.bootstrap_filename = path.join('state_bootstraps',
                                            f"{state_name.lower().replace(' ', '_')}_{model_type_name}_{n_bootstraps}_bootstraps_max_date_{max_date_str.replace('-', '_')}.joblib")
        self.likelihood_samples_filename_format_str = path.join('state_likelihood_samples',
                                                                f"{state_name.lower().replace(' ', '_')}_{model_type_name}_{{}}_{n_likelihood_samples}_samples_max_date_{max_date_str.replace('-', '_')}.joblib")
        self.likelihood_samples_from_bootstraps_filename = path.join('state_likelihood_samples',
                                                                     f"{state_name.lower().replace(' ', '_')}_{model_type_name}_{n_bootstraps}_bootstraps_likelihoods_max_date_{max_date_str.replace('-', '_')}.joblib")

        self.plot_subfolder = f'{max_date_str.replace("-", "_")}_date_{model_type_name}_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples'
        self.plot_subfolder = path.join('state_plots', self.plot_subfolder)
        self.plot_filename_base = path.join(self.plot_subfolder,
                                            state_name.lower().replace(' ', '_').replace('.', ''))

        if not os.path.exists('state_bootstraps'):
            os.mkdir('state_bootstraps')
        if not os.path.exists('state_likelihood_samples'):
            os.mkdir('state_likelihood_samples')
        if not os.path.exists('state_plots'):
            os.mkdir('state_plots')
        if not os.path.exists(self.plot_subfolder):
            os.mkdir(self.plot_subfolder)
        if not os.path.exists(self.plot_filename_base):
            os.mkdir(self.plot_filename_base)

        if load_data_obj is None:
            from sub_units import load_data as load_data_obj

        state_data = load_data_obj.get_state_data(state_name)

        # I replaced this with the U.S. total so everyone's on the same playing field, otherwise: state_data['sip_date']
        self.SIP_date = datetime.datetime.strptime('2020-03-20', '%Y-%m-%d')

        self.cases_indices = None
        self.deaths_indices = None

        self.min_date = state_data['min_date']
        self.population = state_data['population']
        self.n_count_data = state_data['series_data'][:, 1].size
        self.SIP_date_in_days = (self.SIP_date - self.min_date).days
        self.max_date_in_days = (self.max_date - self.min_date).days
        self.series_data = state_data['series_data'][:self.max_date_in_days, :]  # cut off recent days if desired

        n_t_vals = self.max_date_in_days + 21  # project three weeks into the future
        self.t_vals = np.linspace(-burn_in, n_t_vals, burn_in + n_t_vals + 1)
        # print('min_date', self.min_date)
        # print('max_date_in_days', self.max_date_in_days)
        # print('t_vals', self.t_vals)

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

        self.extra_params = {key: partial(val, map_name_to_sorted_ind=self.map_name_to_sorted_ind) for key, val in
                             extra_params.items()}

        if plot_param_names is None:
            self.plot_param_names = self.sorted_names
        else:
            self.plot_param_names = plot_param_names

    @staticmethod
    def norm(x, mu=0, std=0):
        return np.exp(-((x - mu) / std) ** 2) / (np.sqrt(2 * np.pi) * std)

    def _errfunc_for_least_squares(self,
                                   in_params,
                                   data_new_tested=None,
                                   data_new_dead=None,
                                   cases_bootstrap_indices=None,
                                   deaths_bootstrap_indices=None,
                                   precursor_func=None,
                                   ):
        '''
        Helper function for scipy.optimize.least_squares
          Basically returns log likelihood precursors
        :param in_params: dictionary or list of parameters
        :param cases_bootstrap_indices: bootstrap indices when applicable
        :param deaths_bootstrap_indices:  bootstrap indices when applicable
        :param cases_bootstrap_indices: which indices to include in the likelihood?
        :param deaths_bootstrap_indices: which indices to include in the likelihood?
        :return: list: distances and other loss function contributions
        '''

        if precursor_func is None:
            precursor_func = self._get_log_likelihood_precursor

        positive_dists, deceased_dists, other_errs, sol, positive_vals, deceased_vals = precursor_func(
            in_params,
            data_new_tested=data_new_tested,
            data_new_dead=data_new_dead,
            cases_bootstrap_indices=cases_bootstrap_indices,
            deaths_bootstrap_indices=deaths_bootstrap_indices)

        dists = positive_dists + deceased_dists
        vals = positive_vals + deceased_vals

        sigmas = [1 for _ in
                  vals]  # least squares optimization doesn't care about the sigmas, it's just looking for the mode
        coeffs = [1 / x for x in sigmas]
        new_dists = [
            np.sqrt(dist ** 2 / (2 * sigma ** 2)) for dist, sigma, coeff in zip(dists, sigmas, coeffs)]

        # new_dists = [dists[i] / np.sqrt(2 * np.log(np.sqrt(vals[i]))) for i in range(len(dists))]
        # new_dists = [dists[i] / np.sqrt(2 * in_params[self.map_name_to_sorted_ind['sigma']]) for i in range(len(dists))]

        return new_dists + other_errs

    def get_log_likelihood(self,
                           in_params,
                           data_new_tested=None,
                           data_new_dead=None,
                           cases_bootstrap_indices=None,
                           deaths_bootstrap_indices=None,
                           opt_return_sol=False,
                           precursor_func=None,
                           ):
        '''
        Obtain the log likelihood given a set of in_params
        :param in_params: dictionary or list of parameters
        :param cases_bootstrap_indices: bootstrap indices when applicable
        :param deaths_bootstrap_indices:  bootstrap indices when applicable
        :param cases_bootstrap_indices: which indices to include in the likelihood?
        :param deaths_bootstrap_indices: which indices to include in the likelihood?
        :return: float: log likelihood
        '''

        params = self.convert_params_as_list_to_dict(in_params)

        if precursor_func is None:
            precursor_func = self._get_log_likelihood_precursor

        dists_positive, dists_deceased, other_errs, sol, vals_positive, vals_deceased = precursor_func(
            params,
            data_new_tested=data_new_tested,
            data_new_dead=data_new_dead,
            cases_bootstrap_indices=cases_bootstrap_indices,
            deaths_bootstrap_indices=deaths_bootstrap_indices)

        # sigmas = [1 / np.sqrt(x) for x in vals] # using the rule that log(x) - log(x - y) => 1/y for x >> y, and here y = sqrt(x)
        sigmas_positive = [params['sigma_positive'] for _ in vals_positive]
        sigmas_deceased = [params['sigma_deceased'] for _ in vals_deceased]

        coeffs_positive = [1 / x for x in sigmas_positive]
        coeffs_deceased = [1 / x for x in sigmas_deceased]
        return_val_positive = sum(
            -dist ** 2 / (2 * sigma ** 2) + np.log(coeff) for dist, sigma, coeff in
            zip(dists_positive, sigmas_positive, coeffs_positive))
        return_val_deceased = sum(
            -dist ** 2 / (2 * sigma ** 2) + np.log(coeff) for dist, sigma, coeff in
            zip(dists_deceased, sigmas_deceased, coeffs_deceased))
        return_val_other = - sum(x ** 2 for x in other_errs)

        return_val = return_val_positive + return_val_deceased + return_val_other

        if opt_return_sol:
            return return_val, sol
        else:
            return return_val

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
        optimize_test_errfunc = partial(self._errfunc_for_least_squares,
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

        # # have to compute sigma after the fact
        # sol = self.run_simulation(params_as_dict)
        # 
        # new_positive = sol[1]
        # new_deceased = sol[2]
        # params_as_dict['sigma'] = 0.06 # quick hack

        return params_as_dict

    def fit_curve_via_likelihood(self,
                                 in_params,
                                 tested_indices=None,
                                 deaths_indices=None,
                                 method='SLSQP',  # 'Nelder-Mead
                                 print_success=False
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

        p0 = self.convert_params_as_dict_to_list(in_params)

        def get_neg_log_likelihood(p):
            return -self.get_log_likelihood(p,
                                            cases_bootstrap_indices=tested_indices,
                                            deaths_bootstrap_indices=deaths_indices
                                            )

        bounds_to_use = [self.curve_fit_bounds[name] for name in self.sorted_names]
        results = sp.optimize.minimize(get_neg_log_likelihood, p0, bounds=bounds_to_use, method=method)
        if print_success:
            print(f'success? {results.success}')
        params_as_list = results.x
        params_as_dict = {key: params_as_list[i] for i, key in enumerate(self.sorted_names)}

        return params_as_dict

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

            min_slice = None
            if self.min_sol_date is not None:
                for i in range(len(sol_plot_date_range)):
                    if sol_plot_date_range[i] >= self.min_sol_date:
                        min_slice = i
                        break

            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    new_positive[min_plot_pt: max_plot_pt][slice(min_slice, None)], 'green', label='positive')
            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    new_deceased[min_plot_pt: max_plot_pt][slice(min_slice, None)], 'red', label='deceased')
            ax.plot(data_plot_date_range, self.data_new_tested, '.', color='darkgreen', label='confirmed cases')
            ax.plot(data_plot_date_range, self.data_new_dead, '.', color='darkred', label='confirmed deaths')

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
            plt.close()

            # for i in range(len(sol)):
            #     print(f'index: {i}, odeint_value: {sol[i]}, real_value: {[None, series_data[i]]}')

    def plot_all_solutions(self, n_samples=1000, key='bootstrap', n_sols_to_plot=1000):
        '''
        Plot all the bootstrap simulation solutions
        :param n_sols_to_plot: how many simulations should we sample for the plot?
        :return: None
        '''

        if key == 'bootstrap':
            params = self.bootstrap_params
            param_inds_to_plot = list(range(len(params)))
        elif key == 'bootstrap_with_priors':
            params = self.bootstrap_params
            param_inds_to_plot = [i for i, sol in enumerate(params) if round(self.bootstrap_weights[i])]
        elif key == 'likelihood_samples':
            params, _, _, log_probs = self.get_weighted_samples_via_direct_sampling()
            param_inds_to_plot = list(range(len(params)))
        elif key == 'random_walk':
            params = self.all_random_walk_samples_as_list
            param_inds_to_plot = list(range(len(params)))
        elif key == 'MVN_fit':
            params, _, _, log_probs = self.get_weighted_samples_via_MVN()
            param_inds_to_plot = list(range(len(params)))
        else:
            raise ValueError

        print(f'Rendering solutions for {key}...')
        param_inds_to_plot = np.random.choice(param_inds_to_plot, min(n_samples, len(param_inds_to_plot)),
                                              replace=False)
        sols_to_plot = [self.run_simulation(in_params=params[param_ind]) for param_ind in tqdm(param_inds_to_plot)]
        self._plot_all_solutions_sub_distinct_lines_with_alpha(sols_to_plot,
                                                               plot_filename_filename=f'{key}_solutions_discrete.png')
        self._plot_all_solutions_sub_filled_quantiles(sols_to_plot,
                                                      plot_filename_filename=f'{key}_solutions_filled_quantiles.png')

    def _plot_all_solutions_sub_filled_quantiles(self,
                                                 sols_to_plot,
                                                 plot_filename_filename=None):
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
        fig, ax = plt.subplots()
        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + 14 + self.burn_in)
        data_plot_date_range = [self.min_date + datetime.timedelta(days=1) * i for i in
                                range(len(self.data_new_tested))]

        sol_plot_date_range = [self.min_date - datetime.timedelta(days=self.burn_in) + datetime.timedelta(
            days=1) * i for i in
                               range(len(sol[0]))][min_plot_pt:max_plot_pt]

        map_t_val_ind_to_tested_distro = dict()
        map_t_val_ind_to_deceased_distro = dict()
        for sol in sols_to_plot:
            new_tested = sol[1]
            # cum_tested = np.cumsum(new_tested)
            new_dead = sol[2]
            # cum_dead = np.cumsum(new_dead)

            for val_ind, val in enumerate(self.t_vals):
                if val_ind not in map_t_val_ind_to_tested_distro:
                    map_t_val_ind_to_tested_distro[val_ind] = list()
                map_t_val_ind_to_tested_distro[val_ind].append(new_tested[val_ind])
                if val_ind not in map_t_val_ind_to_deceased_distro:
                    map_t_val_ind_to_deceased_distro[val_ind] = list()
                map_t_val_ind_to_deceased_distro[val_ind].append(new_dead[val_ind])

        p5_curve = [np.percentile(map_t_val_ind_to_deceased_distro[val_ind], 5) for val_ind in range(len(self.t_vals))]
        p25_curve = [np.percentile(map_t_val_ind_to_deceased_distro[val_ind], 25) for val_ind in
                     range(len(self.t_vals))]
        p50_curve = [np.percentile(map_t_val_ind_to_deceased_distro[val_ind], 50) for val_ind in
                     range(len(self.t_vals))]
        p75_curve = [np.percentile(map_t_val_ind_to_deceased_distro[val_ind], 75) for val_ind in
                     range(len(self.t_vals))]
        p95_curve = [np.percentile(map_t_val_ind_to_deceased_distro[val_ind], 95) for val_ind in
                     range(len(self.t_vals))]

        min_slice = None
        if self.min_sol_date is not None:
            for i in range(len(sol_plot_date_range)):
                if sol_plot_date_range[i] >= self.min_sol_date:
                    min_slice = i
                    break
        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p5_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p95_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        color="red", alpha=0.3)
        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p25_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p75_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        color="red", alpha=0.6)
        ax.plot(sol_plot_date_range[slice(min_slice, None)], p50_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                color="darkred")

        p5_curve = [np.percentile(map_t_val_ind_to_tested_distro[val_ind], 5) for val_ind in range(len(self.t_vals))]
        p25_curve = [np.percentile(map_t_val_ind_to_tested_distro[val_ind], 25) for val_ind in
                     range(len(self.t_vals))]
        p50_curve = [np.percentile(map_t_val_ind_to_tested_distro[val_ind], 50) for val_ind in
                     range(len(self.t_vals))]
        p75_curve = [np.percentile(map_t_val_ind_to_tested_distro[val_ind], 75) for val_ind in
                     range(len(self.t_vals))]
        p95_curve = [np.percentile(map_t_val_ind_to_tested_distro[val_ind], 95) for val_ind in
                     range(len(self.t_vals))]

        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p5_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p95_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        color="green", alpha=0.3)
        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p25_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p75_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        color="green", alpha=0.6)
        ax.plot(sol_plot_date_range[slice(min_slice, None)], p50_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                color="darkgreen")

        ax.plot(data_plot_date_range, self.data_new_tested, '.', color='darkgreen', label='cases')
        ax.plot(data_plot_date_range, self.data_new_dead, '.', color='darkred', label='deaths')
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
        plt.close()

    def _plot_all_solutions_sub_distinct_lines_with_alpha(self,
                                                          sols_to_plot,
                                                          n_sols_to_plot=1000,
                                                          plot_filename_filename=None):
        '''
        Helper function to plot_all_solutions
        :param n_sols_to_plot: how many simulations should we sample for the plot?
        :param plot_filename_filename: string to add to the plot filename
        :return: None
        '''
        full_output_filename = path.join(self.plot_filename_base, plot_filename_filename)
        if path.exists(full_output_filename) and not self.opt_force_plot:
            return

        if n_sols_to_plot > len(sols_to_plot):
            n_sols_to_plot = len(sols_to_plot)
        sols_to_plot = [sols_to_plot[i] for i in np.random.choice(len(sols_to_plot), n_sols_to_plot, replace=False)]

        print('Printing...', path.join(self.plot_filename_base, plot_filename_filename))
        sol = self.bootstrap_sols[0]
        n_sols = len(sols_to_plot)
        fig, ax = plt.subplots()
        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + 14 + self.burn_in)
        data_plot_date_range = [self.min_date + datetime.timedelta(days=1) * i for i in
                                range(len(self.data_new_tested))]

        for sol in sols_to_plot:
            new_tested = sol[1]
            # cum_tested = np.cumsum(new_tested)

            new_dead = sol[2]
            # cum_dead = np.cumsum(new_dead)

            sol_plot_date_range = [self.min_date - datetime.timedelta(days=self.burn_in) + datetime.timedelta(
                days=1) * i for i in
                                   range(len(sol[0]))][min_plot_pt:max_plot_pt]

            # ax.plot(plot_date_range[min_plot_pt:], [(sol[i][0]) for i in range(min_plot_pt, len(sol[0))], 'b', alpha=0.1)
            # ax.plot(plot_date_range[min_plot_pt:max_plot_pt], [(sol[i][1]) for i in range(min_plot_pt, max_plot_pt)], 'g', alpha=0.1)

            min_slice = None
            if self.min_sol_date is not None:
                for i in range(len(sol_plot_date_range)):
                    if sol_plot_date_range[i] >= self.min_sol_date:
                        min_slice = i
                        break
            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    new_tested[min_plot_pt:max_plot_pt][slice(min_slice, None)], 'g',
                    alpha=5 / n_sols)
            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    new_dead[min_plot_pt:max_plot_pt][slice(min_slice, None)], 'r',
                    alpha=5 / n_sols)

        ax.plot(data_plot_date_range, self.data_new_tested, '.', color='darkgreen', label='cases')
        ax.plot(data_plot_date_range, self.data_new_dead, '.', color='darkred', label='deaths')
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
        plt.close()

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
            self.loaded_bootstraps = True
        except:
            self.loaded_bootstraps = False

        if not success and self.opt_calc:

            print('\n----\nRendering bootstrap model fits... starting with the all-data one...\n----')

            # This is kind of a kludge, I find more reliabel fits with fit_curve_exactly_with_jitter
            #   But it doesn't fit the observation error, which I need for likelihood samples
            #   So I use it to fit everything BUT observation error, then insert the test_params entries for the sigmas,
            #   and re-fit using the jankier (via_likelihood) method that fits the observation error
            #   TODO: make the sigma substitutions empirical, rather than hacky the way I've done it
            test_params_as_list = [self.test_params[key] for key in self.sorted_names]
            all_data_params = self.fit_curve_exactly_with_jitter(test_params_as_list)
            all_data_params['sigma_positive'] = self.test_params['sigma_positive']
            all_data_params['sigma_deceased'] = self.test_params['sigma_deceased']
            print('refitting all-data params')
            all_data_params = self.fit_curve_via_likelihood(all_data_params,
                                                            print_success=True)  # fit_curve_via_likelihood
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
                if True:
                    test_params_as_list = [self.test_params[key] for key in self.sorted_names]
                    all_data_params2 = self.fit_curve_via_likelihood(test_params_as_list,
                                                                     method=method, print_success=True)

                    print('\nParameters when trained on all data (this is our starting point for optimization):')
                    [print(f'{key}: {val:.4g}') for key, val in all_data_params2.items()]
                else:
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
                # tested_jitter = [
                #     max(0.01, self.data_new_tested[i] + np.random.normal(0, np.sqrt(self.data_new_tested[i]))) for i in
                #     range(len(self.data_new_tested))]
                # dead_jitter = [max(0.01, self.data_new_dead[i] + np.random.normal(0, np.sqrt(self.data_new_dead[i])))
                #                for i in
                #                range(len(self.data_new_dead))]

                # here is where we select the all-data parameters as our starting point
                starting_point_as_list = [all_data_params[key] for key in self.sorted_names]

                # NB: define the model constraints (mainly, positive values)
                # This is the old version in which it still attempts to fit exactly on jittered data
                params_as_dict = self.fit_curve_via_likelihood(starting_point_as_list,
                                                               # data_tested=tested_jitter,
                                                               # data_dead=dead_jitter,
                                                               tested_indices=cases_bootstrap_indices,
                                                               deaths_indices=deaths_bootstrap_indices
                                                               )

                # This is the new version which just uses the scalar likelihood
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

            all_propensities_as_list = [len(all_samples_as_list)] * len(all_samples_as_list)
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

    def _add_samples(self, samples, vals, propensities, key=False):
        print('adding samples...')
        print('checking sizes...')

        if key == 'likelihood_samples':
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
        elif key == 'random_walk':
            print(f'samples: {len(samples)}, vals: {len(vals)}, propensities: {len(propensities)}')
            self.all_random_walk_samples_as_list += samples
            self.all_random_walk_log_probs_as_list += vals

            shuffled_ind = list(range(len(self.all_random_walk_samples_as_list)))
            np.random.shuffle(shuffled_ind)
            self.all_random_walk_samples_as_list = [self.all_random_walk_samples_as_list[i] for i in shuffled_ind]
            self.all_random_walk_log_probs_as_list = [self.all_random_walk_log_probs_as_list[i] for i in shuffled_ind]
            print('...done!')
        elif key == 'PyMC3':
            print(f'samples: {len(samples)}, vals: {len(vals)}, propensities: {len(propensities)}')
            self.all_PYMC3_samples_as_list += samples
            self.all_PYMC3_log_probs_as_list += vals

            shuffled_ind = list(range(len(self.all_PYMC3_samples_as_list)))
            np.random.shuffle(shuffled_ind)
            self.all_PYMC3_samples_as_list = [self.all_PYMC3_samples_as_list[i] for i in shuffled_ind]
            self.all_PYMC3_log_probs_as_list = [self.all_PYMC3_log_probs_as_list[i] for i in shuffled_ind]
            print('...done!')

    @lru_cache(maxsize=10)
    def get_propensity_model(self, sample_scale_param, which_distro=WhichDistro.norm):

        sigma = {key: (val[1] - val[0]) / sample_scale_param for key, val in self.curve_fit_bounds.items()}

        # overwrite sigma for values that are strictly positive multipliers of unknown scale
        for param_name in self.logarithmic_params:
            sigma[param_name] = 1 / sample_scale_param
        sigma_as_list = [sigma[name] for name in self.sorted_names]

        if which_distro == WhichDistro.norm:
            cov = np.diag([max(1e-8, x ** 2) for x in sigma_as_list])
            propensity_model = sp.stats.multivariate_normal(cov=cov)
        elif which_distro == WhichDistro.laplace:
            propensity_model = sp.stats.laplace(scale=sigma_as_list)

        return propensity_model

    def MCMC(self, p0, opt_walk=True,
             sample_shape_param=None,
             which_distro=WhichDistro.norm  # 'norm', 'laplace'
             ):

        n_samples = self.n_likelihood_samples
        if opt_walk:
            MCMC_burn_in_frac = 0.2
            if which_distro != WhichDistro.norm:
                raise ValueError
        else:
            MCMC_burn_in_frac = 0

        # take smaller steps for a random walk, larger ones for a gaussian sampling
        if sample_shape_param is None:
            if opt_walk:
                sample_shape_param = 100
            else:
                sample_shape_param = 10

        bounds_to_use = self.curve_fit_bounds

        def get_bunched_up_on_bounds(input_params, new_val=None):
            if new_val is None:
                new_val = np.array([input_params[name] for name in self.sorted_names])
            bunched_up_on_bounds = input_params.copy()
            offending_params = list()
            for param_ind, param_name in enumerate(self.sorted_names):
                lower, upper = self.curve_fit_bounds[param_name]
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

        def acquisition_function(input_params, sample_shape_param):
            output_params = input_params
            jitter_propensity = 1
            accepted = False
            n_attempts = 0

            propensity_model = self.get_propensity_model(sample_shape_param, which_distro=which_distro)

            while n_attempts < 100 and not accepted:  # this limits endless searches with wide sigmas
                n_attempts += 1

                # acq_timer=Stopwatch()
                jitter = propensity_model.rvs()
                # sample_time = acq_timer.elapsed_time()
                # print(f'Sample time: {sample_time * 1000:.4g} ms')
                # acq_timer.reset()
                jitter_propensity = np.prod(propensity_model.pdf(jitter))  # works correctly for laplace and MVN distros
                # pdf_time = acq_timer.elapsed_time()
                # print(f'PDF time: {pdf_time * 1000:.4g} ms')
                # print(f'n_attempts: {n_attempts}')
                for param_name in self.logarithmic_params:
                    if param_name not in self.map_name_to_sorted_ind:
                        continue
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
            filename_str = f'MCMC'
        else:
            filename_str = f'MCMC_fixed'

        filename_str += f'_{which_distro}_sample_shape_param_{int(sample_shape_param)}'

        success = False

        try:
            print(f'loading from {self.likelihood_samples_filename_format_str.format(filename_str)}...')
            tmp_dict = joblib.load(self.likelihood_samples_filename_format_str.format(filename_str))
            samples = tmp_dict['samples']
            log_probs = tmp_dict['vals']
            propensities = tmp_dict['propensities']
            success = True
            print('...done!')
            self.loaded_MCMC.append({'opt_walk': opt_walk, 'sample_shape_param': sample_shape_param})
        except:
            print('...load failed!... doing calculations...')

        timer = Stopwatch()
        prev_test_ind = -1
        n_accepted = 0
        n_accepted_turn = 0
        if not success and self.opt_calc:

            for test_ind in tqdm(range(n_samples)):

                # sub_timer = Stopwatch()
                proposed_p, proposed_propensity = acquisition_function(prev_p, sample_shape_param)
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
                              f'\n sample_shape_param: {sample_shape_param}' + \
                              f'\n acceptance ratio: {acceptance_ratio:.4g}' + \
                              ''.join(f'\n  {key}: {proposed_p[key]:.4g}' for key in self.sorted_names))
                        timer.reset()
                        if opt_walk:
                            sample_shape_param = sample_shape_param * 100

                    if rand_num <= acceptance_ratio and np.isfinite(acceptance_ratio):

                        # it's important to do this as soon as we get a good fit
                        if opt_walk:
                            sample_shape_param = sample_shape_param

                        n_accepted += 1
                        n_accepted_turn += 1
                        if timer.elapsed_time() > 1:
                            n_test_ind_turn = test_ind - prev_test_ind
                            print(
                                f'\n how many samples accepted overall? {n_accepted} of {test_ind} ({n_accepted / test_ind * 100:.4g}%)' + \
                                f'\n how many samples accepted since last update? {n_accepted_turn} of {n_test_ind_turn} ({n_accepted_turn / n_test_ind_turn * 100:.4g}%)' + \
                                f'\n prev. log likelihood: {ll:.4g}' + \
                                f'\n       log likelihood: {proposed_ll:.4g}' + \
                                f'\n sample_shape_param {sample_shape_param}' + \
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

        if opt_walk:
            samples_key = 'random_walk'
        else:
            samples_key = 'likelihood_samples'
        self._add_samples(samples_as_list[MCMC_burn_in:], log_probs[MCMC_burn_in:], propensities[MCMC_burn_in:],
                          key=samples_key)

    @staticmethod
    def cov2corr(cov):
        std_devs_mat = np.diag(np.sqrt(np.diagonal(cov)))
        cov_inv = np.linalg.inv(std_devs_mat)
        corr = cov_inv @ cov @ cov_inv
        return corr

    def fit_MVN_to_likelihood(self,
                              cov_type='full',
                              opt_walk=False):
        if opt_walk:
            print('Since this is a random walk, ignoring weights when fitting NVM...')
            params = self.all_random_walk_samples_as_list
            log_probs = self.all_random_walk_log_probs_as_list
            weights = [1] * len(params)
            filename_str = 'MVN_random_walk'
        else:
            params, weights, log_probs, propensities = self.get_direct_likelihood_samples()
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
        corr = self.cov2corr(cov)

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
            plt.close()

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
            plt.close()

        full_output_filename = path.join(self.plot_filename_base, f'{filename_str}_actual_vs_predicted_vals_linear.png')
        if not path.exists(full_output_filename) or self.opt_force_plot:
            plt.clf()
            plt.plot(np.exp(log_probs), predicted_vals, '.',
                     alpha=max(100 / len(predicted_vals), 0.01))
            plt.xlabel('actual values')
            plt.ylabel('predicted values')
            plt.savefig(full_output_filename, dpi=self.plot_dpi)
            plt.close()

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
            # plt.close()

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

    def get_weighted_samples_via_MVN(self, n_samples=10000):
        '''
        Retrieves likelihood samples in parameter space, weighted by their likelihood (raw, not log) 
        :param n_samples: how many samples to re-sample from the list of likelihood samples
        :return: tuple of weight_sampled_params, params, weights, log_probs
        '''

        if self.MVN_sample_model is None:
            raise ValueError('Need to fit MVN before using get_weighted_samples_via_MVN')

        weight_sampled_params = self.MVN_sample_model.rvs(n_samples)
        log_probs = [self.get_log_likelihood(x) for x in weight_sampled_params]

        return weight_sampled_params, weight_sampled_params, [1] * len(weight_sampled_params), log_probs

    def get_weighted_samples(self, n_samples=10000):
        '''
        Re-route to the MVN solution as opposed to direct sampling solution
        :param n_samples: 
        :return: 
        '''
        return self.get_weighted_samples_via_MVN(n_samples=n_samples)

    def get_direct_likelihood_samples(self):

        valid_ind = [i for i, x in enumerate(self.all_log_probs_as_list) if np.isfinite(np.exp(x)) and \
                     np.isfinite(self.all_propensities_as_list[i]) and \
                     self.all_propensities_as_list[i] > 0]

        propensities = np.array(self.all_propensities_as_list)[valid_ind]
        log_probs = np.array(self.all_log_probs_as_list)[valid_ind]
        params = np.array(self.all_samples_as_list)[valid_ind]

        weights = np.array([np.exp(x) / propensities[i] for i, x in enumerate(log_probs)])
        weights /= sum(weights)

        return params, weights, log_probs, propensities

    def get_weighted_samples_via_direct_sampling(self, n_samples=10000):
        '''
        I DON'T USE THIS NOW
        Retrieves likelihood samples in parameter space, weighted by their likelihood (raw, not log) 
        :param n_samples: how many samples to re-sample from the list of likelihood samples
        :return: tuple of weight_sampled_params, params, weights, log_probs
        '''

        timer = Stopwatch()
        params, weights, log_probs, propensities = self.get_direct_likelihood_samples()

        # # get rid of unlikely points that have a small propensity
        # log_probs = np.array(self.all_log_probs_as_list)[valid_ind]
        # probs = [np.exp(x) for x in log_probs]
        # prob_threshold = max(probs) / 100.0
        # 
        # valid_ind = [i for i, x in enumerate(self.all_log_probs_as_list) if np.isfinite(np.exp(x)) and \
        #              np.isfinite(self.all_propensities_as_list[i]) and \
        #              self.all_propensities_as_list[i] > 0 and \
        #              np.exp(x) > prob_threshold]

        if n_samples > len(params):
            n_samples = len(params)

        valid_ind = [i for i, x in enumerate(weights) if np.isfinite(x)]

        sampled_valid_inds = np.random.choice(len(valid_ind), n_samples, p=np.array(weights)[valid_ind], replace=True)
        param_inds = [valid_ind[i] for i in sampled_valid_inds]
        weight_sampled_params = [params[i] for i in param_inds]

        max_weight_ind = np.argmax(weights)
        max_log_prob_ind = np.argmax(log_probs)

        print(f'took {timer.elapsed_time():.2g} seconds to process resampling')

        print(f'max weight: {max(weights)}')
        print(f'max log-prob: {max(log_probs)}')
        print(f'propensity at max weight: {propensities[max_weight_ind]:.4g}')
        print(f'log_prob at max weight: {log_probs[max_weight_ind]:.4g}')

        print(f'propensity at max log-prob: {propensities[max_log_prob_ind]:.4g}')
        print(f'params at max log-prob: {params[max_log_prob_ind]}')
        print(f'log_prob at all-data fit: {self.get_log_likelihood(self.all_data_params):.4g}')
        print(f'params at max weight: {params[max_weight_ind]}')
        print(f'params at max weight: {self.get_log_likelihood(params[max_weight_ind])}')
        desc_weights_inds = np.argsort(-weights)[:10]
        print(f'descending list of weights: {[weights[i] for i in desc_weights_inds]}')
        print(f'log-probs at descending list of weights {[log_probs[i] for i in desc_weights_inds]}')
        print(f'propensities at descending list of weights {[propensities[i] for i in desc_weights_inds]}')

        return weight_sampled_params, params, weights, log_probs

    def render_and_plot_cred_int(self,
                                 param_type=None,  # bootstrap, likelihood_sample, random_walk, MVN_fit, statsmodels
                                 ):
        '''
        Use arviz to plot the credible intervals
        :return: None, just adds attributes to the object
        '''

        calc_param_names = self.sorted_names + list(self.extra_params.keys())

        if param_type == 'bootstrap':
            params = [[x[param_name] for param_name in self.sorted_names] for x in self.bootstrap_params]
            prior_weights = self.bootstrap_weights
        elif param_type == 'likelihood_sample':
            params, _, _, _ = self.get_weighted_samples_via_direct_sampling()
            # params = [[x[param_name] for param_name in self.sorted_names] for x in params]
            prior_weights = [1] * len(params)
        elif param_type == 'random_walk':
            print('Since this is a random walk, ignoring weights when fitting NVM...')
            params = self.all_random_walk_samples_as_list
            prior_weights = [1] * len(params)
        elif param_type == 'MVN_fit':
            params, _, _, _ = self.get_weighted_samples_via_MVN()
            prior_weights = [1] * len(params)
        elif param_type == 'statsmodels':
            params, _, _, _ = self.get_weighted_samples_via_statsmodels()
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

        full_output_filename = path.join(self.plot_filename_base,
                                         '{}_param_distro_without_priors.png'.format(param_type))
        if not path.exists(full_output_filename) or self.opt_force_plot:
            try:
                print('Printing...', full_output_filename)
                az.plot_posterior(data,
                                  round_to=3,
                                  credible_interval=0.9,
                                  group='posterior',
                                  var_names=self.plot_param_names)  # show=True allows for plotting within IDE
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

            full_output_filename = path.join(self.plot_filename_base,
                                             '{}_param_distro_with_priors.png'.format(param_type))
            if not path.exists(full_output_filename) or self.opt_force_plot:
                try:
                    print('Printing...', full_output_filename)
                    az.plot_posterior(data,
                                      round_to=3,
                                      credible_interval=0.9,
                                      group='posterior',
                                      var_names=self.plot_param_names)  # show=True allows for plotting within IDE
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

        # Do statsmodels
        self.render_statsmodels_fit()

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='statsmodels')

        # Training Data Bootstraps
        self.render_bootstraps()

        # Plot all-data solution 
        self.solve_and_plot_solution(in_params=self.all_data_params,
                                     title='All-Data Solution',
                                     plot_filename_filename='all_data_solution.png')

        # Plot example solutions from bootstrap
        bootstrap_selection = np.random.choice(self.bootstrap_params)
        self.solve_and_plot_solution(in_params=bootstrap_selection,
                                     title='Random Bootstrap Selection',
                                     plot_filename_filename='random_bootstrap_selection.png')

        # Plot all bootstraps
        self.plot_all_solutions(key='bootstrap')
        self.plot_all_solutions(key='bootstrap_with_priors')

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='bootstrap')

        # Do random walks around the overall fit
        # print('\nSampling around MLE with wide sigma')
        # self.MCMC(self.all_data_params, opt_walk=False,
        #           sample_shape_param=1, which_distro='norm')
        # print('Sampling around MLE with medium sigma')
        # self.MCMC(self.all_data_params, opt_walk=False,
        #           sample_shape_param=10, which_distro='norm')
        # print('Sampling around MLE with narrow sigma')
        # self.MCMC(self.all_data_params, opt_walk=False,
        #           sample_shape_param=100, which_distro='norm')
        # print('Sampling around MLE with ultra-narrow sigma')
        # self.MCMC(self.all_data_params, opt_walk=False,
        #           sample_shape_param=1000, which_distro=WhichDistro.norm)

        print('Sampling around MLE with medium exponential parameter')
        self.MCMC(self.all_data_params, opt_walk=False,
                  sample_shape_param=100, which_distro=WhichDistro.laplace)
        # print('Sampling around MLE with narrow exponential parameter')
        # self.MCMC(self.all_data_params, opt_walk=False,
        #           sample_shape_param=100, which_distro=WhichDistro.laplace)
        # print('Sampling around MLE with ultra-narrow exponential parameter')
        # self.MCMC(self.all_data_params, opt_walk=False,
        #           sample_shape_param=1000, which_distro=WhichDistro.laplace)

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='likelihood_sample')

        # Plot all solutions...
        self.plot_all_solutions(key='likelihood_samples')

        # Next define MVN model on likelihood and fit        
        self.fit_MVN_to_likelihood(opt_walk=False)

        # Plot all solutions...
        self.plot_all_solutions(key='MVN_fit')

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='MVN_fit')

        print('Sampling via random walk MCMC, starting with MLE')  # a random bootstrap selection')
        # bootstrap_selection = np.random.choice(self.n_bootstraps)
        # starting_point = self.bootstrap_params[bootstrap_selection]
        self.MCMC(self.all_data_params, opt_walk=True)

        # print('Sampling via random walk NUTS, starting with MLE')
        # self.NUTS(self.all_data_params)

        # Plot all solutions...
        self.plot_all_solutions(key='random_walk')

        # Next define MVN model on likelihood and fit
        self.fit_MVN_to_likelihood(opt_walk=True)

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(param_type='random_walk')

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
