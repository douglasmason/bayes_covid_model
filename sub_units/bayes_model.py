from sub_units.utils import Stopwatch, ApproxType
import numpy as np
import pandas as pd
from enum import Enum

pd.plotting.register_matplotlib_converters()  # addresses complaints about Timestamp instead of float for plotting x-values
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
matplotlib.use('Agg')
import matplotlib.dates as mdates
import scipy as sp
import joblib
from os import path
from tqdm import tqdm
import os
from scipy.optimize import approx_fprime
from functools import lru_cache, partial
from abc import ABC, abstractmethod

import datetime
import arviz as az
import seaborn as sns
import numdifftools


class WhichDistro(Enum):
    norm = 'norm'
    laplace = 'laplace'
    sphere = 'hypersphere'
    norm_trunc = 'norm_trunc'
    laplace_trunc = 'laplace_trunc'

    def __str__(self):
        return str(self.value)


class BayesModel(ABC):

    @staticmethod
    def FWHM(in_list_locs, in_list_vals):
        '''
        Calculate full width half maximum
        :param in_list_locs: list of locations
        :param in_list_valss: list of values
        :return: 
        '''
        sorted_ind = np.argsort(in_list_locs)
        sorted_locs = in_list_locs[sorted_ind]
        sorted_vals = in_list_vals[sorted_ind]
        peak = max(in_list_vals)
        start = None
        end = None
        for i in range(len(sorted_ind)):
            loc = sorted_locs[i]
            next_loc = sorted_locs[i + 1]
            val = sorted_vals[i]
            next_val = sorted_vals[i + 1]
            if start is not None and val < peak / 2 and next_val >= peak / 2:
                start = (next_loc - loc) / 2
            if end is not None and val >= peak / 2 and next_val < peak / 2:
                end = (next_loc - loc) / 2
        return end - start

    # this fella isn't necessary like other abstractmethods, but optional in a subclass that supports statsmodels solutions
    def render_statsmodels_fit(self):
        pass

    # this fella isn't necessary like other abstractmethods, but optional in a subclass that supports statsmodels solutions
    def render_PyMC3_fit(self):
        pass

    @abstractmethod
    def run_simulation(self, in_params):
        pass

    # this fella isn't necessary like other abstractmethods, but optional in a subclass that supports statsmodels solutions
    def run_fits_simplified(self, in_params):
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
                 opt_force_calc=False,
                 model_type_name=None,
                 plot_param_names=None,
                 opt_simplified=False,
                 log_offset=0.1,
                 # this kwarg became redundant after I filled in zeros with 0.1 in load_data, leave at 0
                 opt_smoothing=True,
                 prediction_window=28 * 3,  # predict three months into the future
                 model_approx_types=[ApproxType.BS, ApproxType.LS, ApproxType.MCMC],
                 plot_two_vals=None,
                 override_max_date_str=None,
                 **kwargs
                 ):

        for key, val in kwargs.items():
            print(f'Adding extra params to attributes... {key}: {val}')
            setattr(self, key, val)

        if load_data_obj is None:
            from sub_units import load_data as load_data_obj

        tmp_dict = load_data_obj.get_state_data(state_name)
        max_date_str = datetime.datetime.strftime(tmp_dict['max_date'], '%Y-%m-%d')
        self.max_date_str = max_date_str

        if hasattr(self, 'moving_window_size'):
            self.min_sol_date = tmp_dict['max_date'] - datetime.timedelta(days=self.moving_window_size)

        self.override_max_date_str = override_max_date_str
        self.plot_two_vals = plot_two_vals
        self.prediction_window = prediction_window
        self.map_approx_type_to_MVN = dict()
        self.model_approx_types = model_approx_types
        self.opt_smoothing = opt_smoothing  # determines whether to smooth results from load_data_obj.get_state_data
        self.log_offset = log_offset
        self.model_type_name = model_type_name
        self.state_name = state_name
        self.n_bootstraps = n_bootstraps
        self.n_likelihood_samples = n_likelihood_samples
        self.burn_in = burn_in
        self.max_date = datetime.datetime.strptime(max_date_str, '%Y-%m-%d')
        self.static_params = static_params

        if self.opt_smoothing:
            smoothing_str = 'smoothed_'
        else:
            smoothing_str = ''

        if override_max_date_str is None:
            hyperparameter_max_date_str = datetime.datetime.today().strftime('%Y-%m-%d')
        else:
            hyperparameter_max_date_str = override_max_date_str

        state_lc = state_name.lower().replace(' ', '_').replace(':', '_')

        self.all_data_fit_filename = path.join('state_all_data_fits',
                                               f"{state_lc}_{smoothing_str}{model_type_name}_max_date_{hyperparameter_max_date_str.replace('-', '_')}.joblib")
        self.bootstrap_filename = path.join('state_bootstraps',
                                            f"{state_lc}_{smoothing_str}{model_type_name}_{n_bootstraps}_bootstraps_max_date_{hyperparameter_max_date_str.replace('-', '_')}.joblib")
        self.likelihood_samples_filename_format_str = path.join('state_likelihood_samples',
                                                                f"{state_lc}_{smoothing_str}{model_type_name}_{{}}_{n_likelihood_samples}_samples_max_date_{hyperparameter_max_date_str.replace('-', '_')}.joblib")
        self.likelihood_samples_from_bootstraps_filename = path.join('state_likelihood_samples',
                                                                     f"{state_lc}_{smoothing_str}{model_type_name}_{n_bootstraps}_bootstraps_likelihoods_max_date_{hyperparameter_max_date_str.replace('-', '_')}.joblib")
        self.PyMC3_filename = path.join('state_PyMC3_traces',
                                        f"{state_lc}_{smoothing_str}{model_type_name}_max_date_{hyperparameter_max_date_str.replace('-', '_')}.joblib")

        if opt_simplified:
            self.plot_subfolder = f'{hyperparameter_max_date_str.replace("-", "_")}_date_{smoothing_str}{model_type_name}_{"_".join(val.value[1] for val in model_approx_types)}'
        else:
            self.plot_subfolder = f'{hyperparameter_max_date_str.replace("-", "_")}_date_{smoothing_str}{model_type_name}_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples'

        self.plot_subfolder = path.join('state_plots', self.plot_subfolder)
        self.plot_filename_base = path.join(self.plot_subfolder,
                                            state_name.lower().replace(' ', '_').replace('.', ''))

        if not os.path.exists('state_all_data_fits'):
            os.mkdir('state_all_data_fits')
        if not os.path.exists('state_bootstraps'):
            os.mkdir('state_bootstraps')
        if not os.path.exists('state_likelihood_samples'):
            os.mkdir('state_likelihood_samples')
        if not os.path.exists('state_PyMC3_traces'):
            os.mkdir('state_PyMC3_traces')
        if not os.path.exists('state_plots'):
            os.mkdir('state_plots')
        if not os.path.exists(self.plot_subfolder):
            os.mkdir(self.plot_subfolder)
        if not os.path.exists(self.plot_filename_base):
            os.mkdir(self.plot_filename_base)

        state_data = load_data_obj.get_state_data(state_name, opt_smoothing=self.opt_smoothing)

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

        n_t_vals = self.max_date_in_days + self.prediction_window
        self.t_vals = np.linspace(-burn_in, n_t_vals, burn_in + n_t_vals + 1)
        # print('min_date', self.min_date)
        # print('max_date_in_days', self.max_date_in_days)
        # print('t_vals', self.t_vals)

        try:
            self.threshold_cases = min(20, self.series_data[-1, 1] * 0.1)
            print(f"Setting cases threshold to {self.threshold_cases} ({self.series_data[-1, 1]} total)")
            self.day_of_threshold_met_case = \
                [i for i, x in enumerate(self.series_data[:, 1]) if x >= self.threshold_cases][0]
        except:
            self.day_of_threshold_met_case = len(self.series_data) - 1
        try:
            self.threshold_deaths = min(20, self.series_data[-1, 2] * 0.1)
            print(f"Setting death threshold to {self.threshold_deaths} ({self.series_data[-1, 2]} total)")
            self.day_of_threshold_met_death = \
                [i for i, x in enumerate(self.series_data[:, 2]) if x >= self.threshold_deaths][0]
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
        self.opt_force_calc = opt_force_calc

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

        positive_dists, deceased_dists, other_errs, sol, positive_vals, deceased_vals, \
        predicted_tested, actual_tested, predicted_dead, actual_dead = precursor_func(
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

        dists_positive, dists_deceased, other_errs, sol, vals_positive, vals_deceased, \
        predicted_tested, actual_tested, predicted_dead, actual_dead = precursor_func(
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

    def fit_curve_via_curve_fit(self,
                                p0,
                                data_tested=None,
                                data_dead=None,
                                tested_indices=None,
                                deaths_indices=None):
        '''
        Given initial parameters, fit the curve with scipy's curve_fit method
        :param p0: initial parameters
        :param data_tested: list of observables (passable since we may want to add jitter)
        :param data_dead: list of observables (passable since we may want to add jitter)
        :param tested_indices: bootstrap indices when applicable
        :param deaths_indices: bootstrap indices when applicable
        :return: optimized parameters as dictionary
        '''

        positive_dists, deceased_dists, other_errs, sol, positive_vals, deceased_vals, \
        predicted_tested, actual_tested, predicted_dead, actual_dead = self._get_log_likelihood_precursor(
            self.test_params)

        inv_deaths_indices = {val: ind for ind, val in enumerate(self.deaths_indices)}
        inv_cases_indices = {val: ind for ind, val in enumerate(self.cases_indices)}

        ind_use = list()
        for x in range(len(predicted_tested) + len(predicted_dead)):
            if x > len(predicted_tested) - 1:
                if deaths_indices is None or deaths_indices is not None and int(x) - len(
                        predicted_tested) in [inv_deaths_indices[tmp_ind] for tmp_ind in deaths_indices]:
                    ind_use.append(int(x))
            else:
                if tested_indices is None or tested_indices is not None and int(x) in [inv_cases_indices[tmp_ind] for tmp_ind in tested_indices]:
                    ind_use.append(int(x))

        # print('ind_use:', ind_use)
        
        data_use = actual_tested + actual_dead
        data_use = [data_use[i] for i in ind_use]
        
        def curve_fit_func(x_list, *params):
            positive_dists, deceased_dists, other_errs, sol, positive_vals, deceased_vals, \
            predicted_tested, actual_tested, predicted_dead, actual_dead = self._get_log_likelihood_precursor(
                self.convert_params_as_list_to_dict(params))
            out_list = list()
            for x in x_list:
                if x > len(predicted_tested) - 1:
                    out_list.append(predicted_dead[int(x) - len(predicted_tested)])
                else:
                    out_list.append(predicted_tested[int(x)])
            return out_list

        params_as_list, cov = sp.optimize.curve_fit(curve_fit_func,
                                                    ind_use,
                                                    data_use,
                                                    p0=self.convert_params_as_dict_to_list(self.test_params),
                                                    bounds=(
                                                        [self.curve_fit_bounds[name][0] for name in self.sorted_names],
                                                        [self.curve_fit_bounds[name][1] for name in self.sorted_names]))

        return self.convert_params_as_list_to_dict(params_as_list), cov

    def fit_curve_exactly_via_least_squares(self,
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
        # cov = results.hess_inv

        # positive_dists, deceased_dists, other_errs, sol, positive_vals, deceased_vals, \
        # predicted_tested, actual_tested, predicted_dead, actual_dead = self._get_log_likelihood_precursor(
        #     self.all_data_params)
        # 
        # def curve_fit_func(x, *params):
        #     positive_dists, deceased_dists, other_errs, sol, positive_vals, deceased_vals, \
        #     predicted_tested, actual_tested, predicted_dead, actual_dead = self._get_log_likelihood_precursor(
        #         self.convert_params_as_list_to_dict(params))
        #     if x > len(predicted_tested):
        #         return predicted_dead[x - len(predicted_tested)]
        #     else:
        #         return predicted_tested[x]
        # 
        # params, cov = sp.optimize.curve_fit(curve_fit_func,
        #                                      list(range(len(predicted_tested) + len(predicted_dead))),
        #                                      actual_tested + actual_dead,
        #                                      p0=self.convert_params_as_dict_to_list(self.all_data_params),
        #                                      bounds=(
        #                                          [self.curve_fit_bounds[name][0] for name in self.sorted_names],
        #                                          [self.curve_fit_bounds[name][1] for name in self.sorted_names]))
        # 
        # params_as_list = results.x

        params_as_dict = {key: params_as_list[i] for i, key in enumerate(self.sorted_names)}

        # # have to compute sigma after the fact
        # sol = self.run_simulation(params_as_dict)
        # 
        # new_positive = sol[1]
        # new_deceased = sol[2]
        # params_as_dict['sigma'] = 0.06 # quick hack

        return params_as_dict

    def get_covariance_matrix(self, in_params):

        p0 = self.convert_params_as_dict_to_list(in_params)

        # hess = numdifftools.Hessian(lambda x: np.exp(self.get_log_likelihood(x)))(p0)
        hess = numdifftools.Hessian(self.get_log_likelihood)(p0)

        # this uses the jacobian approx to the hessian, but I end up with a singular matrix
        # jacobian = numdifftools.Jacobian(self.get_log_likelihood)(p0)
        # hess = jacobian.T @ jacobian
        # eigenw, eigenv = np.linalg.eig(hess)
        # print('hess eigenw:')
        # print(eigenw)

        # print('hess:')
        # print(hess)

        hess = self.remove_sigma_entries_from_matrix(hess)

        cov = np.linalg.inv(-hess)

        print('p0:')
        print(self.convert_params_as_list_to_dict(p0))
        print('hess:')
        print(hess)
        print('cov:')
        print(cov)
        eigenw, eigenv = np.linalg.eig(cov)
        print('orig_eig:')
        print(eigenw)
        # print('orig diagonal elements of cov:')
        # print(self.convert_params_as_list_to_dict(np.diagonal(cov)))

        # get rid of negative eigenvalue contributions to make PSD
        cov = np.zeros(cov.shape)
        for ind, val in enumerate(eigenw):
            vec = eigenv[:, ind]
            if val > 0:
                tmp_contrib = np.outer(vec, vec)
                cov += tmp_contrib * val

        cov = self.recover_sigma_entries_from_matrix(cov)

        eigenw, eigenv = np.linalg.eig(cov)
        # print('new_cov:')
        # print(cov)
        print('new_eig:')
        print(eigenw)
        print('new diagonal elements of cov:')
        print(self.convert_params_as_list_to_dict(np.diagonal(cov)))

        return cov

    def fit_curve_via_likelihood(self,
                                 in_params,
                                 tested_indices=None,
                                 deaths_indices=None,
                                 method=None,
                                 print_success=False,
                                 opt_cov=False
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

        if method is None:
            method = self.optimizer_method

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

        if opt_cov:
            cov = self.get_covariance_matrix(p0)
        else:
            cov = None  # results.hess_inv # this fella only works for certain methods so avoid for now

        return params_as_dict, cov

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
        max_plot_pt = min(len(sol[0]), len(self.series_data) + self.prediction_window + self.burn_in)
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

    def plot_all_solutions(self, n_samples=1000, approx_type=ApproxType.BS, mvn_fit=False, n_sols_to_plot=1000,
                           offset=0):
        '''
        Plot all the bootstrap simulation solutions
        :param n_sols_to_plot: how many simulations should we sample for the plot?
        :return: None
        '''

        if offset == 0:
            offset_str = ''
        else:
            offset_str = f'_offset_{offset}_days'

        key = approx_type.value[1]
        output_filename = f'{key}{offset_str}_solutions_discrete.png'
        output_filename2 = f'{key}{offset_str}_solutions_filled_quantiles.png'
        output_filename3 = f'{key}{offset_str}_solutions_cumulative_discrete.png'
        output_filename4 = f'{key}{offset_str}_solutions_cumulative_filled_quantiles.png'
        if all(path.exists(path.join(self.plot_filename_base, x)) for x in \
               [output_filename, output_filename2, output_filename3, output_filename4]) and not self.opt_force_plot:
            return

        params, _, _, log_probs = self.get_weighted_samples(approx_type=approx_type, mvn_fit=mvn_fit)
        param_inds_to_plot = list(range(len(params)))

        # Put this in to diagnose plotting
        # distro_list = list()
        # for param_ind in range(len(params[0])):
        #     distro_list.append([params[i][param_ind] for i in range(len(params))])
        #     print(f'{self.sorted_names[param_ind]}: Mean: {np.average(distro_list[param_ind]):.4g}, Std.: {np.std(distro_list[param_ind]):.4g}')

        print(f'Rendering solutions for {key}...')
        param_inds_to_plot = np.random.choice(param_inds_to_plot, min(n_samples, len(param_inds_to_plot)),
                                              replace=False)
        sols_to_plot = [self.run_simulation(in_params=params[param_ind], offset=offset) for param_ind in
                        tqdm(param_inds_to_plot)]

        data_plot_kwargs = {'markersize': 6, 'markeredgewidth': 0.5, 'markeredgecolor': 'black'}
        self._plot_all_solutions_sub_distinct_lines_with_alpha(sols_to_plot,
                                                               plot_filename_filename=output_filename,
                                                               data_plot_kwargs=data_plot_kwargs,
                                                               offset=offset)
        self._plot_all_solutions_sub_filled_quantiles(sols_to_plot,
                                                      plot_filename_filename=output_filename2,
                                                      data_plot_kwargs=data_plot_kwargs,
                                                      offset=offset)
        self._plot_all_solutions_sub_distinct_lines_with_alpha_cumulative(sols_to_plot,
                                                                          plot_filename_filename=output_filename3,
                                                                          data_plot_kwargs=data_plot_kwargs)
        self._plot_all_solutions_sub_filled_quantiles_cumulative(sols_to_plot,
                                                                 plot_filename_filename=output_filename4,
                                                                 data_plot_kwargs=data_plot_kwargs)

    def _plot_all_solutions_sub_filled_quantiles(self,
                                                 sols_to_plot,
                                                 plot_filename_filename=None,
                                                 data_markersize=36,
                                                 data_plot_kwargs=dict(),
                                                 offset=0):
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
        sol = sols_to_plot[0]
        fig, ax = plt.subplots()
        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + self.prediction_window + self.burn_in)
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
                if sol_plot_date_range[i] >= self.min_sol_date - datetime.timedelta(days=offset):
                    min_slice = i
                    break

        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p5_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p95_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        facecolor=matplotlib.colors.colorConverter.to_rgba('red', alpha=0.3),
                        edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                        )
        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p25_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p75_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        facecolor=matplotlib.colors.colorConverter.to_rgba('red', alpha=0.6),
                        edgecolor=(0, 0, 0, 0)  # r=get rid of the darker edge
                        )
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
                        facecolor=matplotlib.colors.colorConverter.to_rgba('green', alpha=0.3),
                        edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                        )
        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p25_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p75_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        facecolor=matplotlib.colors.colorConverter.to_rgba('green', alpha=0.6),
                        edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                        )
        ax.plot(sol_plot_date_range[slice(min_slice, None)], p50_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                color="darkgreen")

        ax.plot(data_plot_date_range, self.data_new_tested, '.', color='darkgreen', label='Infections',
                **data_plot_kwargs)
        ax.plot(data_plot_date_range, self.data_new_dead, '.', color='darkred', label='Deaths', **data_plot_kwargs)
        fig.autofmt_xdate()

        # this removes the year from the x-axis ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        plt.yscale('log')
        plt.ylabel('Daily Reported Counts')
        plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_threshold_met_case - 10), None))
        plt.ylim((0.1, max(self.data_new_tested) * 100))
        plt.legend()
        # plt.title(f'{state} Data (points) and Model Predictions (lines)')
        plt.savefig(full_output_filename, dpi=self.plot_dpi)
        plt.close()

    def _plot_all_solutions_sub_filled_quantiles_cumulative(self,
                                                            sols_to_plot,
                                                            plot_filename_filename=None,
                                                            opt_predict=True,
                                                            data_plot_kwargs=dict()):
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
        sol = sols_to_plot[0]
        fig, ax = plt.subplots()
        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + self.prediction_window + self.burn_in)
        data_plot_date_range = [self.min_date + datetime.timedelta(days=1) * i for i in
                                range(len(self.data_new_tested))]

        sol_plot_date_range = [self.min_date - datetime.timedelta(days=self.burn_in) + datetime.timedelta(
            days=1) * i for i in
                               range(len(sol[0]))][min_plot_pt:max_plot_pt]

        map_t_val_ind_to_tested_distro = dict()
        map_t_val_ind_to_deceased_distro = dict()
        for sol in sols_to_plot:

            if opt_predict:
                start_ind_sol = len(self.data_new_tested) + self.burn_in
            else:
                start_ind_sol = len(self.data_new_tested) + self.burn_in - self.moving_window_size

            start_ind_data = start_ind_sol - 1 - self.burn_in

            tested = [max(sol[1][i] - self.log_offset, 0) for i in range(len(sol[1]))]
            tested_range = np.cumsum(tested[start_ind_sol:])

            dead = [max(sol[2][i] - self.log_offset, 0) for i in range(len(sol[2]))]
            dead_range = np.cumsum(dead[start_ind_sol:])

            data_tested_at_start = np.cumsum(self.data_new_tested)[start_ind_data]
            data_dead_at_start = np.cumsum(self.data_new_dead)[start_ind_data]

            tested = [0] * start_ind_sol + [data_tested_at_start + tested_val for tested_val in tested_range]
            dead = [0] * start_ind_sol + [data_dead_at_start + dead_val for dead_val in dead_range]

            for val_ind, val in enumerate(self.t_vals):
                if val_ind not in map_t_val_ind_to_tested_distro:
                    map_t_val_ind_to_tested_distro[val_ind] = list()
                map_t_val_ind_to_tested_distro[val_ind].append(tested[val_ind])
                if val_ind not in map_t_val_ind_to_deceased_distro:
                    map_t_val_ind_to_deceased_distro[val_ind] = list()
                map_t_val_ind_to_deceased_distro[val_ind].append(dead[val_ind])

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
        if opt_predict:
            use_min_sol_date = self.max_date
        else:
            use_min_sol_date = self.min_sol_date
        if use_min_sol_date is not None:
            for i in range(len(sol_plot_date_range)):
                if sol_plot_date_range[i] >= use_min_sol_date:
                    min_slice = i
                    break

        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p5_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p95_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        facecolor=matplotlib.colors.colorConverter.to_rgba('red', alpha=0.3),
                        edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                        )
        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p25_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p75_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        facecolor=matplotlib.colors.colorConverter.to_rgba('red', alpha=0.6),
                        edgecolor=(0, 0, 0, 0)  # r=get rid of the darker edge
                        )
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
                        facecolor=matplotlib.colors.colorConverter.to_rgba('green', alpha=0.3),
                        edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                        )
        ax.fill_between(sol_plot_date_range[slice(min_slice, None)],
                        p25_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        p75_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                        facecolor=matplotlib.colors.colorConverter.to_rgba('green', alpha=0.6),
                        edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                        )
        ax.plot(sol_plot_date_range[slice(min_slice, None)], p50_curve[min_plot_pt:max_plot_pt][slice(min_slice, None)],
                color="darkgreen")

        ax.plot(data_plot_date_range, np.cumsum(self.data_new_tested), '.', color='darkgreen', label='Infections',
                **data_plot_kwargs)
        ax.plot(data_plot_date_range, np.cumsum(self.data_new_dead), '.', color='darkred', label='Deaths',
                **data_plot_kwargs)
        fig.autofmt_xdate()

        # this removes the year from the x-axis ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        plt.yscale('log')
        plt.ylabel('Cumulative Reported Counts')
        plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_threshold_met_case - 10), None))
        plt.ylim((1, sum(self.data_new_tested) * 100))
        plt.legend()
        # plt.title(f'{state} Data (points) and Model Predictions (lines)')
        plt.savefig(full_output_filename, dpi=self.plot_dpi)
        plt.close()

    def _plot_all_solutions_sub_distinct_lines_with_alpha(self,
                                                          sols_to_plot,
                                                          n_sols_to_plot=1000,
                                                          plot_filename_filename=None,
                                                          data_plot_kwargs=dict(),
                                                          offset=0):
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
        sol = sols_to_plot[0]
        n_sols = len(sols_to_plot)
        fig, ax = plt.subplots()
        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + self.prediction_window + self.burn_in)
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
                    if sol_plot_date_range[i] >= self.min_sol_date - datetime.timedelta(days=offset):
                        min_slice = i
                        break
            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    new_tested[min_plot_pt:max_plot_pt][slice(min_slice, None)], 'g',
                    alpha=5 / n_sols)
            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    new_dead[min_plot_pt:max_plot_pt][slice(min_slice, None)], 'r',
                    alpha=5 / n_sols)

        ax.plot(data_plot_date_range, self.data_new_tested, '.', color='darkgreen', label='Infections',
                **data_plot_kwargs)
        ax.plot(data_plot_date_range, self.data_new_dead, '.', color='darkred', label='Deaths', **data_plot_kwargs)
        fig.autofmt_xdate()

        # this removes the year from the x-axis ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        plt.yscale('log')
        plt.ylabel('Daily Reported Counts')
        plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_threshold_met_case - 10), None))
        plt.ylim((0.1, max(self.data_new_tested) * 100))
        plt.legend()
        # plt.title(f'{state} Data (points) and Model Predictions (lines)')
        plt.savefig(full_output_filename, dpi=self.plot_dpi)
        plt.close()

    def _plot_all_solutions_sub_distinct_lines_with_alpha_cumulative(self,
                                                                     sols_to_plot,
                                                                     n_sols_to_plot=1000,
                                                                     plot_filename_filename=None,
                                                                     opt_predict=True,
                                                                     data_plot_kwargs=dict()):
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
        sol = sols_to_plot[0]
        n_sols = len(sols_to_plot)
        fig, ax = plt.subplots()
        min_plot_pt = self.burn_in
        max_plot_pt = min(len(sol[0]), len(self.series_data) + self.prediction_window + self.burn_in)
        data_plot_date_range = [self.min_date + datetime.timedelta(days=1) * i for i in
                                range(len(self.data_new_tested))]

        for sol in sols_to_plot:

            if opt_predict:
                start_ind_sol = len(self.data_new_tested) + self.burn_in
            else:
                start_ind_sol = len(self.data_new_tested) + self.burn_in - self.moving_window_size

            start_ind_data = start_ind_sol - 1 - self.burn_in

            tested = [max(sol[1][i] - self.log_offset, 0) for i in range(len(sol[1]))]
            tested_range = np.cumsum(tested[start_ind_sol:])

            dead = [max(sol[2][i] - self.log_offset, 0) for i in range(len(sol[2]))]
            dead_range = np.cumsum(dead[start_ind_sol:])

            data_tested_at_start = np.cumsum(self.data_new_tested)[start_ind_data]
            data_dead_at_start = np.cumsum(self.data_new_dead)[start_ind_data]

            tested = [0] * start_ind_sol + [data_tested_at_start + tested_val for tested_val in tested_range]
            dead = [0] * start_ind_sol + [data_dead_at_start + dead_val for dead_val in dead_range]

            sol_plot_date_range = [self.min_date - datetime.timedelta(days=self.burn_in) + datetime.timedelta(
                days=1) * i for i in
                                   range(len(sol[0]))][min_plot_pt:max_plot_pt]

            # ax.plot(plot_date_range[min_plot_pt:], [(sol[i][0]) for i in range(min_plot_pt, len(sol[0))], 'b', alpha=0.1)
            # ax.plot(plot_date_range[min_plot_pt:max_plot_pt], [(sol[i][1]) for i in range(min_plot_pt, max_plot_pt)], 'g', alpha=0.1)

            min_slice = None
            if opt_predict:
                use_min_sol_date = self.max_date
            else:
                use_min_sol_date = self.min_sol_date
            if use_min_sol_date is not None:
                for i in range(len(sol_plot_date_range)):
                    if sol_plot_date_range[i] >= use_min_sol_date:
                        min_slice = i
                        break

            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    tested[min_plot_pt:max_plot_pt][slice(min_slice, None)], 'g',
                    alpha=5 / n_sols)
            ax.plot(sol_plot_date_range[slice(min_slice, None)],
                    dead[min_plot_pt:max_plot_pt][slice(min_slice, None)], 'r',
                    alpha=5 / n_sols)

        ax.plot(data_plot_date_range, np.cumsum(self.data_new_tested), '.', color='darkgreen', label='Infections',
                **data_plot_kwargs)
        ax.plot(data_plot_date_range, np.cumsum(self.data_new_dead), '.', color='darkred', label='Deaths',
                **data_plot_kwargs)
        fig.autofmt_xdate()

        # this removes the year from the x-axis ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
        plt.yscale('log')
        plt.ylabel('Cumulative Reported Counts')
        plt.xlim((self.min_date + datetime.timedelta(days=self.day_of_threshold_met_case - 10), None))
        plt.ylim((1, sum(self.data_new_tested) * 100))
        plt.legend()
        # plt.title(f'{state} Data (points) and Model Predictions (lines)')
        plt.savefig(full_output_filename, dpi=self.plot_dpi)
        plt.close()

    @staticmethod
    def norm_2d(xv, yv, mu=(0, 0), sigma=(1, 1)):
        arg = -((xv - mu[0]) ** 2 / sigma[0] + (yv - mu[1]) ** 2 / sigma[1])
        vals = np.exp(arg)
        return vals

    def render_all_data_fit(self,
                            passed_params=None,
                            method='curve_fit',  # orig, curve_fit
                            ):

        '''
        Compute the all-data MLE/MAP
        :param params: dictionary of parameters to replace the usual fit, if necessary
        :return: None
        '''

        success = False

        try:
            all_data_dict = joblib.load(self.all_data_fit_filename)
            all_data_params = all_data_dict['all_data_params']
            all_data_sol = all_data_dict['all_data_sol']
            all_data_cov = all_data_dict['all_data_cov']
            success = True
            self.loaded_all_data_fit = True
        except:
            self.loaded_all_data_fit = False

        if passed_params is not None:
            all_data_params = passed_params
            all_data_sol = self.run_simulation(passed_params)
            all_data_cov = self.get_covariance_matrix(passed_params)

        # TODO: Break out all-data fit to its own method, not embedded in render_bootstraps
        if (not success and self.opt_calc and passed_params is None) or self.opt_force_calc:

            print('\n----\nRendering all-data model fits... \n----')

            # This is kind of a kludge, I find more reliable fits with fit_curve_exactly_with_jitter
            #   But it doesn't fit the observation error, which I need for likelihood samples
            #   So I use it to fit everything BUT observation error, then insert the test_params entries for the sigmas,
            #   and re-fit using the jankier (via_likelihood) method that fits the observation error
            #   TODO: make the sigma substitutions empirical, rather than hacky the way I've done it
            if method == 'orig':

                test_params_as_list = [self.test_params[key] for key in self.sorted_names]
                all_data_params, _ = self.fit_curve_exactly_via_least_squares(test_params_as_list)
                all_data_params['sigma_positive'] = self.test_params['sigma_positive']
                all_data_params['sigma_deceased'] = self.test_params['sigma_deceased']
                print('refitting all-data params to get sigma values')
                all_data_params_for_sigma, all_data_cov = self.fit_curve_via_likelihood(all_data_params,
                                                                                        print_success=True,
                                                                                        opt_cov=True)

                print('\nOrig params:')
                self.pretty_print_params(all_data_params)
                print('\nRe-fit params for sigmas:')
                self.pretty_print_params(all_data_params_for_sigma)

                for key in all_data_params:
                    if 'sigma' in key:
                        print(f'Stealing value for {key}: {all_data_params_for_sigma[key]}')
                        all_data_params[key] = all_data_params_for_sigma[key]

            elif method == 'curve_fit':

                print('Employing Scipy\'s curve_fit method...')

                test_params_as_list = [self.test_params[key] for key in self.sorted_names]
                all_data_params, all_data_cov = self.fit_curve_via_curve_fit(test_params_as_list)

                print('refitting all-data params to get sigma values')
                all_data_params_for_sigma, all_data_cov_for_sigma = self.fit_curve_via_likelihood(all_data_params,
                                                                                                  print_success=True,
                                                                                                  opt_cov=True)

                print('\nOrig params:')
                self.pretty_print_params(all_data_params)
                print('\nRe-fit params for sigmas:')
                self.pretty_print_params(all_data_params_for_sigma)

                print('\nOrig std-errs:')
                self.pretty_print_params(np.sqrt(np.diagonal(all_data_cov)))
                self.plot_correlation_matrix(self.cov2corr(all_data_cov), filename_str='curve_fit')
                print('\nRe-fit std-errs for sigmas:')
                self.pretty_print_params(np.sqrt(np.diagonal(all_data_cov_for_sigma)))
                self.plot_correlation_matrix(self.cov2corr(all_data_cov_for_sigma), filename_str='numdifftools')
                
                for ind, name in enumerate(self.sorted_names):
                    if 'sigma' in name:
                        all_data_cov[ind,:] = all_data_cov_for_sigma[ind,:]
                        all_data_cov[:, ind] = all_data_cov_for_sigma[:, ind]

                for key in all_data_params:
                    if 'sigma' in key:
                        print(f'Stealing value for {key}: {all_data_params_for_sigma[key]}')
                        all_data_params[key] = all_data_params_for_sigma[key]

            all_data_sol = self.run_simulation(all_data_params)
            print('\nParameters when trained on all data (this is our starting point for optimization):')
            self.pretty_print_params(all_data_params)

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
                    # all_data_params2, cov = self.fit_curve_via_likelihood(test_params_as_list,
                    #                                                       method=method, print_success=True)
                    all_data_params2, _ = self.fit_curve_exactly_via_least_squares(test_params_as_list,
                                                                                   method=method, print_success=True)

                    print(f'\nParameters when trained on all data using method {method}:')
                    [print(f'{key}: {val:.4g}') for key, val in all_data_params2.items()]
                except:
                    print(f'method {method} failed!')

        # Add deterministic parameters to all-data solution
        for extra_param, extra_param_func in self.extra_params.items():
            all_data_params[extra_param] = extra_param_func(
                [all_data_params[name] for name in self.sorted_names])

        print(f'saving bootstraps to {self.all_data_fit_filename}...')
        joblib.dump({'all_data_sol': all_data_sol, 'all_data_params': all_data_params, 'all_data_cov': all_data_cov},
                    self.all_data_fit_filename)
        print('...done!')

        self.all_data_params = all_data_params
        self.all_data_sol = all_data_sol
        self.all_data_cov = all_data_cov
        print('all_data_params:')
        print(all_data_params)
        print('all_data_cov:')
        print(all_data_cov)

        self.hessian_model = sp.stats.multivariate_normal(mean=self.convert_params_as_dict_to_list(all_data_params),
                                                          cov=all_data_cov, allow_singular=True)

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
            success = True
            self.loaded_bootstraps = True
        except:
            self.loaded_bootstraps = False

        # TODO: Break out all-data fit to its own method, not embedded in render_bootstraps
        if (not success and self.opt_calc) or self.opt_force_calc:

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
                starting_point_as_list = [self.all_data_params[key] for key in self.sorted_names]

                # params_as_dict, cov = self.fit_curve_via_likelihood(starting_point_as_list,
                #                                                # data_tested=tested_jitter,
                #                                                # data_dead=dead_jitter,
                #                                                tested_indices=cases_bootstrap_indices,
                #                                                deaths_indices=deaths_bootstrap_indices
                #                                                )

                # params_as_dict = self.fit_curve_exactly_via_least_squares(starting_point_as_list,
                #                                                           # data_tested=tested_jitter,
                #                                                           # data_dead=dead_jitter,
                #                                                           tested_indices=cases_bootstrap_indices,
                #                                                           deaths_indices=deaths_bootstrap_indices
                #                                                           )
                params_as_dict, cov = self.fit_curve_via_curve_fit(starting_point_as_list,
                                                                   # data_tested=tested_jitter,
                                                                   # data_dead=dead_jitter,
                                                                   tested_indices=cases_bootstrap_indices,
                                                                   deaths_indices=deaths_bootstrap_indices
                                                                   )

                sol = self.run_simulation(params_as_dict)
                bootstrap_sols.append(sol)
                bootstrap_params.append(params_as_dict)

            print(f'saving bootstraps to {self.bootstrap_filename}...')
            joblib.dump({'bootstrap_sols': bootstrap_sols, 'bootstrap_params': bootstrap_params},
                        self.bootstrap_filename)
            print('...done!')

        print('\nParameters when trained on all data (this is our starting point for optimization):')
        [print(f'{key}: {val:.4g}') for key, val in self.all_data_params.items()]

        # Add deterministic parameters to bootstraps
        for params in bootstrap_params:
            for extra_param, extra_param_func in self.extra_params.items():
                params[extra_param] = extra_param_func([params[name] for name in self.sorted_names])

        self.bootstrap_sols = bootstrap_sols
        self.bootstrap_params = bootstrap_params

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

        if (not success and self.opt_calc) or self.opt_force_calc:

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
            self.all_PyMC3_samples_as_list += samples
            self.all_PyMC3_log_probs_as_list += vals

            shuffled_ind = list(range(len(self.all_PYMC3_samples_as_list)))
            np.random.shuffle(shuffled_ind)
            self.all_PyMC3_samples_as_list = [self.convert_params_as_dict_to_list(self.all_PYMC3_samples_as_list[i]) for
                                              i in shuffled_ind]
            self.all_PyMC3_log_probs_as_list = [self.all_PYMC3_log_probs_as_list[i] for i in shuffled_ind]
            print('...done!')

    @lru_cache(maxsize=10)
    def get_propensity_model(self, sample_scale_param, which_distro=WhichDistro.norm, opt_walk=False):
        
        if sample_scale_param == 'empirical':
            cov = self.all_data_cov
            for ind, name in enumerate(self.sorted_names):
                if name in self.logarithmic_params:
                    cov[ind, :] = cov[ind, :] / self.all_data_params[name]
                    cov[:, ind] = cov[:, ind] / self.all_data_params[name]
            if opt_walk:
                cov = cov / 100
            propensity_model = sp.stats.multivariate_normal(cov=cov, allow_singular=True)
        else:
        
            sigma = {key: (val[1] - val[0]) / sample_scale_param for key, val in self.curve_fit_bounds.items()}
    
            # overwrite sigma for values that are strictly positive multipliers of unknown scale
            for param_name in self.logarithmic_params:
                sigma[
                    param_name] = 10 / sample_scale_param  # Note that I boost the width by a factor of 10 since it often gets too narrow
            sigma_as_list = [sigma[name] for name in self.sorted_names]
    
            if which_distro in [WhichDistro.norm, WhichDistro.norm_trunc, WhichDistro.sphere]:
                cov = np.diag([max(1e-8, x ** 2) for x in sigma_as_list])
                propensity_model = sp.stats.multivariate_normal(cov=cov)
            elif which_distro in [WhichDistro.laplace, WhichDistro.laplace_trunc]:
                propensity_model = sp.stats.laplace(scale=sigma_as_list)
    
        return propensity_model

    def MCMC(self, p0, opt_walk=True,
             sample_shape_param='empirical', # or an integer like 10 or 100
             which_distro=WhichDistro.norm 
             ):
        n_samples = self.n_likelihood_samples
        if opt_walk:
            MCMC_burn_in_frac = 0.2
            if which_distro != WhichDistro.norm:
                raise ValueError
        else:
            MCMC_burn_in_frac = 0

        bounds_to_use = self.curve_fit_bounds
        
        print('sigmas...')
        self.pretty_print_params(np.sqrt(np.diagonal(self.all_data_cov)))

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
            print('offending parameters:')
            print(offending_params)
            print('parameters')
            self.pretty_print_params(p0)
            return

        def acquisition_function(input_params, sample_shape_param):
            output_params = input_params
            jitter_propensity = 1
            accepted = False
            n_attempts = 0

            propensity_model = self.get_propensity_model(sample_shape_param, which_distro=which_distro, opt_walk=opt_walk)

            while n_attempts < 100 and not accepted:  # this limits endless searches with wide sigmas
                n_attempts += 1

                if which_distro in [WhichDistro.norm, WhichDistro.laplace]:
                    # acq_timer=Stopwatch()
                    jitter = propensity_model.rvs()
                    # sample_time = acq_timer.elapsed_time()
                    # print(f'Sample time: {sample_time * 1000:.4g} ms')
                    # acq_timer.reset()
                    jitter_propensity = np.prod(propensity_model.pdf(jitter))
                    # pdf_time = acq_timer.elapsed_time()
                    # print(f'PDF time: {pdf_time * 1000:.4g} ms')
                    # print(f'n_attempts: {n_attempts}')
                elif which_distro in [WhichDistro.sphere, WhichDistro.norm_trunc, WhichDistro.laplace_trunc]:
                    accepted = False
                    n_tries = 0
                    while not accepted:
                        jitter = propensity_model.rvs()  # this will be multivariate norm
                        jitter_propensity = np.prod(propensity_model.pdf(jitter))
                        center_propensity = np.prod(propensity_model.pdf(np.zeros_like(jitter)))

                        # the idea here is to discard points that have low propensities, which approximates
                        #   to the hypersphere as cheaply as I could figure out
                        # print(f'jitter_propensity/jitter_propensity: {jitter_propensity/center_propensity:.4g}')
                        if jitter_propensity > 0.01 * center_propensity:
                            accepted = True
                            # print('accepted')
                        else:
                            n_tries += 1
                            # print('rejected')
                        if n_tries > 1000:
                            raise ValueError('Discarding too many points, please tune your acceptance rate...')

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

        filename_str += f'_{which_distro}_sample_shape_param_{sample_shape_param}'

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
        use_sample_shape_param = sample_shape_param

        if (not success and self.opt_calc) or self.opt_force_calc:

            for test_ind in tqdm(range(n_samples)):

                # sub_timer = Stopwatch()
                proposed_p, proposed_propensity = acquisition_function(prev_p, use_sample_shape_param)
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
                            use_sample_shape_param = sample_shape_param * 100

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
            print(f'Dumping to {self.likelihood_samples_filename_format_str.format(filename_str)}...')
            joblib.dump({'samples': samples, 'vals': log_probs, 'propensities': propensities},
                        self.likelihood_samples_filename_format_str.format(filename_str))
            print('...done!')

        samples_as_list = [[sample[key] for key in self.sorted_names] for sample in samples]
        MCMC_burn_in = int(MCMC_burn_in_frac * len(samples_as_list))

        if opt_walk:
            samples_key = 'random_walk'
        else:
            samples_key = 'likelihood_samples'
        self._add_samples(samples_as_list[MCMC_burn_in:], log_probs[MCMC_burn_in:], propensities[MCMC_burn_in:],
                          key=samples_key)

    def remove_sigma_entries_from_matrix(self, in_matrix):

        sigma_inds = [i for i, name in enumerate(self.sorted_names) if 'sigma' in name]
        normal_inds = [i for i in range(len(self.sorted_names)) if i not in sigma_inds]

        in_matrix = in_matrix[normal_inds, :]
        in_matrix = in_matrix[:, normal_inds]

        return in_matrix

    def recover_sigma_entries_from_matrix(self, in_matrix):

        sigma_inds = [i for i, name in enumerate(self.sorted_names) if 'sigma' in name]
        normal_inds = [i for i in range(len(self.sorted_names)) if i not in sigma_inds]

        # fill back in sigma entries
        in_matrix_with_sigmas = np.eye(len(self.sorted_names), len(self.sorted_names)) * 1e-8
        for ind_i, i in enumerate(normal_inds):
            for ind_j, j in enumerate(normal_inds):
                in_matrix_with_sigmas[i, j] = in_matrix[ind_i, ind_j]
        return in_matrix_with_sigmas

    def cov2corr(self, cov, opt_replace_sigma=False):
        if opt_replace_sigma:
            cov = self.remove_sigma_entries_from_matrix(cov)
        std_devs_mat = np.diag(np.sqrt(np.diagonal(cov)))
        cov_inv = np.linalg.inv(std_devs_mat)
        corr = cov_inv @ cov @ cov_inv
        if opt_replace_sigma:
            corr = self.recover_sigma_entries_from_matrix(corr)
        return corr

    def get_weighted_samples_via_hessian(self, n_samples=1000, ):
        '''
        Retrieves likelihood samples in parameter space, weighted by their standard errors from statsmodels
        :param n_samples: how many samples to re-sample from the list of likelihood samples
        :return: tuple of weight_sampled_params, params, weights, log_probs
        '''

        weight_sampled_params = self.hessian_model.rvs(n_samples)

        log_probs = [self.get_log_likelihood(x) for x in weight_sampled_params]

        return weight_sampled_params, weight_sampled_params, [1] * len(weight_sampled_params), log_probs

    def get_weighted_samples(self, approx_type=ApproxType.BS, mvn_fit=False):
        if not mvn_fit:
            if approx_type == ApproxType.BS:
                params = self.bootstrap_params
                log_probs = [self.get_log_likelihood(x) for x in params]
                weights = [1] * len(params)
                weighted_params = params
            elif approx_type == ApproxType.LS:
                weighted_params, params, weights, log_probs = self.get_weighted_samples_via_direct_sampling()
            elif approx_type == ApproxType.MCMC:
                params = self.all_random_walk_samples_as_list
                log_probs = [self.get_log_likelihood(x) for x in params]
                weights = [1] * len(params)
                weighted_params = params
            elif approx_type == ApproxType.SM:
                weighted_params, params, weights, log_probs = self.get_weighted_samples_via_statsmodels()
            elif approx_type == ApproxType.PyMC3:
                weighted_params, params, weights, log_probs = self.get_weighted_samples_via_PyMC3()
            elif approx_type == ApproxType.Hess:
                weighted_params, params, weights, log_probs = self.get_weighted_samples_via_hessian()
            else:
                raise ValueError
        else:
            weighted_params, params, weights, log_probs = self.get_weighted_samples_via_MVN(approx_type=approx_type)

        weighted_params = [self.convert_params_as_dict_to_list(x) for x in weighted_params]
        params = [self.convert_params_as_dict_to_list(x) for x in params]
        return weighted_params, params, weights, log_probs

    def plot_correlation_matrix(self,
                                corr,
                                filename_str='test'):

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

    def fit_MVN_to_likelihood(self,
                              cov_type='full',
                              approx_type=ApproxType.LS):
        filename_str = f'MVN_{approx_type.value[1]}'

        weighted_params, params, weights, log_probs = self.get_weighted_samples(approx_type=approx_type, mvn_fit=False)

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
        print(f'means for {approx_type.value[1]}:', means_as_list)
        print(f'std_devs {approx_type.value[1]}:', std_devs_as_list)

        self.solve_and_plot_solution(in_params=means,
                                     title=f'Mean of {filename_str.replace("_", " ")} Fit Solution',
                                     plot_filename_filename=f'mean_of_{filename_str}_solution.png')

        if sum(std_devs_as_list) < 1e-6:
            cov = np.diag([1e-8 for _ in std_devs_as_list])
        else:
            if cov_type == 'diag':
                cov = np.diag([x ** 2 for x in std_devs_as_list])
            else:
                cov = np.cov(np.vstack(params).T,
                             aweights=weights)

        # If calculating the correlation matrix is an issue, just get rid of the sigma entries
        try:
            corr = self.cov2corr(cov)
        except:
            corr = self.cov2corr(cov, opt_replace_sigma=True)

        print('corr:', corr)
        model = sp.stats.multivariate_normal(mean=means_as_list, cov=cov, allow_singular=True)

        conf_ints = dict()
        for param_name, std_dev in std_devs.items():
            mu = means[param_name]
            lower = float(mu - std_dev * 1.645)
            upper = float(mu + std_dev * 1.645)
            conf_ints[param_name] = (lower, upper)
            print(
                f'{approx_type.value[1]}: Param {param_name} mean and 90% conf. int.: {mu:.4g} ({lower:.4g}, {upper:.4g})')

        predicted_vals = model.pdf(np.vstack(params))

        # Pairplot!
        full_output_filename = path.join(self.plot_filename_base, f'{filename_str}_pairplot.png')
        if not path.exists(full_output_filename) or self.opt_force_plot:
            weighted_params_as_df = pd.DataFrame(weighted_params)
            weighted_params_as_df.columns = self.sorted_names
            weighted_params_as_df = weighted_params_as_df[
                [x for x in weighted_params_as_df.columns if 'sigma' not in x and 'multiplier' not in x]]
            try:
                plt.clf()
                print('weighted_params_as_df')
                print(weighted_params_as_df)
                sns.pairplot(weighted_params_as_df, markers="+", palette="husl", diag_kind="kde")
                plt.subplots_adjust(left=0.1, bottom=0.1)  # add to left margin
                plt.savefig(full_output_filename, dpi=self.plot_dpi)
                plt.close()
            except Exception as ee:
                print(f'Error printing pairplot: "{ee}"')

        print(
            f'len(log_probs): {len(log_probs)}; len(params): {len(params)}; len(predicted_vals): {len(predicted_vals)}')
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

        if self.plot_two_vals is not None:
            val0 = [weighted_params[i][self.map_name_to_sorted_ind[self.plot_two_vals[0]]] for i in
                    range(len(weighted_params))]
            val1 = [weighted_params[i][self.map_name_to_sorted_ind[self.plot_two_vals[1]]] for i in
                    range(len(weighted_params))]
            full_output_filename = path.join(self.plot_filename_base,
                                             f'{filename_str}_{self.plot_two_vals[1]}_vs_{self.plot_two_vals[0]}.png')
            if not path.exists(full_output_filename) or self.opt_force_plot:
                plt.clf()
                plt.plot(val0, val1, '.',
                         alpha=max(100 / len(val0), 0.01))
                plt.xlabel(self.plot_two_vals[0])
                plt.ylabel(self.plot_two_vals[1])
                if self.plot_two_vals[0] in self.logarithmic_params:
                    plt.xscale('log')
                if self.plot_two_vals[1] in self.logarithmic_params:
                    plt.yscale('log')
                plt.savefig(full_output_filename, dpi=self.plot_dpi)
                plt.close()

        # full_output_filename = path.join(self.plot_filename_base, f'{filename_str}_actual_vs_predicted_vals_linear.png')
        # if not path.exists(full_output_filename) or self.opt_force_plot:
        #     plt.clf()
        #     plt.plot(np.exp(log_probs), predicted_vals, '.',
        #              alpha=max(100 / len(predicted_vals), 0.01))
        #     plt.xlabel('actual values')
        #     plt.ylabel('predicted values')
        #     plt.savefig(full_output_filename, dpi=self.plot_dpi)
        #     plt.close()

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

        tmp_dict = {'model': model, 'conf_ints': conf_ints, 'means': means, 'std_devs': std_devs,
                    'cov': cov, 'corr': corr}
        self.map_approx_type_to_MVN[approx_type] = tmp_dict

    def get_weighted_samples_via_MVN(self, approx_type=ApproxType.LS, n_samples=10000):
        '''
        Retrieves likelihood samples in parameter space, weighted by their likelihood (raw, not log) 
        :param n_samples: how many samples to re-sample from the list of likelihood samples
        :return: tuple of weight_sampled_params, params, weights, log_probs
        '''

        if approx_type not in self.map_approx_type_to_MVN:
            raise ValueError('Need to fit MVN to likelihood samples')

        weight_sampled_params = self.map_approx_type_to_MVN[approx_type]['model'].rvs(n_samples)
        log_probs = [self.get_log_likelihood(x) for x in weight_sampled_params]

        return weight_sampled_params, weight_sampled_params, [1] * len(weight_sampled_params), log_probs

    def get_weighted_samples_via_direct_sampling(self, n_samples=10000):
        '''
        I DON'T USE THIS NOW
        Retrieves likelihood samples in parameter space, weighted by their likelihood (raw, not log) 
        :param n_samples: how many samples to re-sample from the list of likelihood samples
        :return: tuple of weight_sampled_params, params, weights, log_probs
        '''

        valid_ind = [i for i, x in enumerate(self.all_log_probs_as_list) if np.isfinite(np.exp(x)) and \
                     np.isfinite(self.all_propensities_as_list[i]) and \
                     self.all_propensities_as_list[i] > 0]

        if len(valid_ind) == 0:
            return [], [], [], []

        propensities = np.array(self.all_propensities_as_list)[valid_ind]
        log_probs = np.array(self.all_log_probs_as_list)[valid_ind]
        params = np.array(self.all_samples_as_list)[valid_ind]

        weights = np.array([np.exp(x) / propensities[i] for i, x in enumerate(log_probs)])
        weights /= sum(weights)

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

        if len(valid_ind) == 0:
            return [], [], [], []

        sampled_valid_inds = np.random.choice(len(valid_ind), n_samples, p=np.array(weights)[valid_ind], replace=True)
        param_inds = [valid_ind[i] for i in sampled_valid_inds]
        weight_sampled_params = [params[i] for i in param_inds]

        # TODO: add a plot to show distribution of weights and log-probs for likelihood samples
        # max_weight_ind = np.argmax(weights)
        # max_log_prob_ind = np.argmax(log_probs)

        # print(f'took {timer.elapsed_time():.2g} seconds to process resampling')
        # 
        # print(f'max weight: {max(weights)}')
        # print(f'max log-prob: {max(log_probs)}')
        # print(f'propensity at max weight: {propensities[max_weight_ind]:.4g}')
        # print(f'log_prob at max weight: {log_probs[max_weight_ind]:.4g}')
        # 
        # print(f'propensity at max log-prob: {propensities[max_log_prob_ind]:.4g}')
        # print(f'params at max log-prob: {params[max_log_prob_ind]}')
        # print(f'log_prob at all-data fit: {self.get_log_likelihood(self.all_data_params):.4g}')
        # print(f'params at max weight: {params[max_weight_ind]}')
        # print(f'params at max weight: {self.get_log_likelihood(params[max_weight_ind])}')
        # desc_weights_inds = np.argsort(-weights)[:10]
        # print(f'descending list of weights: {[weights[i] for i in desc_weights_inds]}')
        # print(f'log-probs at descending list of weights {[log_probs[i] for i in desc_weights_inds]}')
        # print(f'propensities at descending list of weights {[propensities[i] for i in desc_weights_inds]}')

        return weight_sampled_params, params, weights, log_probs

    def render_and_plot_cred_int(self,
                                 approx_type=None,
                                 mvn_fit=False
                                 ):
        '''
        Use arviz to plot the credible intervals
        :return: None, just adds attributes to the object
        '''

        param_type = approx_type.value[1]
        calc_param_names = self.sorted_names + list(self.extra_params.keys())

        params, _, _, _ = self.get_weighted_samples(approx_type=approx_type, mvn_fit=mvn_fit)

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

    def run_fits(self):
        '''
        Builder that goes through each method in its proper sequence
        :return: None
        '''

        # # Sample plot just to check
        # self.solve_and_plot_solution(title='Test Plot with Default Parameters',
        #                              plot_filename_filename='test_plot.png')

        self.render_all_data_fit()

        # Plot all-data solution 
        self.solve_and_plot_solution(in_params=self.all_data_params,
                                     title='All-Data Solution',
                                     plot_filename_filename='all_data_solution.png')

        if ApproxType.Hess in self.model_approx_types:
            # Plot all hessian solutions
            self.plot_all_solutions(approx_type=ApproxType.Hess)

            # Get and plot parameter distributions from hessian
            self.render_and_plot_cred_int(approx_type=ApproxType.Hess)

        if ApproxType.SM in self.model_approx_types:
            try:
                # Do statsmodels
                self.render_statsmodels_fit_timeseries()
                self.plot_growth_rate_timeseries(plot_filename_filename='statsmodels_growth_rate_time_series.png')
                self.render_statsmodels_fit()
                self.fit_MVN_to_likelihood(cov_type='full', approx_type=ApproxType.SM)
            except:
                print('Error calculating and rendering statsmodels fit')

        # self.all_data_params = self.statsmodels_params
        # for param_name in self.sorted_names:
        #     if 'sigma' in param_name:
        #         self.all_data_params[param_name] = 0.1
        # self.render_all_data_fit(passed_params=self.all_data_params)

        if ApproxType.BS in self.model_approx_types:
            try:
                # Training Data Bootstraps
                self.render_bootstraps()

                # Plot example solutions from bootstrap
                bootstrap_selection = np.random.choice(self.bootstrap_params)
                self.solve_and_plot_solution(in_params=bootstrap_selection,
                                             title='Random Bootstrap Selection',
                                             plot_filename_filename='random_bootstrap_selection.png')

                # Plot all bootstraps
                self.plot_all_solutions(approx_type=ApproxType.BS)

                # Get and plot parameter distributions from bootstraps
                self.render_and_plot_cred_int(approx_type=ApproxType.BS)
            except:
                print('Error calculating and rendering bootstraps')

            try:
                # Next define MVN model on likelihood and fit   
                self.fit_MVN_to_likelihood(approx_type=ApproxType.BS)
                # Plot all solutions...
                self.plot_all_solutions(approx_type=ApproxType.BS, mvn_fit=True)
                # Get and plot parameter distributions from bootstraps
                self.render_and_plot_cred_int(approx_type=ApproxType.BS, mvn_fit=True)
            except:
                print('Error calculating and rendering MVN fit to bootstraps')

        if ApproxType.PyMC3 in self.model_approx_types:
            try:
                # Do statsmodels
                self.render_PyMC3_fit()
            except:
                print('Error calculating and rendering PyMC3 fit')

        if ApproxType.LS in self.model_approx_types:
            try:
                print('Sampling around MLE with empirical sigma')
                self.MCMC(self.all_data_params, opt_walk=False,
                          sample_shape_param='empirical', which_distro=WhichDistro.norm)
                # Do random walks around the overall fit
                # print('\nSampling around MLE with wide sigma')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=1, which_distro='norm')
                # print('Sampling around MLE with medium sigma')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=10, which_distro='norm')
                # print('Sampling around MLE with narrow sigma')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=10, which_distro=WhichDistro.norm)
                # print('Sampling around MLE with narrow sigma')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=100, which_distro=WhichDistro.norm_trunc)
                # print('Sampling around MLE with ultra-narrow sigma')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=1000, which_distro=WhichDistro.norm_trunc)

                # print('Sampling around MLE with medium exponential parameter')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=10, which_distro=WhichDistro.laplace)
                # print('Sampling around MLE with narrow exponential parameter')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=100, which_distro=WhichDistro.laplace)
                # print('Sampling around MLE with ultra-narrow exponential parameter')
                # self.MCMC(self.all_data_params, opt_walk=False,
                #           sample_shape_param=1000, which_distro=WhichDistro.laplace)

                # Get and plot parameter distributions from bootstraps
                self.render_and_plot_cred_int(approx_type=ApproxType.LS)

                # Plot all solutions...
                self.plot_all_solutions(approx_type=ApproxType.LS)
            except:
                print('Error calculating and rendering direct likelihood samples')

            try:
                # Next define MVN model on likelihood and fit   
                self.fit_MVN_to_likelihood(approx_type=ApproxType.LS)
                # Plot all solutions...
                self.plot_all_solutions(approx_type=ApproxType.LS, mvn_fit=True)
                # Get and plot parameter distributions from bootstraps
                self.render_and_plot_cred_int(approx_type=ApproxType.LS, mvn_fit=True)
            except:
                print('Error calculating and rendering MVN fit to likelihood')

        if ApproxType.MCMC in self.model_approx_types:
            try:
                print('Sampling via random walk MCMC, starting with MLE')  # a random bootstrap selection')
                # bootstrap_selection = np.random.choice(self.n_bootstraps)
                # starting_point = self.bootstrap_params[bootstrap_selection]
                self.MCMC(self.all_data_params, opt_walk=True)

                # print('Sampling via random walk NUTS, starting with MLE')
                # self.NUTS(self.all_data_params)

                # Plot all solutions...
                self.plot_all_solutions(approx_type=ApproxType.MCMC)

                # Get and plot parameter distributions from bootstraps
                self.render_and_plot_cred_int(approx_type=ApproxType.MCMC)
            except:
                print('Error calculating and rendering random walk')

            try:
                # Next define MVN model on likelihood and fit   
                self.fit_MVN_to_likelihood(approx_type=ApproxType.MCMC)
                # Plot all solutions...
                self.plot_all_solutions(approx_type=ApproxType.MCMC, mvn_fit=True)
                # Get and plot parameter distributions from bootstraps
                self.render_and_plot_cred_int(approx_type=ApproxType.MCMC, mvn_fit=True)
            except:
                print('Error calculating and rendering MVN fit to random walk')

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

    def pretty_print_params(self, in_obj):
        '''
        Helper function for printing our parameter values and bounds consistently
        :param in_dict: dictionary of parameters to pretty print
        :return: None, just prints
        '''
        if in_obj is None:
            print('None')
        else:
            if type(in_obj) != dict:
                in_dict = self.convert_params_as_list_to_dict(in_obj)
            else:
                in_dict = in_obj.copy()
            for name in self.sorted_names:
                val = in_dict[name]
                if type(val) == tuple and len(val) == 2:
                    val_str = f'({val[0]:.4g}, {val[1]:.4g})'
                else:
                    val_str = f'{val:.4g}'
                print(f'{name}: {val_str}')
