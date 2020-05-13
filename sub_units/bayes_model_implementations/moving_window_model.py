from sub_units.bayes_model import BayesModel
import numpy as np
import pandas as pd
import datetime
import scipy as sp


class MovingWindowModel(BayesModel):

    # add model_type_str to kwargs when instantiating super
    def __init__(self, state, max_date_str, moving_window_size=14, **kwargs):
        min_sol_date = datetime.datetime.strptime(max_date_str, '%Y-%m-%d') - datetime.timedelta(
            days=moving_window_size)
        model_type_name = f'moving_window_{moving_window_size}_days'

        # these kwargs will be added as object attributes
        kwargs.update({'model_type_name': model_type_name,
                       'moving_window_size': moving_window_size,
                       'min_sol_date': min_sol_date})
        super(MovingWindowModel, self).__init__(state, max_date_str, **kwargs)

        ind1 = max(self.day_of_threshold_met_case, len(self.series_data) - moving_window_size)
        cases_indices = list(range(ind1, len(self.series_data)))
        ind1 = max(self.day_of_threshold_met_death, len(self.series_data) - moving_window_size)
        deaths_indices = list(range(ind1, len(self.series_data)))
        self.cases_indices = [i for i in cases_indices if self.data_new_tested[i] > 0]
        self.deaths_indices = [i for i in deaths_indices if self.data_new_dead[i] > 0]

    def run_simulation(self, in_params):
        '''
        run combined ODE and convolution simulation
        :param params: dictionary of relevant parameters
        :return: N
        '''

        params = self.convert_params_as_list_to_dict(in_params)

        params.update(self.static_params)
        contagious = np.array([None] * len(self.t_vals))  # this guy doesn't matter for MovingWindowModel

        # do intercept at the beginning of moving window
        intercept_t_val = self.max_date_in_days - self.moving_window_size
        end_positive_count = np.exp(intercept_t_val * params['positive_slope'])
        end_deceased_count = np.exp(intercept_t_val * params['deceased_slope'])
        positive = np.array(np.exp(self.t_vals * params['positive_slope'])) * np.array(
            params['positive_intercept'] / end_positive_count)
        deceased = np.array(np.exp(self.t_vals * params['deceased_slope'])) * np.array(
            params['deceased_intercept'] / end_deceased_count)

        # positive = np.array(np.exp(self.t_vals * params['positive_slope'])) * params['positive_intercept']
        # deceased = np.array(np.exp(self.t_vals * params['deceased_slope'])) * params['deceased_intercept']

        # for i in range(len(positive)):
        #     if i % 7 == 0:
        #         positive[i] *= params['day0_multiplier']
        #         deceased[i] *= params['day0_multiplier']
        #     if i % 7 == 1:
        #         positive[i] *= params['day1_multiplier']
        #         deceased[i] *= params['day1_multiplier']
        #     if i % 7 == 2:
        #         positive[i] *= params['day2_multiplier']
        #         deceased[i] *= params['day2_multiplier']
        #     if i % 7 == 3:
        #         positive[i] *= params['day3_multiplier']
        #         deceased[i] *= params['day3_multiplier']
        #     if i % 7 == 4:
        #         positive[i] *= params['day4_multiplier']
        #         deceased[i] *= params['day4_multiplier']
        #     if i % 7 == 5:
        #         positive[i] *= params['day5_multiplier']
        #         deceased[i] *= params['day5_multiplier']
        #     if i % 7 == 6:
        #         positive[i] *= params['day6_multiplier']
        #         deceased[i] *= params['day6_multiplier']

        for i in range(len(positive)):
            if i % 7 == 0:
                positive[i] *= params['day0_positive_multiplier']
                deceased[i] *= params['day0_deceased_multiplier']
            if i % 7 == 1:
                positive[i] *= params['day1_positive_multiplier']
                deceased[i] *= params['day1_deceased_multiplier']
            if i % 7 == 2:
                positive[i] *= params['day2_positive_multiplier']
                deceased[i] *= params['day2_deceased_multiplier']
            if i % 7 == 3:
                positive[i] *= params['day3_positive_multiplier']
                deceased[i] *= params['day3_deceased_multiplier']
            if i % 7 == 4:
                positive[i] *= params['day4_positive_multiplier']
                deceased[i] *= params['day4_deceased_multiplier']
            if i % 7 == 5:
                positive[i] *= params['day5_positive_multiplier']
                deceased[i] *= params['day5_deceased_multiplier']
            if i % 7 == 6:
                positive[i] *= params['day6_positive_multiplier']
                deceased[i] *= params['day6_deceased_multiplier']

        return np.vstack([np.squeeze(contagious), positive[:contagious.size], deceased[:contagious.size]])

    def _get_log_likelihood_precursor(self,
                                      in_params,
                                      data_new_tested=None,
                                      data_new_dead=None,
                                      cases_bootstrap_indices=None,
                                      deaths_bootstrap_indices=None):
        '''
        Obtains the precursors for obtaining the log likelihood for a given parameter dictionary
          Used for self._errfunc_for_least_squares (error function for scipy.optimize.least_squares)
          and self.get_log_likelihood (neg. loss function for scipy.optimize.minimize(method='SLSQP'))
        :param in_params: dictionary or list of parameters
        :param cases_bootstrap_indices: bootstrap indices when applicable
        :param deaths_bootstrap_indices:  bootstrap indices when applicable
        :param cases_bootstrap_indices: which indices to include in the likelihood?
        :param deaths_bootstrap_indices: which indices to include in the likelihood?
        :return: tuple of lists: distances, other errors, and simulated solution
        '''

        # convert from list to dictionary (for compatibility with the least-sq solver
        params = self.convert_params_as_list_to_dict(in_params)

        if data_new_tested is None:
            data_new_tested = self.data_new_tested
        if data_new_dead is None:
            data_new_dead = self.data_new_dead

        if cases_bootstrap_indices is None:
            cases_bootstrap_indices = self.cases_indices[-self.moving_window_size:]
        if deaths_bootstrap_indices is None:
            deaths_bootstrap_indices = self.deaths_indices[-self.moving_window_size:]

        # timer = Stopwatch()
        sol = self.run_simulation(params)
        new_tested_from_sol = sol[1]
        new_deceased_from_sol = sol[2]
        # print(f'Simulation took {timer.elapsed_time() * 100} ms')
        # timer = Stopwatch()

        new_tested_dists = [ \
            (np.log(data_new_tested[i]) - np.log(new_tested_from_sol[i + self.burn_in]))
            for i in cases_bootstrap_indices]
        new_dead_dists = [ \
            (np.log(data_new_dead[i]) - np.log(new_deceased_from_sol[i + self.burn_in]))
            for i in deaths_bootstrap_indices]

        tested_vals = [data_new_tested[i] for i in cases_bootstrap_indices]
        deceased_vals = [data_new_dead[i] for i in deaths_bootstrap_indices]
        other_errs = list()

        return new_tested_dists, new_dead_dists, other_errs, sol, tested_vals, deceased_vals

    def render_statsmodels_fit(self, opt_simplified=False):
        '''
        Performs fit using statsmodels, since this is a standard linear regression. This model gives us standard errors.
        :return: 
        '''

        # this only needs to be imported if it's being used...
        import statsmodels.formula.api as smf

        name_mapping_positive = {'DOW[T.1]': 'day1_positive_multiplier',
                                 'DOW[T.2]': 'day2_positive_multiplier',
                                 'DOW[T.3]': 'day3_positive_multiplier',
                                 'DOW[T.4]': 'day4_positive_multiplier',
                                 'DOW[T.5]': 'day5_positive_multiplier',
                                 'DOW[T.6]': 'day6_positive_multiplier',
                                 'x': 'positive_slope',
                                 'Intercept': 'positive_intercept'
                                 }

        name_mapping_deceased = {'DOW[T.1]': 'day1_deceased_multiplier',
                                 'DOW[T.2]': 'day2_deceased_multiplier',
                                 'DOW[T.3]': 'day3_deceased_multiplier',
                                 'DOW[T.4]': 'day4_deceased_multiplier',
                                 'DOW[T.5]': 'day5_deceased_multiplier',
                                 'DOW[T.6]': 'day6_deceased_multiplier',
                                 'x': 'deceased_slope',
                                 'Intercept': 'deceased_intercept'
                                 }

        positive_names = [name for name in self.sorted_names if 'positive' in name and 'sigma' not in name]
        deceased_names = [name for name in self.sorted_names if 'deceased' in name and 'sigma' not in name]

        data_new_tested = self.data_new_tested
        data_new_deceased = self.data_new_dead
        moving_window_size = self.moving_window_size
        cases_bootstrap_indices = self.cases_indices[-moving_window_size:]
        deaths_bootstrap_indices = self.deaths_indices[-moving_window_size:]

        data = pd.DataFrame([{'x': ind,
                              # add burn_in to orig_ind since simulations start earlier than the data
                              'orig_ind': i + self.burn_in,
                              'new_positive': data_new_tested[i],
                              'new_deceased': data_new_deceased[i],
                              } for ind, i in
                             enumerate(range(len(data_new_tested) - moving_window_size, len(data_new_tested)))])

        # need to use orig_ind to align intercept with run_simulation
        ind_offset = 0  # had to hand-tune this to zero
        data['DOW'] = [str((x + ind_offset) % 7) for x in data['orig_ind']]

        #####
        # Do fit on positive curve
        #####

        model_positive = smf.ols(formula='np.log(new_positive + 0.1) ~ x + DOW', data=data)
        results_positive = model_positive.fit()
        print(results_positive.summary())
        params_positive = results_positive.params
        for name1, name2 in name_mapping_positive.items():
            params_positive[name2] = params_positive.pop(name1)
        # params_positive['positive_intercept'] = np.exp(params_positive['positive_intercept'])
        bse_positive = dict(results_positive.bse)
        for name1, name2 in name_mapping_positive.items():
            bse_positive[name2] = bse_positive.pop(name1)
        # bse_positive['positive_intercept'] = bse_positive['positive_intercept'] * \
        #    params_positive['positive_intercept']  # due to A = 1000; B = 0.01; B * np.exp(A) = np.exp(A + B) - np.exp(A)
        # also, note that we have already applied hte exponential transofrm on A
        means_as_list = [params_positive[name] for name in positive_names]
        sigma_as_list = [bse_positive[name] for name in positive_names]
        print('sigma_as_list:', sigma_as_list)
        print('means_as_list:', means_as_list)
        cov = np.diag([max(1e-8, x ** 2) for x in sigma_as_list])
        self.statsmodels_model_positive = sp.stats.multivariate_normal(mean=means_as_list, cov=cov)

        #####
        # Do fit on deceased curve
        #####

        # add 0.1 so you don't bonk on log(0)
        model_deceased = smf.ols(formula='np.log(new_deceased + 0.1) ~ x + DOW', data=data)
        results_deceased = model_deceased.fit()
        print(results_deceased.summary())
        params_deceased = dict(results_deceased.params)
        for name1, name2 in name_mapping_deceased.items():
            params_deceased[name2] = params_deceased.pop(name1)
        # params_deceased['deceased_intercept'] = np.exp(params_deceased['deceased_intercept'])
        means_as_list = [params_deceased[name] for name in deceased_names]
        bse_deceased = dict(results_deceased.bse)
        for name1, name2 in name_mapping_deceased.items():
            bse_deceased[name2] = bse_deceased.pop(name1)
        # bse_deceased['deceased_intercept'] = bse_deceased['deceased_intercept'] * \
        #     params_deceased['deceased_intercept']  # due to A = 1000; B = 0.01; B * np.exp(A) = np.exp(A + B) - np.exp(A)
        # also, note that we have already applied hte exponential transofrm on A
        print(bse_deceased)
        sigma_as_list = [bse_deceased[name] for name in deceased_names]

        cov = np.diag([max(1e-8, x ** 2) for x in sigma_as_list])

        print('sigma_as_list:', sigma_as_list)
        print('means_as_list:', means_as_list)
        self.statsmodels_model_deceased = sp.stats.multivariate_normal(mean=means_as_list, cov=cov)

        if not opt_simplified:
            self.render_and_plot_cred_int(param_type='statsmodels')
        
        self.plot_all_solutions(key='statsmodels')

    def get_weighted_samples_via_statsmodels(self, n_samples=1000, ):
        '''
        Retrieves likelihood samples in parameter space, weighted by their standard errors from statsmodels
        :param n_samples: how many samples to re-sample from the list of likelihood samples
        :return: tuple of weight_sampled_params, params, weights, log_probs
        '''

        weight_sampled_params_positive = self.statsmodels_model_positive.rvs(n_samples)
        weight_sampled_params_deceased = self.statsmodels_model_deceased.rvs(n_samples)

        positive_names = [name for name in self.sorted_names if 'positive' in name and 'sigma' not in name]
        map_name_to_sorted_ind_positive = {val: i for i, val in enumerate(positive_names)}
        deceased_names = [name for name in self.sorted_names if 'deceased' in name and 'sigma' not in name]
        map_name_to_sorted_ind_deceased = {val: i for i, val in enumerate(deceased_names)}

        weight_sampled_params = list()
        for i in range(n_samples):
            positive_dict = self.convert_params_as_list_to_dict(weight_sampled_params_positive[i],
                                                                map_name_to_sorted_ind=map_name_to_sorted_ind_positive)
            deceased_dict = self.convert_params_as_list_to_dict(weight_sampled_params_deceased[i],
                                                                map_name_to_sorted_ind=map_name_to_sorted_ind_deceased)
            all_dict = positive_dict.copy()
            all_dict.update(deceased_dict)
            all_dict['sigma_positive'] = 1
            all_dict['sigma_deceased'] = 1
            for name in self.logarithmic_params:
                all_dict[name] = np.exp(all_dict[name])
            weight_sampled_params.append(self.convert_params_as_dict_to_list(all_dict.copy()))

        log_probs = [self.get_log_likelihood(x) for x in weight_sampled_params]

        return weight_sampled_params, weight_sampled_params, [1] * len(weight_sampled_params), log_probs

    def run_fits_simplified(self):
        '''
        Builder that goes through each method in its proper sequence
        :return: None
        '''

        # Do statsmodels. Yes. It's THAT simplified.
        self.render_statsmodels_fit(opt_simplified=True)
