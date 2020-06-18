from sub_units.bayes_model import BayesModel
from scipy.integrate import odeint
import numpy as np
import datetime
from sub_units.utils import ApproxType
import pandas as pd
import joblib


class FrechetModel(BayesModel):

    # add model_type_str to kwargs when instantiating super
    def __init__(self,
                 *args,
                 optimizer_method='BFGS',  # 'Nelder-Mead', #'SLSQP',
                 burn_in=0,
                 **kwargs):
        kwargs.update({'model_type_name': 'frechet',
                       'min_sol_date': None,  # TODO: find a better way to set this attribute
                       'optimizer_method': optimizer_method,
                       'burn_in': burn_in,
                       'cases_cnt_threshold': 100,
                       'deaths_cnt_threshold': 100,
                       'n_params_for_emcee': 8
                       })

        model_approx_types = [ApproxType.PyMC3]
        # model_approx_types = [ApproxType.BS, ApproxType.SP_CF, ApproxType.NDT_Hess, ApproxType.NDT_Jac, ApproxType.LS,
        #                       ApproxType.MCMC, ApproxType.PyMC3]
        kwargs.update({'model_approx_types': model_approx_types})
        super(FrechetModel, self).__init__(*args, **kwargs)
        self.cases_indices = list(range(self.day_of_threshold_met_case, len(self.series_data)))
        self.deaths_indices = list(range(self.day_of_threshold_met_death, len(self.series_data)))

        # dont need to filter out zero-values since we now add the logarithm offset
        # self.cases_indices = [i for i in cases_indices if self.data_new_tested[i] > 0]
        # self.deaths_indices = [i for i in deaths_indices if self.data_new_dead[i] > 0]

    # from https://en.wikipedia.org/wiki/Fréchet_distribution
    @staticmethod
    def frechet_pdf(x, a, s, m, mult):
        if s == 0:
            raise ValueError
        if x == m and a > 1 or x == m and a > 0:
            return 1
        return (a / s) * ((x - m) / s) ** (-1 - a) * np.exp(-((x - m) / s) ** -a) * mult

    def run_simulation(self, in_params, **kwargs):
        '''
        run combined ODE and convolution simulation
        :param params: dictionary of relevant parameters
        :return: N
        '''

        params = self.convert_params_as_list_to_dict(in_params)

        params.update(self.static_params)

        # print('t_vals:', self.t_vals)

        positive = np.array([self.frechet_pdf(x, params['alpha_positive'], params['s_positive'], params['m_positive'],
                                              params['mult_positive']) for x in self.t_vals])
        deceased = np.array([self.frechet_pdf(x, params['alpha_deceased'], params['s_deceased'], params['m_deceased'],
                                              params['mult_deceased']) for x in self.t_vals])

        return np.vstack([np.zeros_like(positive),
                          np.maximum(positive, 0),
                          np.maximum(deceased, 0)])

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
            cases_bootstrap_indices = self.cases_indices
        if deaths_bootstrap_indices is None:
            deaths_bootstrap_indices = self.deaths_indices

        # timer = Stopwatch()
        sol = self.run_simulation(params)
        new_tested_from_sol = sol[1]
        new_deceased_from_sol = sol[2]

        actual_tested = [np.log(data_new_tested[i] + self.log_offset) for i in cases_bootstrap_indices]
        predicted_tested = [np.log(new_tested_from_sol[i + self.burn_in] + self.log_offset) for i in
                            cases_bootstrap_indices]
        predicted_tested = [0 if not np.isfinite(x) else x for x in predicted_tested]

        actual_dead = [np.log(data_new_dead[i] + self.log_offset) for i in deaths_bootstrap_indices]
        predicted_dead = [np.log(new_deceased_from_sol[i + self.burn_in] + self.log_offset) for i in
                          deaths_bootstrap_indices]
        predicted_dead = [0 if not np.isfinite(x) else x for x in predicted_dead]

        new_tested_dists = [predicted_tested[i] - actual_tested[i] for i in range(len(predicted_tested))]
        new_dead_dists = [predicted_dead[i] - actual_dead[i] for i in range(len(predicted_dead))]

        tested_vals = [data_new_tested[i] for i in cases_bootstrap_indices]
        deceased_vals = [data_new_dead[i] for i in deaths_bootstrap_indices]

        other_errs = list()
        # for param_name in self.logarithmic_params:
        #     pos_only_val = params[param_name]
        #     other_errs += [abs(pos_only_val) if pos_only_val < 0 else 0] # NB: get_log_likelihood applies square operation

        return new_tested_dists, new_dead_dists, other_errs, sol, tested_vals, deceased_vals, \
               predicted_tested, actual_tested, predicted_dead, actual_dead

    def render_PyMC3_fit(self, opt_simplified=False):

        import theano.tensor as tt
        import pymc3 as pm
        from theano.compile.ops import as_op

        success = False
        print(f'Loading from {self.PyMC3_filename}...')
        try:
            PyMC3_dict = joblib.load(self.PyMC3_filename)
            all_PyMC3_samples_as_list = PyMC3_dict['all_PyMC3_samples_as_list']
            all_PyMC3_log_probs_as_list = PyMC3_dict['all_PyMC3_log_probs_as_list']
            success = True
            self.loaded_PyMC3 = True
        except:
            print('...Loading failed!')
            self.loaded_PyMC3 = False

        # TODO: Break out all-data fit to its own method, not embedded in render_bootstraps
        if (not success and self.opt_calc) or self.opt_force_calc:

            print('Massaging data into dataframe...')

            # PyMC3 requires a Pandas dataframe, so let's get cooking!
            data_as_list_of_dicts = [{'new_tested': max(self.data_new_tested[i], 0) + self.log_offset,
                                      'new_dead': max(self.data_new_dead[i], 0) + self.log_offset,
                                      'log_new_tested': np.log(max(self.data_new_tested[i], 0) + self.log_offset),
                                      'log_new_dead': np.log(max(self.data_new_dead[i], 0) + self.log_offset),
                                      'orig_ind': i + self.burn_in,
                                      'x': i,
                                      } for ind, i in enumerate(
                range(len(self.data_new_tested)))]

            data = pd.DataFrame(data_as_list_of_dicts)

            ind_offset = 0  # had to hand-tune this to zero
            data['DOW'] = [str((x + ind_offset) % 7) for x in data['orig_ind']]
            data['day1'] = [1 if (x + ind_offset) % 7 == 1 else 0 for x in data['orig_ind']]
            data['day2'] = [1 if (x + ind_offset) % 7 == 2 else 0 for x in data['orig_ind']]
            data['day3'] = [1 if (x + ind_offset) % 7 == 3 else 0 for x in data['orig_ind']]
            data['day4'] = [1 if (x + ind_offset) % 7 == 4 else 0 for x in data['orig_ind']]
            data['day5'] = [1 if (x + ind_offset) % 7 == 5 else 0 for x in data['orig_ind']]
            data['day6'] = [1 if (x + ind_offset) % 7 == 6 else 0 for x in data['orig_ind']]

            print('day_of_threshold_met_case:', self.day_of_threshold_met_case)
            start_ind = self.day_of_threshold_met_case
            x = data['x'].values[start_ind:]

            print('Running PyMC3 fit...')
            with pm.Model() as model_positive:  # model specifications in PyMC3 are wrapped in a with-statement
                # Define priors
                a = pm.Uniform('alpha_positive', 1e-8, 1000)
                s = pm.Uniform('s_positive', 1e-3, 1000)
                m = pm.Uniform('m_positive', -100, 100)
                mult = pm.Uniform('mult_positive', 1e-8, 1e12)
                sigma = pm.HalfNormal('sigma_positive', sigma=1)

                curve = pm.Deterministic('curve_positive',
                                         tt.log((a / s) * tt.power(((x - m) / s), (-1 - a)) * tt.exp(
                                             -tt.power(((x - m) / s), -a)) * mult))

                # Define likelihood
                Y_obs = pm.Normal('new_tested',
                                  mu=curve,
                                  sigma=sigma,
                                  observed=data['log_new_tested'].values[start_ind:]
                                  )

                # Inference!
                # print("Searching for MAP...")
                MAP = {key: val for key, val in self.all_data_params.items() if 'positive' in key}
                # MAP = pm.find_MAP(model=model_positive, start=start_params)
                # print('...done! MAP:')
                # print(MAP)
                positive_trace = pm.sample(2000, start=MAP)

            positive_trace_as_dict = {name: positive_trace.get_values(name) for name in self.sorted_names if
                                      'positive' in name}
            positive_trace_as_list_of_dicts = list()
            for i in range(len(positive_trace.get_values('alpha_positive'))):
                dict_to_add = dict()
                for name in positive_trace_as_dict:
                    dict_to_add.update({name: positive_trace_as_dict[name][i]})
                positive_trace_as_list_of_dicts.append(dict_to_add)

            print('day_of_threshold_met_death:', self.day_of_threshold_met_death)
            start_ind = self.day_of_threshold_met_death
            x = data['x'].values[start_ind:]

            with pm.Model() as model_deceased:  # model specifications in PyMC3 are wrapped in a with-statement

                # Define priors
                a = pm.Uniform('alpha_deceased', 1e-8, 1000)
                s = pm.Uniform('s_deceased', 1e-3, 1000)
                m = pm.Uniform('m_deceased', -100, 100)
                mult = pm.Uniform('mult_deceased', 1e-8, 1e12)
                sigma = pm.HalfNormal('sigma_deceased', sigma=1)

                curve = pm.Deterministic('curve_deceased',
                                         tt.log(tt.maximum((a / s) * tt.power(((x - m) / s), (-1 - a)) * tt.exp(
                                             -tt.power(((x - m) / s), -a)) * mult, 0) + self.log_offset))

                # Define likelihood
                Y_obs = pm.Normal('new_dead',
                                  mu=curve,
                                  sigma=sigma,
                                  observed=data['log_new_dead'].values[start_ind:]
                                  )

                # Inference!
                # print("Searching for MAP...")
                MAP = {key: val for key, val in self.all_data_params.items() if 'deceased' in key}
                # MAP = pm.find_MAP(model=model_positive, start=start_params)
                # print('...done! MAP:')
                # print(MAP)
                deceased_trace = pm.sample(2000, start=MAP)

            deceased_trace_as_dict = {name: deceased_trace.get_values(name) for name in self.sorted_names if
                                      'deceased' in name}
            deceased_trace_as_list_of_dicts = list()
            for i in range(len(deceased_trace.get_values('alpha_deceased'))):
                dict_to_add = dict()
                for name in deceased_trace_as_dict:
                    dict_to_add.update({name: deceased_trace_as_dict[name][i]})
                deceased_trace_as_list_of_dicts.append(dict_to_add)

            trace_as_list_of_dicts = list()
            for positive_params, deceased_params in zip(positive_trace_as_list_of_dicts,
                                                        deceased_trace_as_list_of_dicts):
                dict_to_add = positive_params.copy()
                dict_to_add.update(deceased_params)
                trace_as_list_of_dicts.append(dict_to_add)

            for tmp_dict in trace_as_list_of_dicts:
                for key in tmp_dict:
                    if key in self.logarithmic_params:
                        tmp_dict[key] = np.exp(tmp_dict[key])

            all_PyMC3_samples_as_list = [self.convert_params_as_dict_to_list(tmp_dict) for tmp_dict in
                                         trace_as_list_of_dicts]
            all_PyMC3_log_probs_as_list = [self.get_log_likelihood(params) for params in trace_as_list_of_dicts]

            tmp_dict = {'all_PyMC3_samples_as_list': all_PyMC3_samples_as_list,
                        'all_PyMC3_log_probs_as_list': all_PyMC3_log_probs_as_list}

            print(f'saving PyMC3 trace to {self.PyMC3_filename}...')
            joblib.dump(tmp_dict, self.PyMC3_filename)
            print('...done!')

        self.all_PyMC3_samples_as_list = all_PyMC3_samples_as_list
        self.all_PyMC3_log_probs_as_list = all_PyMC3_log_probs_as_list

        # Plot all solutions...
        self.plot_all_solutions(approx_type=ApproxType.PyMC3)

        # Get and plot parameter distributions from bootstraps
        self.render_and_plot_cred_int(approx_type=ApproxType.PyMC3)
