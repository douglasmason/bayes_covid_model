from sub_units.bayes_model import BayesModel
from scipy.integrate import odeint
import numpy as np
import datetime
from sub_units.utils import ApproxType


class ConvolutionModel(BayesModel):

    # add model_type_str to kwargs when instantiating super
    def __init__(self,
                 *args,
                 optimizer_method='Nelder-Mead',  # 'Nelder-Mead', #'SLSQP',
                 **kwargs):
        kwargs.update({'model_type_name': 'convolution',
                       'min_sol_date': None,  # TODO: find a better way to set this attribute
                       'optimizer_method': optimizer_method,
                       })
        super(ConvolutionModel, self).__init__(*args, **kwargs)
        self.cases_indices = list(range(self.day_of_threshold_met_case, len(self.series_data)))
        self.deaths_indices = list(range(self.day_of_threshold_met_death, len(self.series_data)))

        # dont need to filter out zero-values since we now add the logarithm offset
        # self.cases_indices = [i for i in cases_indices if self.data_new_tested[i] > 0]
        # self.deaths_indices = [i for i in deaths_indices if self.data_new_dead[i] > 0]

    @staticmethod
    def _ODE_system(y, t, *p):
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

        params = self.convert_params_as_list_to_dict(in_params)

        params.update(self.static_params)

        # First we simulate how the growth rate results into total # of contagious
        param_tuple = tuple(params[x] for x in ['alpha_1', 'alpha_2']) + (self.SIP_date_in_days,)
        contagious = odeint(self._ODE_system,
                            [params['I_0']],
                            self.t_vals,
                            args=param_tuple)
        contagious = np.array(contagious)

        # then use convolution to simulate transition to positive
        convolution_kernel = self.norm(np.linspace(0, len(self.t_vals), len(self.t_vals) + 1),
                                       mu=params['contagious_to_positive_delay'],
                                       std=params['contagious_to_positive_width'])
        if sum(convolution_kernel) == 0:
            convolution_kernel = np.zeros_like(convolution_kernel)
        else:
            convolution_kernel /= sum(convolution_kernel)
        convolution_kernel = np.array(convolution_kernel)
        positive = np.convolve(np.squeeze(contagious),
                               convolution_kernel) * 0.1  # params['contagious_to_positive_mult']

        # then use convolution to simulate transition to deceased
        convolution_kernel = self.norm(np.linspace(0, len(self.t_vals), len(self.t_vals) + 1),
                                       mu=params['contagious_to_deceased_delay'],
                                       std=params['contagious_to_deceased_width'])
        if sum(convolution_kernel) == 0:
            convolution_kernel = np.zeros_like(convolution_kernel)
        else:
            convolution_kernel /= sum(convolution_kernel)
        convolution_kernel = np.array(convolution_kernel)
        deceased = np.convolve(np.squeeze(contagious), convolution_kernel) * params['contagious_to_deceased_mult']

        return np.vstack([np.maximum(np.squeeze(contagious), 0),
                          np.maximum(np.array(positive[:contagious.size]), 0),
                          np.maximum(np.array(deceased[:contagious.size]), 0)])

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
        # print(f'Simulation took {timer.elapsed_time() * 100} ms')
        # timer = Stopwatch()

        actual_tested = [np.log(data_new_tested[i] + self.log_offset) for i in cases_bootstrap_indices]
        predicted_tested = [np.log(new_tested_from_sol[i + self.burn_in] + self.log_offset) for i in cases_bootstrap_indices]
        actual_dead = [np.log(data_new_dead[i] + self.log_offset) for i in deaths_bootstrap_indices]
        predicted_dead = [np.log(new_deceased_from_sol[i + self.burn_in] + self.log_offset) for i in deaths_bootstrap_indices]

        new_tested_dists = [predicted_tested[i] - actual_tested[i] for i in range(len(predicted_tested))]
        new_dead_dists = [predicted_dead[i] - actual_dead[i] for i in range(len(predicted_dead))]

        # ensure the two delays are physical
        val1 = params['contagious_to_positive_delay']
        val2 = params['contagious_to_deceased_delay']
        err_from_reversed_delays = val1 - val2 if val1 > val2 else 0

        tested_vals = [data_new_tested[i] for i in cases_bootstrap_indices]
        deceased_vals = [data_new_dead[i] for i in deaths_bootstrap_indices]
        other_errs = [err_from_reversed_delays]

        return new_tested_dists, new_dead_dists, other_errs, sol, tested_vals, deceased_vals, \
               predicted_tested, actual_tested, predicted_dead, actual_dead
