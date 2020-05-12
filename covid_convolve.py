from sub_units.bayes_model import \
    ConvolutionModel  # want to make an instance of this class for each state / set of params
from sub_units.utils import run_everything as run_everything_imported  # for plotting the report across all states
import sub_units.load_data as load_data  # only want to load this once, so import as singleton pattern

#####
# Set up model
#####

n_bootstraps = 1000
n_likelihood_samples = 100000
max_date_str = '2020-05-09'
opt_calc = True
opt_force_plot = False

state_models_filename = f'state_models_convolution_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{max_date_str.replace("-", "_")}_max_date.joblib'
state_report_filename = f'state_report_convolution_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{max_date_str.replace("-", "_")}_max_date.joblib'

# fixing parameters I don't want to train for saves a lot of computer power
static_params = {'contagious_to_positive_width': 7,
                 'contagious_to_deceased_width': 7,
                 'contagious_to_positive_mult': 0.1}
logarithmic_params = ['I_0', 'contagious_to_deceased_mult']
sorted_init_condit_names = ['I_0']
sorted_param_names = ['alpha_1',
                      'alpha_2',
                      'contagious_to_positive_delay',
                      'contagious_to_deceased_delay',
                      'contagious_to_deceased_mult'
                      ]
plot_param_names = ['alpha_1', 'alpha_2', 'contagious_to_positive_delay',
                                           'positive_to_deceased_delay',
                                           'positive_to_deceased_mult']


def get_positive_to_deceased_delay(x, map_name_to_sorted_ind=None):
    return x[map_name_to_sorted_ind['contagious_to_deceased_delay']] - x[
        map_name_to_sorted_ind['contagious_to_positive_delay']]


def get_positive_to_deceased_mult(x, map_name_to_sorted_ind=None):
    return x[map_name_to_sorted_ind['contagious_to_deceased_mult']] / 0.1


extra_params = {
    'positive_to_deceased_delay': get_positive_to_deceased_delay,
    'positive_to_deceased_mult': get_positive_to_deceased_mult
}

curve_fit_bounds = {'I_0': (1e-12, 100.0),  # starting infections
                    'alpha_1': (-1, 2),
                    'alpha_2': (-1, 2),
                    'contagious_to_positive_delay': (-14, 21),
                    'contagious_to_positive_width': (0, 14),
                    'contagious_to_deceased_delay': (-14, 42),
                    'contagious_to_deceased_width': (0, 14),
                    'contagious_to_deceased_mult': (1e-12, 1),
                    }

test_params = {'I_0': 2e-3,  # starting infections
               'alpha_1': 0.23,
               'alpha_2': 0.01,
               'contagious_to_positive_delay': 9,
               'contagious_to_positive_width': 7,
               'contagious_to_deceased_delay': 15,
               'contagious_to_deceased_width': 7,
               'contagious_to_deceased_mult': 0.01,
               }

# uniform priors with bounds:
priors = {'I_0': (1e-12, 1e2),  # starting infections
          'alpha_1': (0, 1),
          'alpha_2': (-0.5, 0.5),
          'contagious_to_positive_delay': (-10, 20),
          'contagious_to_positive_width': (-2, 14),
          'contagious_to_deceased_delay': (-10, 30),
          'contagious_to_deceased_width': (1, 17),
          'contagious_to_deceased_mult': (1e-6, 0.1),
          }

# cycle over most populous states first
population_ranked_state_names = sorted(load_data.map_state_to_population.keys(),
                                       key=lambda x: -load_data.map_state_to_population[x])
run_states = population_ranked_state_names


####
# Make whisker plots
####
def run_everything():
    return run_everything_imported(run_states,
                                   ConvolutionModel,
                                   max_date_str,
                                   load_data,
                                   sorted_init_condit_names,
                                   sorted_param_names,
                                   extra_params,
                                   n_bootstraps=n_bootstraps,
                                   n_likelihood_samples=n_likelihood_samples,
                                   load_data_obj=load_data,
                                   sorted_param_names=sorted_param_names,
                                   sorted_init_condit_names=sorted_init_condit_names,
                                   curve_fit_bounds=curve_fit_bounds,
                                   priors=priors,
                                   test_params=test_params,
                                   static_params=static_params,
                                   opt_calc=opt_calc,
                                   opt_force_plot=opt_force_plot,
                                   logarithmic_params=logarithmic_params,
                                   extra_params=extra_params,
                                   plot_param_names=plot_param_names,
                                   )


if __name__ == '__main__':
    run_everything()
