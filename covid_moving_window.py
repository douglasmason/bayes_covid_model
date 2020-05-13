from sub_units.bayes_model_implementations.moving_window_model import \
    MovingWindowModel  # want to make an instance of this class for each state / set of params
from sub_units.utils import run_everything as run_everything_imported  # for plotting the report across all states
import sub_units.load_data as load_data  # only want to load this once, so import as singleton pattern

#####
# Set up model
#####

n_bootstraps = 100
n_likelihood_samples = 100000
moving_window_size = 21  # three weeks
max_date_str = '2020-05-12'
opt_calc = True
opt_force_plot = True
opt_simplified = True # just do statsmodels as a simplified service

state_models_filename = f'state_models_moving_window_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{max_date_str.replace("-", "_")}_max_date.joblib'
state_report_filename = f'state_report_moving_window_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{max_date_str.replace("-", "_")}_max_date.joblib'

# fixing parameters I don't want to train for saves a lot of computer power
extra_params = dict()
static_params = {'day0_positive_multiplier': 1,
                 'day0_deceased_multiplier': 1}
logarithmic_params = ['positive_intercept',
                      'deceased_intercept',
                      'sigma_positive',
                      'sigma_deceased',
                      # 'day0_positive_multiplier',
                      'day1_positive_multiplier',
                      'day2_positive_multiplier',
                      'day3_positive_multiplier',
                      'day4_positive_multiplier',
                      'day5_positive_multiplier',
                      'day6_positive_multiplier',
                      # 'day0_deceased_multiplier',
                      'day1_deceased_multiplier',
                      'day2_deceased_multiplier',
                      'day3_deceased_multiplier',
                      'day4_deceased_multiplier',
                      'day5_deceased_multiplier',
                      'day6_deceased_multiplier',
                      ]
exp_transform_param_names = logarithmic_params
plot_param_names = ['positive_slope',
                    'positive_intercept',
                    'deceased_slope',
                    'deceased_intercept',
                    'sigma_positive',
                    'sigma_deceased'
                    ]
if opt_simplified:
    plot_param_names = ['positive_slope',
                    'positive_intercept',
                    'deceased_slope',
                    'deceased_intercept']
sorted_init_condit_names = list()
sorted_param_names = ['positive_slope',
                      'positive_intercept',
                      'deceased_slope',
                      'deceased_intercept',
                    'sigma_positive',
                    'sigma_deceased',
                      # 'day0_positive_multiplier',
                      'day1_positive_multiplier',
                      'day2_positive_multiplier',
                      'day3_positive_multiplier',
                      'day4_positive_multiplier',
                      'day5_positive_multiplier',
                      'day6_positive_multiplier',
                      # 'day0_deceased_multiplier',
                      'day1_deceased_multiplier',
                      'day2_deceased_multiplier',
                      'day3_deceased_multiplier',
                      'day4_deceased_multiplier',
                      'day5_deceased_multiplier',
                      'day6_deceased_multiplier']

curve_fit_bounds = {'positive_slope': (-10, 10),
                    'positive_intercept': (0, 1000000),
                    'deceased_slope': (-10, 10),
                    'deceased_intercept': (0, 1000000),
                    'sigma_positive': (0, 100),
                    'sigma_deceased': (0, 100),
                    # 'day0_positive_multiplier': (0, 10),
                    'day1_positive_multiplier': (0, 10),
                    'day2_positive_multiplier': (0, 10),
                    'day3_positive_multiplier': (0, 10),
                    'day4_positive_multiplier': (0, 10),
                    'day5_positive_multiplier': (0, 10),
                    'day6_positive_multiplier': (0, 10),
                    # 'day0_deceased_multiplier': (0, 10),
                    'day1_deceased_multiplier': (0, 10),
                    'day2_deceased_multiplier': (0, 10),
                    'day3_deceased_multiplier': (0, 10),
                    'day4_deceased_multiplier': (0, 10),
                    'day5_deceased_multiplier': (0, 10),
                    'day6_deceased_multiplier': (0, 10)
                    }
test_params = {'positive_slope': 0,
               'positive_intercept': 2500,
               'deceased_slope': 0,
               'deceased_intercept': 250,
               'sigma_positive': 0.05,
               'sigma_deceased': 0.1,
               # 'day0_positive_multiplier': 1,
               'day1_positive_multiplier': 1,
               'day2_positive_multiplier': 1,
               'day3_positive_multiplier': 1,
               'day4_positive_multiplier': 1,
               'day5_positive_multiplier': 1,
               'day6_positive_multiplier': 1,
               # 'day0_deceased_multiplier': 1,
               'day1_deceased_multiplier': 1,
               'day2_deceased_multiplier': 1,
               'day3_deceased_multiplier': 1,
               'day4_deceased_multiplier': 1,
               'day5_deceased_multiplier': 1,
               'day6_deceased_multiplier': 1
               }

# uniform priors with bounds:
priors = curve_fit_bounds

# cycle over most populous states first
population_ranked_state_names = sorted(load_data.map_state_to_population.keys(),
                                       key=lambda x: -load_data.map_state_to_population[x])
run_states = population_ranked_state_names


def run_everything():
    return run_everything_imported(run_states,
                                   MovingWindowModel,
                                   max_date_str,
                                   load_data,
                                   state_models_filename=state_models_filename,
                                   state_report_filename=state_report_filename,
                                   moving_window_size=moving_window_size,
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
                                   opt_statsmodels=True,
                                   opt_simplified=opt_simplified
                                   )


if __name__ == '__main__':
    run_everything()
