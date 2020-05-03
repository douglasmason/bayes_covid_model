from sub_units.utils import ConvolutionModel # want to make an instance of this class for each state / set of params
import sub_units.load_data as load_data # only want to load this once, so import as singleton pattern
import pandas as pd
import numpy as np

#####
# Set up model
#####

n_bootstraps = 1000
n_likelihood_samples = 1000
max_date_str = '2020-05-01'

# fixing parameters I don't want to train for saves a lot of computer power
static_params = {'contagious_to_positive_width': 7,
                 'contagious_to_deceased_width': 7}

sorted_init_condit_names = ['I_0']
sorted_param_names = [ 'alpha_1',
                       'alpha_2',
                       'contagious_to_positive_delay',
                       'contagious_to_positive_width',
                       #'contagious_to_positive_mult',
                       'contagious_to_deceased_delay',
                       'contagious_to_deceased_width',
                       'contagious_to_deceased_mult'
                      ]

curve_fit_bounds = {'I_0': (-100.0, 100.0),  # starting infections
           'alpha_1': (-1, 2),
           'alpha_2': (-1, 2),
           'contagious_to_positive_delay': (-14, 21),
           'contagious_to_positive_width': (0, 14),
           #'contagious_to_positive_mult': (0, 2),
           'contagious_to_deceased_delay': (-14, 42),
           'contagious_to_deceased_width': (0, 14),
           'contagious_to_deceased_mult': (0, 1),
           }


test_params = {'I_0': 2e-3,  # starting infections
               'alpha_1': 0.23,
               'alpha_2': 0.01,
               'contagious_to_positive_delay': 9,
               'contagious_to_positive_width': 7,
               #'contagious_to_positive_mult': 0.1,
               'contagious_to_deceased_delay': 15,
               'contagious_to_deceased_width': 7,
               'contagious_to_deceased_mult': 0.01,
               }

# uniform priors with bounds:
priors = {'I_0': (-10.0, 10.0),  # starting infections
          'alpha_1': (0, 1),
          'alpha_2': (-0.5, 0.5),
          'contagious_to_positive_delay': (-10, 20),
          'contagious_to_positive_width': (1, 14),
          # 'contagious_to_positive_mult': (0, 2),
          'contagious_to_deceased_delay': (-10, 30),
          'contagious_to_deceased_width': (1, 17),
          'contagious_to_deceased_mult': (0, 0.05),
          }


tight_bounds = {'I_0': (1e-6, 1e-4),  # starting infections
          'alpha_1': (0.2, 0.4),
          'alpha_2': (-0.05, 0.05),
          'contagious_to_positive_delay': (6, 12),
          'contagious_to_positive_width': (1, 10),
          # 'contagious_to_positive_mult': (0, 2),
          'contagious_to_deceased_delay': (14, 20),
          'contagious_to_deceased_width': (1, 12),
          'contagious_to_deceased_mult': (0.005, 0.01),
          }

medium_bounds = {'I_0': (1e-3, 1e-3),  # starting infections
          'alpha_1': (0.1, 0.5),
          'alpha_2': (-0.1, 0.1),
          'contagious_to_positive_delay': (3, 15),
          'contagious_to_positive_width': (1, 20),
          # 'contagious_to_positive_mult': (0, 2),
          'contagious_to_deceased_delay': (5, 30),
          'contagious_to_deceased_width': (1, 30),
          'contagious_to_deceased_mult': (0.002, 0.02),
          }

loose_bounds = {'I_0': (-100.0, 100.0),  # starting infections
           'alpha_1': (-1, 2),
           'alpha_2': (-1, 2),
           'contagious_to_positive_delay': (0, 100),
           'contagious_to_positive_width': (0, 20),
           #'contagious_to_positive_mult': (0, 2),
           'contagious_to_deceased_delay': (0, 100),
           'contagious_to_deceased_width': (0, 20),
           'contagious_to_deceased_mult': (0, 1),
           }

#####
# Loop over states
#####

map_state_name_to_model = dict()

# cycle over most populous states first
population_ranked_state_names = sorted(load_data.map_state_to_population.keys(), key=lambda x: -load_data.map_state_to_population[x])

for state_ind, state in enumerate(population_ranked_state_names):
    print(f'\n----\nProcessing {state} ({state_ind} of {len(population_ranked_state_names)}, pop. {load_data.map_state_to_population[state]:,})...\n----')
    
    try:
        state_model = ConvolutionModel(state,
                                       max_date_str,
                                       n_bootstraps=n_bootstraps,
                                       n_likelihood_samples=n_likelihood_samples,
                                       load_data_obj=load_data,
                                       sorted_param_names=sorted_param_names,
                                       sorted_init_condit_names=sorted_init_condit_names,
                                       curve_fit_bounds=curve_fit_bounds,
                                       tight_bounds=tight_bounds,
                                       medium_bounds=medium_bounds,
                                       loose_bounds=loose_bounds,
                                       priors=priors,
                                       test_params=test_params,
                                       static_params=static_params
                                       )
        
        state_model.run_fits()
        
    except Exception as ee:
        print("Whoops! An error occurred:")
        print(ee)
        state_model = None

    map_state_name_to_model[state] = state_model


state_report_as_list_of_dicts = list()
for state in population_ranked_state_names:
    
    if state in map_state_name_to_model:
        state_model = map_state_name_to_model[state]
    else:
        print(f'Skipping {state}!')
        continue
    
    try:
        frac_bootstraps_used_after_prior = sum(state_model.bootstrap_weights)/state_model.n_bootstraps
    except:
        frac_bootstraps_used_after_prior = -1
        
    print(f'\n----\nResults for {state} ({state_ind} of {len(population_ranked_state_names)}, pop. {load_data.map_state_to_population[state]:,}, {frac_bootstraps_used_after_prior*100:.4g} bootstraps used after prior applied)...\n----')
    
    try:
        vals_to_retrieve = [
            state_model.bootstrap_means_with_priors,
            state_model.bootstrap_cred_int_with_priors,
            state_model.GMM_means,
            state_model.GMM_confidence_intervals]
        for tmp_dict in vals_to_retrieve:
            state_model.pretty_print_params(tmp_dict)
    except:
        print('Not all vals_to_retrieve present!')
    
    if state_model is not None:
        for param_name in state_model.sorted_names:
            dict_to_add = {'state': state,
                           'param': param_name,
                           'bootstrap_mean_with_priors': state_model.bootstrap_means_with_priors[param_name],
                           'bootstrap_p5_with_priors': state_model.bootstrap_cred_int_with_priors[param_name][0],
                           'bootstrap_p95_with_priors': state_model.bootstrap_cred_int_with_priors[param_name][1],
                           'GMM_mean_with_priors': state_model.GMM_means[param_name],
                           'GMM_p5_with_priors': state_model.GMM_confidence_intervals[param_name][0],
                           'GMM_p95_with_priors': state_model.GMM_confidence_intervals[param_name][1],
                           }
            state_report_as_list_of_dicts.append(dict_to_add)

state_report = pd.DataFrame(state_report_as_list_of_dicts)

tmp_ind = [i for i, x in state_report.iterrows() if x['param'] == 'alpha_2']
tmp_ind = sorted(tmp_ind, key=lambda x: state_report.iloc[x]['bootstrap_mean_with_priors'])
small_state_report = state_report.iloc[tmp_ind]
small_state_report.to_csv('state_report_alpha_2.csv')



######
# Future work
######
# TODO:
#   collect the confidence/credible intervals into a table
#   Do I want to make the 90%/95% aspect tunable?
#   Do I want confidence intervals after applying priors?
#   Also, post my RealReal stuff on Github!
