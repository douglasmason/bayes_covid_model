from sub_units.utils import ConvolutionModel  # want to make an instance of this class for each state / set of params
from sub_units.utils import render_whisker_plot  # for plotting the report across all states
import sub_units.load_data as load_data  # only want to load this once, so import as singleton pattern
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

#####
# Set up model
#####

n_bootstraps = 10
n_likelihood_samples = 20000
max_date_str = '2020-05-06'
opt_calc = True
opt_force_plot = False

state_models_filename = f'state_models_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{max_date_str.replace("-", "_")}_max_date.joblib'
state_report_filename = f'state_report_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{max_date_str.replace("-", "_")}_max_date.joblib'

# fixing parameters I don't want to train for saves a lot of computer power
static_params = {'contagious_to_positive_width': 7,
                 'contagious_to_deceased_width': 7}
logarithmic_params = ['I_0', 'contagious_to_deceased_mult']
sorted_init_condit_names = ['I_0']
sorted_param_names = ['alpha_1',
                      'alpha_2',
                      'contagious_to_positive_delay',
                      'contagious_to_positive_width',
                      # 'contagious_to_positive_mult',
                      'contagious_to_deceased_delay',
                      'contagious_to_deceased_width',
                      'contagious_to_deceased_mult'
                      ]


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
                    # 'contagious_to_positive_mult': (0, 2),
                    'contagious_to_deceased_delay': (-14, 42),
                    'contagious_to_deceased_width': (0, 14),
                    'contagious_to_deceased_mult': (1e-12, 1),
                    }

test_params = {'I_0': 2e-3,  # starting infections
               'alpha_1': 0.23,
               'alpha_2': 0.01,
               'contagious_to_positive_delay': 9,
               'contagious_to_positive_width': 7,
               # 'contagious_to_positive_mult': 0.1,
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
          # 'contagious_to_positive_mult': (0, 2),
          'contagious_to_deceased_delay': (-10, 30),
          'contagious_to_deceased_width': (1, 17),
          'contagious_to_deceased_mult': (1e-6, 0.1),
          }

# cycle over most populous states first
population_ranked_state_names = sorted(load_data.map_state_to_population.keys(),
                                       key=lambda x: -load_data.map_state_to_population[x])
run_states = population_ranked_state_names[38:]

#####
# Loop over states
#####

def loop_over_over_states(run_states):
    map_state_name_to_model = dict()

    try:
        for state_ind, state in enumerate(run_states):
            print(
                f'\n----\n----\nProcessing {state} ({state_ind} of {len(run_states)}, pop. {load_data.map_state_to_population[state]:,})...\n----\n----\n')
    
            try:
                state_model = ConvolutionModel(state,
                                               max_date_str,
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
                                               extra_params=extra_params
                                               )
                state_model.run_fits()
    
                test_params = state_model.fit_curve_via_likelihood(state_model.all_data_params)
                state_model.solve_and_plot_solution(test_params)
                state_model.get_log_likelihood(test_params)
    
                map_state_name_to_model[state] = state_model
    
            except:
                print("Error with state", state)
                continue
    except:
        return map_state_name_to_model
    
    return map_state_name_to_model
        

    # commented out bc this takes way too long
    # print(f'Saving {len(map_state_name_to_model)} state models to {state_models_filename}')
    # joblib.dump(map_state_name_to_model, state_models_filename)


#####
# Now post-process
#####

# map_state_name_to_model = joblib.load(state_models_filename)

def generate_state_report(map_state_name_to_model):
    
    state_report_as_list_of_dicts = list()
    for state_ind, state in enumerate(population_ranked_state_names):

        if state in map_state_name_to_model:
            state_model = map_state_name_to_model[state]
        else:
            print(f'Skipping {state}!')
            continue

        try:
            frac_bootstraps_used_after_prior = sum(state_model.bootstrap_weights) / state_model.n_bootstraps
        except:
            frac_bootstraps_used_after_prior = -1

        print(
            f'\n----\nResults for {state} ({state_ind} of {len(population_ranked_state_names)}, pop. {load_data.map_state_to_population[state]:,}, {frac_bootstraps_used_after_prior * 100:.4g} bootstraps used after prior applied)...\n----')

        try:
            vals_to_retrieve = [
                state_model.bootstrap_params,
                state_model.all_random_walk_samples_as_list,
                # state_model.all_samples_as_list,
            ]
            print('got all vals_to_retrieve')
        except:
            print('Not all vals_to_retrieve present!')
            continue

        if state_model is not None:

            try:
                LS_params, _, _, _ = state_model._get_weighted_samples()
            except:
                LS_params = [0]

            for param_name in state_model.sorted_names + list(state_model.extra_params.keys()):
                if param_name in state_model.sorted_names:
                    BS_vals = [state_model.bootstrap_params[i][param_name] for i in
                               range(len(state_model.bootstrap_params))]
                    LS_vals = [LS_params[i][state_model.map_name_to_sorted_ind[param_name]] for i in
                               range(len(LS_params))]
                    MCMC_vals = [
                        state_model.all_random_walk_samples_as_list[i][state_model.map_name_to_sorted_ind[param_name]]
                        for i
                        in
                        range(len(state_model.all_random_walk_samples_as_list))]
                else:
                    BS_vals = [state_model.extra_params[param_name](
                        [state_model.bootstrap_params[i][key] for key in state_model.sorted_names]) for i in
                        range(len(state_model.bootstrap_params))]
                    LS_vals = [state_model.extra_params[param_name](LS_params[i]) for i
                               in range(len(LS_params))]
                    MCMC_vals = [state_model.extra_params[param_name](state_model.all_random_walk_samples_as_list[i])
                                 for i
                                 in range(len(state_model.all_random_walk_samples_as_list))]

                dict_to_add = {'state': state,
                               'param': param_name
                               }

                try:
                    dict_to_add.update({
                        'bootstrap_mean_with_priors': np.average(BS_vals),
                        'bootstrap_p50_with_priors': np.percentile(BS_vals, 50),
                        'bootstrap_p25_with_priors':
                            np.percentile(BS_vals, 25),
                        'bootstrap_p75_with_priors':
                            np.percentile(BS_vals, 75),
                        'bootstrap_p5_with_priors': np.percentile(BS_vals, 5),
                        'bootstrap_p95_with_priors': np.percentile(BS_vals, 95)
                    })
                except:
                    pass
                try:
                    dict_to_add.update({
                        'random_walk_mean_with_priors': np.average(MCMC_vals),
                        'random_walk_p50_with_priors': np.percentile(MCMC_vals, 50),
                        'random_walk_p5_with_priors': np.percentile(MCMC_vals, 5),
                        'random_walk_p95_with_priors': np.percentile(MCMC_vals, 95),
                        'random_walk_p25_with_priors':
                            np.percentile(MCMC_vals, 25),
                        'random_walk_p75_with_priors':
                            np.percentile(MCMC_vals, 75)
                    })
                except:
                    pass
                try:
                    dict_to_add.update({
                        'likelihood_samples_mean_with_priors': np.average(LS_vals),
                        'likelihood_samples_p50_with_priors': np.percentile(LS_vals, 50),
                        'likelihood_samples_p5_with_priors':
                            np.percentile(LS_vals, 5),
                        'likelihood_samples_p95_with_priors':
                            np.percentile(LS_vals, 95),
                        'likelihood_samples_p25_with_priors':
                            np.percentile(LS_vals, 25),
                        'likelihood_samples_p75_with_priors':
                            np.percentile(LS_vals, 75)
                    })
                except:
                    pass
                state_report_as_list_of_dicts.append(dict_to_add)

    state_report = pd.DataFrame(state_report_as_list_of_dicts)
    print('Saving state report to {}'.format(state_report_filename))
    joblib.dump(state_report, state_report_filename)
    n_states = len(set(state_report['state']))
    print(n_states)

    new_cols = list()
    for col in state_report.columns:
        new_col = col.replace('bootstrap', 'BS') \
            .replace('_with_priors', '') \
            .replace('likelihood_samples', 'LS') \
            .replace('random_walk', 'MCMC') \
            .replace('__', '_')
        new_cols.append(new_col)
    state_report.columns = new_cols
    
    return state_report


####
# Make whisker plots
####

def generate_whisker_plots(state_report):
    for param_name in sorted_init_condit_names + sorted_param_names + list(extra_params.keys()):
        render_whisker_plot(state_report, param_name=param_name)

def run_everything():
    map_state_name_to_model = loop_over_over_states(run_states)
    state_report = generate_state_report(map_state_name_to_model)
    generate_whisker_plots(state_report)

if __name__ == '__main__':
    run_everything()