import numpy as np
import pandas as pd

pd.plotting.register_matplotlib_converters()  # addresses complaints about Timestamp instead of float for plotting x-values
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib

plt.style.use('seaborn-darkgrid')
matplotlib.use('Agg')
from time import time as get_time
from os import path


class Stopwatch:

    def __init__(self):
        self.time0 = get_time()

    def elapsed_time(self):
        return get_time() - self.time0

    def reset(self):
        self.time0 = get_time()


def render_whisker_plot(state_report,
                        param_name='alpha_2',
                        output_filename_format_str='test_boxplot_for_{}_{}.png',
                        opt_log=False,
                        opt_statsmodels=False):
    '''
    Plot all-state box/whiskers for given apram_name
    :param state_report: full state report as pandas dataframe
    :param param_name: param name as string
    :param output_filename_format_str: format string for the output filename, with two open slots
    :param opt_log: boolean for log-transform x-axis
    :return: None, it saves plots to files
    '''
    tmp_ind = [i for i, x in state_report.iterrows() if x['param'] == param_name]
    tmp_ind = sorted(tmp_ind, key=lambda x: state_report.iloc[x]['BS_p50'])

    small_state_report = state_report.iloc[tmp_ind]
    small_state_report.to_csv('state_report_{}.csv'.format(param_name))

    latex_str = small_state_report[['BS_p5', 'BS_p50', 'BS_p95']].to_latex(index=False, float_format="{:0.4f}".format)
    print(param_name)
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
    SM_boxes = list()
    for i in range(len(small_state_report)):
        row = pd.DataFrame([small_state_report.iloc[i]])
        new_box = \
            {
                'label': 'Statsmodels',
                'whislo': row['SM_p5'].values[0],  # Bottom whisker position
                'q1': row['SM_p25'].values[0],  # First quartile (25th percentile)
                'med': row['SM_p50'].values[0],  # Median         (50th percentile)
                'q3': row['SM_p75'].values[0],  # Third quartile (75th percentile)
                'whishi': row['SM_p95'].values[0],  # Top whisker position
                'fliers': []  # Outliers
            }
        SM_boxes.append(new_box)
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
    
    if opt_log:
        plt.xscale('log')
    plt.savefig(output_filename, dpi=300)
    # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])

    ######
    # Plots with MVN
    ######

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
    plt.legend(custom_lines, ('Bootstraps', 'MVN', 'MCMC'))

    # increase left margin
    output_filename = output_filename_format_str.format(param_name, 'with_direct_samples')
    plt.subplots_adjust(left=0.2)
    if opt_log:
        plt.xscale('log')
    plt.savefig(output_filename, dpi=300)
    # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])


    ######
    # Plots with Statsmodels
    ######
    
    if opt_statsmodels:

        plt.close()
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 10.5)
    
        n_groups = 4
        ax1 = ax.bxp(BS_boxes, showfliers=False, positions=range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=1.2 / n_groups, patch_artist=True, vert=False)
        ax2 = ax.bxp(LS_boxes, showfliers=False, positions=range(2, len(LS_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=1.2 / n_groups, patch_artist=True, vert=False)
        ax3 = ax.bxp(MCMC_boxes, showfliers=False, positions=range(3, len(MCMC_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=1.2 / n_groups, patch_artist=True, vert=False)
        ax4 = ax.bxp(SM_boxes, showfliers=False, positions=range(4, len(SM_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=1.2 / n_groups, patch_artist=True, vert=False)
    
        # plt.yticks([x + 0.5 for x in range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1))], small_state_report['state'])
        plt.yticks(range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)), small_state_report['state'])
    
        # fill with colors
        colors = ['red', 'green', 'blue', 'purple']
        for ax, color in zip((ax1, ax2, ax3, ax4), colors):
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
            Line2D([0], [0], color="purple", lw=4),
        ]
        plt.legend(custom_lines, ('Bootstraps', 'MVN', 'MCMC', 'Std. Errors'))
    
        # increase left margin
        output_filename = output_filename_format_str.format(param_name, 'with_direct_samples_and_statsmodels')
        plt.subplots_adjust(left=0.2)
        if opt_log:
            plt.xscale('log')
        plt.savefig(output_filename, dpi=300)
        # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])


def generate_whisker_plots(state_report,
                           output_filename_format_str=None,
                           param_names=None,
                           logarithmic_params=list(),
                           opt_statsmodels=False):
    for param_name in param_names:
        render_whisker_plot(state_report,
                            param_name=param_name,
                            output_filename_format_str=output_filename_format_str,
                            opt_log=param_name in logarithmic_params,
                            opt_statsmodels=opt_statsmodels)


def generate_state_report(map_state_name_to_model,
                          state_report_filename=None):
    state_report_as_list_of_dicts = list()
    for state_ind, state in enumerate(map_state_name_to_model):

        if state in map_state_name_to_model:
            state_model = map_state_name_to_model[state]
        else:
            print(f'Skipping {state}!')
            continue

        # try:
        #     frac_bootstraps_used_after_prior = sum(state_model.bootstrap_weights) / state_model.n_bootstraps
        # except:
        #     frac_bootstraps_used_after_prior = -1

        # print(
        #     f'\n----\nResults for {state} ({state_ind} of {len(map_state_name_to_model)}, pop. {load_data.map_state_to_population[state]:,}, {frac_bootstraps_used_after_prior * 100:.4g} bootstraps used after prior applied)...\n----')

        try:
            _ = [
                state_model.bootstrap_params,
                state_model.all_random_walk_samples_as_list,
                # state_model.all_samples_as_list,
            ]
            print('got all vals to retrieve')
        except:
            print('Not all vals to retrieve present!')
            continue

        if state_model is not None:

            try:
                LS_params, _, _, _ = state_model.get_weighted_samples()
            except:
                LS_params = [0]
                

            try:
                SM_params, _, _, _ = state_model.get_weighted_samples_via_statsmodels()
            except:
                SM_params = [0]

            for param_name in state_model.sorted_names + list(state_model.extra_params.keys()):
                if param_name in state_model.sorted_names:
                    BS_vals = [state_model.bootstrap_params[i][param_name] for i in
                               range(len(state_model.bootstrap_params))]
                    LS_vals = [LS_params[i][state_model.map_name_to_sorted_ind[param_name]] for i in
                               range(len(LS_params))]
                    SM_vals = [SM_params[i][state_model.map_name_to_sorted_ind[param_name]] for i in
                               range(len(SM_params))]
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
                    SM_vals = [state_model.extra_params[param_name](SM_params[i]) for i
                               in range(len(SM_params))]
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
                try:
                    dict_to_add.update({
                        'statsmodels_mean_with_priors': np.average(SM_vals),
                        'statsmodels_p50_with_priors': np.percentile(SM_vals, 50),
                        'statsmodels_p5_with_priors':
                            np.percentile(SM_vals, 5),
                        'statsmodels_p95_with_priors':
                            np.percentile(SM_vals, 95),
                        'statsmodels_p25_with_priors':
                            np.percentile(SM_vals, 25),
                        'statsmodels_p75_with_priors':
                            np.percentile(SM_vals, 75)
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
            .replace('statsmodels', 'SM') \
            .replace('__', '_')
        new_cols.append(new_col)
    state_report.columns = new_cols

    return state_report


def run_everything(run_states,
                   model_class,
                   max_date_str,
                   load_data,
                   sorted_init_condit_names=None,
                   sorted_param_names=None,
                   extra_params=None,
                   state_report_filename=None,
                   logarithmic_params=list(),
                   plot_param_names=None,
                   opt_statsmodels=False,
                   **kwargs):
    # setting intermediate variables to global allows us to inspect these objects via monkey-patching
    global map_state_name_to_model, state_report

    map_state_name_to_model = dict()

    for state_ind, state in enumerate(run_states):
        print(
            f'\n----\n----\nProcessing {state} ({state_ind} of {len(run_states)}, pop. {load_data.map_state_to_population[state]:,})...\n----\n----\n')

        if True:
            print('Building model with the following args...')
            for key in sorted(kwargs.keys()):
                print(f'{key}: {kwargs[key]}')
            state_model = model_class(state,
                                      max_date_str,
                                      sorted_init_condit_names=sorted_init_condit_names,
                                      sorted_param_names=sorted_param_names,
                                      extra_params=extra_params,
                                      logarithmic_params=logarithmic_params,
                                      plot_param_names=plot_param_names,
                                      **kwargs
                                      )
            state_model.run_fits()
            map_state_name_to_model[state] = state_model

        else:
            print("Error with state", state)
            continue

        state_report = generate_state_report(map_state_name_to_model,
                                             state_report_filename=state_report_filename)

        plot_subfolder = state_model.plot_subfolder
        param_names = sorted_init_condit_names + sorted_param_names + list(extra_params.keys())
        filename = path.join(plot_subfolder, f'boxplot_for_{{}}_{{}}.png')
        generate_whisker_plots(state_report, output_filename_format_str=filename,
                               param_names=plot_param_names, logarithmic_params=logarithmic_params,
                               opt_statsmodels=opt_statsmodels)
