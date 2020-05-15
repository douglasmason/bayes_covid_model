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
import os

from yattag import Doc



class Stopwatch:

    def __init__(self):
        self.time0 = get_time()

    def elapsed_time(self):
        return get_time() - self.time0

    def reset(self):
        self.time0 = get_time()


def render_whisker_plot_simplified(state_report,
                                   plot_param_name='alpha_2',
                                   output_filename_format_str='test_boxplot_for_{}_{}.png',
                                   opt_log=False,
                                   opt_param_type=[('SM', 'statsmodels')]):
    '''
    Plot all-state box/whiskers for given apram_name
    :param state_report: full state report as pandas dataframe
    :param param_name: param name as string
    :param output_filename_format_str: format string for the output filename, with two open slots
    :param opt_log: boolean for log-transform x-axis
    :return: None, it saves plots to files
    '''
    tmp_ind = [i for i, x in state_report.iterrows() if x['param'] == plot_param_name]
    tmp_ind = sorted(tmp_ind, key=lambda x: state_report.iloc[x]['SM_p50'])

    small_state_report = state_report.iloc[tmp_ind]
    small_state_report.to_csv('simplified_state_report_{}.csv'.format(plot_param_name))

    for param_name_abbr, param_name in opt_param_type:
        latex_str = small_state_report[
            [f'{param_name_abbr}_p5', f'{param_name_abbr}_p50', f'{param_name_abbr}_p95']].to_latex(index=False,
                                                                                                    float_format="{:0.4f}".format)
        print(param_name)
        print(latex_str)

    map_param_type_to_boxes = dict()
    for param_type_abbr, param_type in opt_param_type:
        tmp_list = list()
        for i in range(len(small_state_report)):
            row = pd.DataFrame([small_state_report.iloc[i]])
            new_box = \
                {
                    'label': 'param_type',
                    'whislo': row[f'{param_type_abbr}_p5'].values[0],  # Bottom whisker position
                    'q1': row[f'{param_type_abbr}_p25'].values[0],  # First quartile (25th percentile)
                    'med': row[f'{param_type_abbr}_p50'].values[0],  # Median         (50th percentile)
                    'q3': row[f'{param_type_abbr}_p75'].values[0],  # Third quartile (75th percentile)
                    'whishi': row[f'{param_type_abbr}_p95'].values[0],  # Top whisker position
                    'fliers': []  # Outliers
                }
            tmp_list.append(new_box)
        map_param_type_to_boxes[param_type_abbr] = tmp_list

    plt.close()
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 10.5)

    n_groups = len(map_param_type_to_boxes)
    
    map_param_type_to_ax = dict()
    for ind, param_type in enumerate(sorted(map_param_type_to_boxes)):
        map_param_type_to_ax[param_type] = ax.bxp(map_param_type_to_boxes[param_type], showfliers=False,
                                                  positions=range(1 + ind, len(map_param_type_to_boxes[param_type]) * (
                                                              n_groups + 1), (n_groups + 1)),
                                                  widths=0.7, patch_artist=True, vert=False)

    setup_boxes = map_param_type_to_boxes[list(map_param_type_to_boxes.keys())[0]]

    # plt.yticks([x + 0.5 for x in range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1))], small_state_report['state'])
    plt.yticks(range(1, len(setup_boxes) * (n_groups + 1), (n_groups + 1)), small_state_report['state'])

    # fill with colors
    colors = ['blue', 'red', 'green', 'purple']
    for param_type, color in zip(sorted(map_param_type_to_ax), colors):
        ax = map_param_type_to_ax[param_type]
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            for patch in ax[item]:
                try:
                    patch.set_facecolor(color)
                except:
                    pass
                patch.set_color(color)

    # add legend
    custom_lines = [
        Line2D([0], [0], color=color, lw=4) for param_type, color in zip(sorted(map_param_type_to_ax), colors)
    ]
    plt.legend(custom_lines, sorted(map_param_type_to_ax))

    # increase left margin
    output_filename = output_filename_format_str.format(plot_param_name, '_'.join(param_type[1] for param_type in opt_param_type))
    plt.subplots_adjust(left=0.2)
    if opt_log:
        plt.xscale('log')
    plt.savefig(output_filename, dpi=300)
    # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])


def render_whisker_plot(state_report,
                        param_name='alpha_2',
                        output_filename_format_str='test_boxplot_for_{}_{}.png',
                        opt_log=False,
                        opt_statsmodels=False,
                        opt_PyMC3=False,
                        boxwidth=0.7,
                        param_types_to_use=None):
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
                 widths=boxwidth, patch_artist=True, vert=False)
    # ax2 = ax.bxp(LS_boxes, showfliers=False, positions=range(2, len(LS_boxes) * (n_groups + 1), (n_groups + 1)),
    #              widths=1.2 / n_groups, patch_artist=True, vert=False)
    ax3 = ax.bxp(MCMC_boxes, showfliers=False, positions=range(2, len(MCMC_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=boxwidth, patch_artist=True, vert=False)

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
                 widths=boxwidth, patch_artist=True, vert=False)
    ax2 = ax.bxp(LS_boxes, showfliers=False, positions=range(2, len(LS_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=boxwidth, patch_artist=True, vert=False)
    ax3 = ax.bxp(MCMC_boxes, showfliers=False, positions=range(3, len(MCMC_boxes) * (n_groups + 1), (n_groups + 1)),
                 widths=boxwidth, patch_artist=True, vert=False)

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

        plt.close()
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 10.5)

        n_groups = 4
        ax1 = ax.bxp(BS_boxes, showfliers=False, positions=range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)
        ax2 = ax.bxp(LS_boxes, showfliers=False, positions=range(2, len(LS_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)
        ax3 = ax.bxp(MCMC_boxes, showfliers=False, positions=range(3, len(MCMC_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)
        ax4 = ax.bxp(SM_boxes, showfliers=False, positions=range(4, len(SM_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)

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

    if opt_PyMC3:

        PyMC3_boxes = list()
        for i in range(len(small_state_report)):
            row = pd.DataFrame([small_state_report.iloc[i]])
            new_box = \
                {
                    'label': 'PyMC3',
                    'whislo': row['PyMC3_p5'].values[0],  # Bottom whisker position
                    'q1': row['PyMC3_p25'].values[0],  # First quartile (25th percentile)
                    'med': row['PyMC3_p50'].values[0],  # Median         (50th percentile)
                    'q3': row['PyMC3_p75'].values[0],  # Third quartile (75th percentile)
                    'whishi': row['PyMC3_p95'].values[0],  # Top whisker position
                    'fliers': []  # Outliers
                }
            PyMC3_boxes.append(new_box)

        plt.close()
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 10.5)

        n_groups = 5
        ax1 = ax.bxp(BS_boxes, showfliers=False, positions=range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)
        ax2 = ax.bxp(LS_boxes, showfliers=False, positions=range(2, len(LS_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)
        ax3 = ax.bxp(MCMC_boxes, showfliers=False,
                     positions=range(3, len(MCMC_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)
        ax4 = ax.bxp(SM_boxes, showfliers=False, positions=range(4, len(SM_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)
        ax5 = ax.bxp(PyMC3_boxes, showfliers=False,
                     positions=range(4, len(PyMC3_boxes) * (n_groups + 1), (n_groups + 1)),
                     widths=boxwidth, patch_artist=True, vert=False)

        # plt.yticks([x + 0.5 for x in range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1))], small_state_report['state'])
        plt.yticks(range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1)), small_state_report['state'])

        # fill with colors
        colors = ['red', 'green', 'blue', 'purple', 'pink']
        for ax, color in zip((ax1, ax2, ax3, ax4, ax5), colors):
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
            Line2D([0], [0], color="pink", lw=4),
        ]
        plt.legend(custom_lines, ('Bootstraps', 'MVN', 'MCMC', 'Std. Errors', 'NUTS'))

        # increase left margin
        output_filename = output_filename_format_str.format(param_name, 'with_direct_samples_and_statsmodels_and_PyMC3')
        plt.subplots_adjust(left=0.2)
        if opt_log:
            plt.xscale('log')
        plt.savefig(output_filename, dpi=300)
        # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])


def generate_state_report(map_state_name_to_model,
                          state_report_filename=None,
                          report_names=None):
    state_report_as_list_of_dicts = list()
    for state_ind, state in enumerate(map_state_name_to_model):

        if state in map_state_name_to_model:
            state_model = map_state_name_to_model[state]
        else:
            print(f'Skipping {state}!')
            continue

        if state_model is not None:

            if report_names is None:
                report_names = state_model.sorted_names + list(state_model.extra_params.keys())

            try:
                LS_params, _, _, _ = state_model.get_weighted_samples()
            except:
                LS_params = [0]

            try:
                SM_params, _, _, _ = state_model.get_weighted_samples_via_statsmodels()
            except:
                SM_params = [0]

            try:
                PyMC3_params, _, _, _ = state_model.get_weighted_samples_via_PyMC3()
            except:
                PyMC3_params = [0]

            for param_name in report_names:
                if param_name in state_model.sorted_names:
                    try:
                        BS_vals = [state_model.bootstrap_params[i][param_name] for i in
                                   range(len(state_model.bootstrap_params))]
                    except:
                        pass
                    try:
                        LS_vals = [LS_params[i][state_model.map_name_to_sorted_ind[param_name]] for i in
                                   range(len(LS_params))]
                    except:
                        pass
                    try:
                        SM_vals = [SM_params[i][state_model.map_name_to_sorted_ind[param_name]] for i in
                                   range(len(SM_params))]
                    except:
                        pass

                    try:
                        PyMC3_vals = [PyMC3_params[i][state_model.map_name_to_sorted_ind[param_name]] for i in
                                      range(len(PyMC3_params))]
                    except:
                        pass
                    try:
                        MCMC_vals = [
                            state_model.all_random_walk_samples_as_list[i][
                                state_model.map_name_to_sorted_ind[param_name]]
                            for i
                            in
                            range(len(state_model.all_random_walk_samples_as_list))]
                    except:
                        pass
                else:
                    try:
                        BS_vals = [state_model.extra_params[param_name](
                            [state_model.bootstrap_params[i][key] for key in state_model.sorted_names]) for i in
                            range(len(state_model.bootstrap_params))]
                    except:
                        pass
                    try:
                        LS_vals = [state_model.extra_params[param_name](LS_params[i]) for i
                                   in range(len(LS_params))]
                    except:
                        pass
                    try:
                        SM_vals = [state_model.extra_params[param_name](SM_params[i]) for i
                                   in range(len(SM_params))]
                    except:
                        pass
                    try:
                        PyMC3_vals = [state_model.extra_params[param_name](state_model.extra_params[param_name](PyMC3_params[i])) for i
                                      in range(len(PyMC3_params))]
                    except:
                        pass
                    try:
                        MCMC_vals = [
                            state_model.extra_params[param_name](state_model.all_random_walk_samples_as_list[i])
                            for i
                            in range(len(state_model.all_random_walk_samples_as_list))]
                    except:
                        pass

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
                        'statsmodels_std_err_with_priors': np.std(SM_vals),
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

                try:
                    dict_to_add.update({
                        'PyMC3_mean_with_priors': np.average(PyMC3_vals),
                        'PyMC3_std_err_with_priors': np.std(PyMC3_vals),
                        'PyMC3_p50_with_priors': np.percentile(PyMC3_vals, 50),
                        'PyMC3_p5_with_priors':
                            np.percentile(PyMC3_vals, 5),
                        'PyMC3_p95_with_priors':
                            np.percentile(PyMC3_vals, 95),
                        'PyMC3_p25_with_priors':
                            np.percentile(PyMC3_vals, 25),
                        'PyMC3_p75_with_priors':
                            np.percentile(PyMC3_vals, 75)
                    })
                except:
                    pass

                state_report_as_list_of_dicts.append(dict_to_add)

    state_report = pd.DataFrame(state_report_as_list_of_dicts)
    print('Saving state report to {}...'.format(state_report_filename))
    joblib.dump(state_report, state_report_filename)
    print('...done!')
    print('Saving state report to {}...'.format(state_report_filename.replace('joblib', 'csv')))
    joblib.dump(state_report.to_csv(), state_report_filename.replace('joblib', 'csv'))
    print('...done!')
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
                   logarithmic_params=list(),
                   plot_param_names=None,
                   opt_statsmodels=False,
                   opt_simplified=False,
                   **kwargs):
    # setting intermediate variables to global allows us to inspect these objects via monkey-patching
    global map_state_name_to_model, state_report

    map_state_name_to_model = dict()

    for state_ind, state in enumerate(run_states):
        print(
            f'\n----\n----\nProcessing {state} ({state_ind} of {len(run_states)}, pop. {load_data.map_state_to_population[state]:,})...\n----\n----\n')

        try:
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
            if opt_simplified:
                state_model.run_fits_simplified()
            else:
                state_model.run_fits()
            map_state_name_to_model[state] = state_model

        except:
            print("Error with state", state)
            continue

        plot_subfolder = state_model.plot_subfolder
    
        if opt_simplified:
            state_report_filename = path.join(plot_subfolder, f'simplified_state_report.csv')
            filename_format_str = path.join(plot_subfolder, f'simplified_boxplot_for_{{}}_{{}}.png')
            if state_ind % 10 == 0 or state_ind == len(run_states) - 1:
                print('Reporting every 10th state and at the end')
                state_report = generate_state_report(map_state_name_to_model,
                                                     state_report_filename=state_report_filename,
                                                     report_names=plot_param_names)
                for param_name in state_model.plot_param_names:
                    render_whisker_plot_simplified(state_report,
                                                   plot_param_name=param_name,
                                                   output_filename_format_str=filename_format_str,
                                                   opt_log=param_name in logarithmic_params,
                                                   opt_param_type=state_model.simplified_model_param_type)
        else:
            state_report_filename = path.join(plot_subfolder, 'state_report.csv')
            filename_format_str = path.join(plot_subfolder, 'boxplot_for_{}_{}.png')
            if state_ind % 10 == 9 or state_ind == len(run_states) - 1:
                print('Reporting every 10th state and at the end')
                state_report = generate_state_report(map_state_name_to_model,
                                                     state_report_filename=state_report_filename)
                for param_name in state_model.plot_param_names:
                    render_whisker_plot(state_report,
                                        param_name=param_name,
                                        output_filename_format_str=filename_format_str,
                                        opt_log=param_name in logarithmic_params,
                                        opt_statsmodels=opt_statsmodels)


def generate_plot_browser(plot_browser_dir, load_data, base_url_dir, github_url, full_report_filename, list_of_figures,
                          list_of_figures_full_report):
    if not path.exists(plot_browser_dir):
        os.mkdir(plot_browser_dir)

    alphabetical_states = sorted(load_data.map_state_to_population.keys())
    alphabetical_states.remove('total')
    alphabetical_states = ['total'] + alphabetical_states

    map_state_to_html = dict()
    for state in alphabetical_states:

        state_lc = state.lower().replace(' ', '_')
        doc, tag, text = Doc(defaults={'title': f'Plots for {state}'}).tagtext()

        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('head'):
                pass
            with tag('body'):
                with tag('div', id='photo-container'):
                    with tag('ul'):
                        with tag('li'):
                            with tag('a', href='../index.html'):
                                text('<-- Back')
                        for figure_name in list_of_figures:
                            tmp_url = base_url_dir + state_lc + '/' + figure_name
                            with tag("a", href=tmp_url):
                                doc.stag('img', src=tmp_url, klass="photo", height="300", width="400")
                            with tag('li'):
                                with doc.tag("a", href=tmp_url):
                                    doc.text(figure_name)
                            with tag('hr'):
                                pass

        result = doc.getvalue()
        map_state_to_html[state] = result

    for state in map_state_to_html:
        state_lc = state.lower().replace(' ', '_')
        if not path.exists(path.join(plot_browser_dir, state_lc)):
            os.mkdir(path.join(plot_browser_dir, state_lc))
        with open(path.join(plot_browser_dir, path.join(state_lc, 'index.html')), 'w') as f:
            f.write(map_state_to_html[state])

    #####
    # Generate state-report page
    #####

    with open(path.join(plot_browser_dir, full_report_filename), 'w') as f:
        doc, tag, text = Doc(defaults={'title': f'Plots for Full U.S. Report'}).tagtext()
        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('head'):
                pass
            with tag('body'):
                with tag('div', id='photo-container'):
                    with tag('ul'):
                        with tag('li'):
                            with tag('a', href='index.html'):
                                text('<-- Back')
                        for figure_name in list_of_figures_full_report:
                            tmp_url = base_url_dir + figure_name
                            with tag("a", href=tmp_url):
                                doc.stag('img', src=tmp_url, klass="photo", height="400", width="300")
                            with tag('li'):
                                with doc.tag("a", href=tmp_url):
                                    doc.text(figure_name)
                            with tag('hr'):
                                pass
        f.write(doc.getvalue())

    #####
    # Generate landing page
    #####

    with open(path.join(plot_browser_dir, 'index.html'), 'w') as f:
        doc, tag, text = Doc(defaults={'title': f'Plots for {state}'}).tagtext()

        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('head'):
                pass
            with tag('body'):
                with tag('ul'):
                    with tag('li'):
                        with tag("a", href=github_url):
                            text('<-- Back to Repository')
                    with tag('li'):
                        with tag("a", href=full_report_filename):
                            text('Full U.S. Report')
                    for state in alphabetical_states:
                        state_lc = state.lower().replace(' ', '_')
                        tmp_url = state_lc + '/index.html'
                        with tag('li'):
                            with tag("a", href=tmp_url):
                                text(state)
        f.write(doc.getvalue())
