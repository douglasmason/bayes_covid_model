import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime

pd.plotting.register_matplotlib_converters()  # addresses complaints about Timestamp instead of float for plotting x-values
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import joblib
from scipy.stats import norm as sp_norm

plt.style.use('seaborn-darkgrid')
matplotlib.use('Agg')
from time import time as get_time
from os import path
import os
from enum import Enum
from yattag import Doc


class Region(Enum):
    US_states = 'US_states'
    countries = 'countries'
    US_counties = 'US_counties'
    provinces = 'provinces'

    def __str__(self):
        return str(self.value)


class ApproxType(Enum):
    __order__ = 'BS LS MCMC SM PyMC3 Hess SM_acc SM_TS SP_CF NDT_Hess NDT_Jac SP_min SP_LS NUTS'
    BS = ('BS', 'bootstrap')
    LS = ('LS', 'likelihood_samples')
    MCMC = ('MCMC', 'random_walk')
    SM = ('SM', 'statsmodels')
    PyMC3 = ('PyMC3', 'PyMC3')
    Hess = ('Hess', 'hessian')
    SM_acc = ('SM_acc', 'statsmodels_acc')
    SM_TS = ('SM_TS', 'statsmodels_time_series')
    SP_CF = ('SP_CF', 'curve_fit_covariance')
    NDT_Hess = ('NDT_Hess', 'numdifftools_hessian')
    NDT_Jac = ('NDT_Jac', 'numdifftools_jacobian')
    SP_min = ('SP_min', 'scipy_minimize')
    SP_LS = ('SP_LS', 'scipy_least_squares')
    NUTS = ('NUTS', 'no_u_turn_random_walk')

    def __str__(self):
        return str(self.value)


class Stopwatch:

    def __init__(self):
        self.time0 = get_time()

    def elapsed_time(self):
        return get_time() - self.time0

    def reset(self):
        self.time0 = get_time()


# def render_whisker_plot_split(state_report,
#                                    plot_param_name='alpha_2',
#                                    output_filename_format_str='test_boxplot_for_{}_{}.png',
#                                    opt_log=False,
#                                    boxwidth=0.7,
#                                    approx_types=[('SM', 'statsmodels')]):
#     
#     state_split = state_report['']
#     for page in range(n_pages):
#         if plot_param_name
#         render_whisker_plot_simplified(state_report,
#                                        plot_param_name='alpha_2',
#                                        output_filename_format_str=output_filename_format_str.replace('.png', f'_page_{page}_of_{n_pages}.png'),
#                                        opt_log=opt_log,
#                                        boxwidth=boxwidth,
#                                        approx_types=approx_types)

def render_whisker_plot_simplified(state_report,
                                   plot_param_name='alpha_2',
                                   output_filename_format_str='test_boxplot_for_{}_{}.png',
                                   opt_log=False,
                                   boxwidth=0.7,
                                   approx_types=[('SM', 'statsmodels')]):
    '''
    Plot all-state box/whiskers for given apram_name
    :param state_report: full state report as pandas dataframe
    :param param_name: param name as string
    :param output_filename_format_str: format string for the output filename, with two open slots
    :param opt_log: boolean for log-transform x-axis
    :return: None, it saves plots to files
    '''
    tmp_ind = [i for i, x in state_report.iterrows() if x['param'] == plot_param_name]
    print('columns:')
    print(state_report.columns)
    tmp_ind = sorted(tmp_ind, key=lambda x: state_report.iloc[x][f'{approx_types[0].value[0]}_p50'])

    small_state_report = state_report.iloc[tmp_ind]
    small_state_report.to_csv('simplified_state_report_{}.csv'.format(plot_param_name))

    if ApproxType.SM_TS in approx_types:
        approx_types.remove(ApproxType.SM_TS)

    for approx_type in approx_types:
        try:
            param_name_abbr, param_name = approx_type.value

            latex_str = small_state_report[
                [f'{param_name_abbr}_p5', f'{param_name_abbr}_p50', f'{param_name_abbr}_p95']].to_latex(index=False,
                                                                                                        float_format="{:0.4f}".format)
            print(param_name)
            print(latex_str)
        except:
            pass

    def convert_growth_rate_to_perc(x):
        return np.exp(x) - 1

    map_approx_type_to_boxes = dict()
    for approx_type in approx_types:
        try:
            param_name_abbr, param_name = approx_type.value
            tmp_list = list()
            for i in range(len(small_state_report)):
                row = pd.DataFrame([small_state_report.iloc[i]])
                new_box = \
                    {
                        'label': approx_type.value[1],
                        'whislo': convert_growth_rate_to_perc(row[f'{param_name_abbr}_p5'].values[0]),  # Bottom whisker position
                        'q1': convert_growth_rate_to_perc(row[f'{param_name_abbr}_p25'].values[0]),  # First quartile (25th percentile)
                        'med': convert_growth_rate_to_perc(row[f'{param_name_abbr}_p50'].values[0]),  # Median         (50th percentile)
                        'q3': convert_growth_rate_to_perc(row[f'{param_name_abbr}_p75'].values[0]),  # Third quartile (75th percentile)
                        'whishi': convert_growth_rate_to_perc(row[f'{param_name_abbr}_p95'].values[0]),  # Top whisker position
                        'fliers': []  # Outliers
                    }
                tmp_list.append(new_box)
            map_approx_type_to_boxes[param_name_abbr] = tmp_list
        except:
            pass

    plt.close()
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 10.5)

    # add vertical line at zero
    plt.axvline(linestyle='--', x=0, color='black')

    n_groups = len(map_approx_type_to_boxes)

    # when just one group, there's extra space between each state we need to account for
    if n_groups == 1:
        boxwidth = 1.2

    map_approx_type_to_ax = dict()
    for ind, approx_type in enumerate(sorted(map_approx_type_to_boxes)):
        try:
            map_approx_type_to_ax[approx_type] = ax.bxp(map_approx_type_to_boxes[approx_type], showfliers=False,
                                                        positions=range(1 + ind,
                                                                        len(map_approx_type_to_boxes[approx_type]) * (
                                                                                n_groups + 1), (n_groups + 1)),
                                                        widths=boxwidth, patch_artist=True, vert=False)
        except:
            pass

    setup_boxes = map_approx_type_to_boxes[list(map_approx_type_to_boxes.keys())[0]]

    # plt.yticks([x + 0.5 for x in range(1, len(BS_boxes) * (n_groups + 1), (n_groups + 1))], small_state_report['state'])

    plt.yticks(range(1, len(setup_boxes) * (n_groups + 1), (n_groups + 1)), small_state_report['state'])

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))

    # fill with colors
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'navy', 'teal', 'orchid', 'tan']
    for approx_type, color in zip(sorted(map_approx_type_to_ax), colors):
        try:
            ax = map_approx_type_to_ax[approx_type]
            for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                for patch in ax[item]:
                    try:
                        patch.set_facecolor(color)
                    except:
                        pass
                    patch.set_color(color)
        except:
            pass

    # add legend
    if n_groups > 1:
        custom_lines = [
            Line2D([0], [0], color=color, lw=4) for approx_type, color in zip(sorted(map_approx_type_to_ax), colors)
        ]
        plt.legend(custom_lines, sorted(map_approx_type_to_ax))

    # increase left margin
    output_filename = output_filename_format_str.format(plot_param_name,
                                                        '_'.join(approx_type.value[1] for approx_type in approx_types))
    plt.subplots_adjust(left=0.2)
    if opt_log:
        plt.xscale('log')
    
    if 'slope' in plot_param_name and ApproxType.SM_acc not in approx_types:
        plt.xlabel('Daily Relative Growth Rate')
    elif 'slope' in plot_param_name and ApproxType.SM_acc in approx_types:
        plt.xlabel('Absolute Week-over-Week Change in Daily Relative Growth Rate')
        
    plt.savefig(output_filename, dpi=300)
    # plt.boxplot(small_state_report['state'], small_state_report[['BS_p5', 'BS_p95']])


def generate_state_prediction(map_state_name_to_model,
                              override_max_date_str,
                              prediction_filename=None,
                              n_samples=1000):
    max_date = datetime.datetime.strptime(override_max_date_str, '%Y-%m-%d')
    all_predictions = list()
    for state_ind, state in enumerate(map_state_name_to_model):

        if state in map_state_name_to_model:
            state_model = map_state_name_to_model[state]
        else:
            print(f'Skipping {state}!')
            continue
        
        print(f'Running predictions on {state} ({state_ind} of {len(map_state_name_to_model)})')
        if state_model is not None:

            for approx_type in state_model.model_approx_types:

                if approx_type == ApproxType.SM or approx_type == ApproxType.SM_acc or approx_type == ApproxType.SM_TS:
                    params, _, _, log_probs = state_model.get_weighted_samples_via_statsmodels(n_samples=200)
                elif approx_type == ApproxType.PyMC3:
                    params, _, _, log_probs = state_model.get_weighted_samples_via_PyMC3(n_samples=200)
                print(approx_type)
                param_inds_to_plot = list(range(len(params)))
                param_inds_to_plot = np.random.choice(param_inds_to_plot, min(n_samples, len(param_inds_to_plot)),
                                                      replace=False)
                sols_to_plot = [state_model.run_simulation(in_params=params[param_ind]) for param_ind in
                                tqdm(param_inds_to_plot)]

                start_ind_sol = len(state_model.data_new_tested) + state_model.burn_in
                start_ind_data = start_ind_sol - 1 - state_model.burn_in
                sol_date_range = [
                    state_model.min_date - datetime.timedelta(days=state_model.burn_in) + datetime.timedelta(
                        days=1) * i for i in range(len(sols_to_plot[0][0]))]

                sols_to_plot_new_tested = list()
                sols_to_plot_new_dead = list()
                sols_to_plot_tested = list()
                sols_to_plot_dead = list()
                for sol in sols_to_plot:
                    tested = [max(sol[1][i] - state_model.log_offset, 0) for i in range(len(sol[1]))]
                    tested_range = np.cumsum(tested[start_ind_sol:])

                    dead = [max(sol[1][i] - state_model.log_offset, 0) for i in range(len(sol[1]))]
                    dead_range = np.cumsum(dead[start_ind_sol:])

                    sols_to_plot_new_tested.append(tested)
                    sols_to_plot_new_dead.append(dead)

                    data_tested_at_start = np.cumsum(state_model.data_new_tested)[start_ind_data]
                    data_dead_at_start = np.cumsum(state_model.data_new_dead)[start_ind_data]

                    tested = [0] * start_ind_sol + [data_tested_at_start + tested_val for tested_val in tested_range]
                    dead = [0] * start_ind_sol + [data_dead_at_start + dead_val for dead_val in dead_range]

                    sols_to_plot_tested.append(tested)
                    sols_to_plot_dead.append(dead)

                output_list_of_dicts = list()
                for date_ind in range(start_ind_sol, len(sols_to_plot_tested[0])):
                    distro_new_tested = [tested[date_ind] for tested in sols_to_plot_new_tested]
                    distro_new_dead = [dead[date_ind] for dead in sols_to_plot_new_dead]
                    distro_tested = [tested[date_ind] for tested in sols_to_plot_tested]
                    distro_dead = [dead[date_ind] for dead in sols_to_plot_dead]
                    tmp_dict = {'model_type': approx_type.value[1],
                                'date': sol_date_range[date_ind],
                                'total_positive_mean': np.average(distro_tested),
                                'total_positive_std': np.std(distro_tested),
                                'total_positive_p5': np.percentile(distro_tested, 5),
                                'total_positive_p25': np.percentile(distro_tested, 25),
                                'total_positive_p50': np.percentile(distro_tested, 50),
                                'total_positive_p75': np.percentile(distro_tested, 75),
                                'total_positive_p95': np.percentile(distro_tested, 95),
                                'total_deceased_mean': np.average(distro_dead),
                                'total_deceased_std': np.std(distro_dead),
                                'total_deceased_p5': np.percentile(distro_dead, 5),
                                'total_deceased_p25': np.percentile(distro_dead, 25),
                                'total_deceased_p50': np.percentile(distro_dead, 50),
                                'total_deceased_p75': np.percentile(distro_dead, 75),
                                'total_deceased_p95': np.percentile(distro_dead, 95),
                                'new_positive_mean': np.average(distro_new_tested),
                                'new_positive_std': np.std(distro_new_tested),
                                'new_positive_p5': np.percentile(distro_new_tested, 5),
                                'new_positive_p25': np.percentile(distro_new_tested, 25),
                                'new_positive_p50': np.percentile(distro_new_tested, 50),
                                'new_positive_p75': np.percentile(distro_new_tested, 75),
                                'new_positive_p95': np.percentile(distro_new_tested, 95),
                                'new_deceased_mean': np.average(distro_new_dead),
                                'new_deceased_std': np.std(distro_new_dead),
                                'new_deceased_p5': np.percentile(distro_new_dead, 5),
                                'new_deceased_p25': np.percentile(distro_new_dead, 25),
                                'new_deceased_p50': np.percentile(distro_new_dead, 50),
                                'new_deceased_p75': np.percentile(distro_new_dead, 75),
                                'new_deceased_p95': np.percentile(distro_new_dead, 95),
                                }
                    output_list_of_dicts.append(tmp_dict.copy())

                    tmp_dict.update({'state': state})
                    all_predictions.append(tmp_dict.copy())

    all_predictions = pd.DataFrame(all_predictions)

    if all([x.startswith('US: ') for x in all_predictions['state']]):
        all_predictions['state'] = [x[3:] for x in all_predictions['state']]

    # filter to just the first two weeks and every 1st of the month
    projection_dates = [max_date + datetime.timedelta(days=x) for x in range(1, 14)]
    for date in set(all_predictions['date']):
        if date.strftime('%Y-%m-%d').endswith('-01'):
            projection_dates.append(date)
    use_date_iloc = [i for i, x in enumerate(all_predictions['date']) if x in projection_dates]

    print('Saving state prediction to {}...'.format(prediction_filename))
    joblib.dump(all_predictions, prediction_filename)
    print('...done!')
    print('Saving state report to {}...'.format(prediction_filename.replace('joblib', 'csv')))
    joblib.dump(all_predictions.iloc[use_date_iloc].to_csv(), prediction_filename.replace('joblib', 'csv'))
    print('...done!')


def generate_state_report(map_state_name_to_model,
                          state_report_filename=None,
                          report_names=None,
                          opt_save_to_csv=True):
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
                LS_params, _, _, _ = state_model.get_weighted_samples_via_MVN()
            except:
                LS_params = [0]

            # try:
            #     SM_params, _, _, _ = state_model.get_weighted_samples_via_statsmodels()
            # except:
            #     SM_params = [0]

            positive_names = [name for name in state_model.sorted_names if 'positive' in name and 'sigma' not in name]
            map_name_to_sorted_ind_positive = {val: i for i, val in enumerate(positive_names)}
            deceased_names = [name for name in state_model.sorted_names if 'deceased' in name and 'sigma' not in name]
            map_name_to_sorted_ind_deceased = {val: i for i, val in enumerate(deceased_names)}

            try:
                SM_acc_params = state_model.map_param_to_acc
            except:
                SM_acc_params = [0]

            approx_type_params = dict()
            for approx_type in state_model.map_approx_type_to_model:
                if True:
                    approx_type_params[approx_type], _, _, _ = state_model.get_weighted_samples_via_model(
                        approx_type=approx_type)
                else:
                    approx_type_params[approx_type] = [0]

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

                    approx_type_vals = dict()
                    for approx_type in state_model.map_approx_type_to_model:
                        try:
                            approx_type_vals[approx_type] = [
                                approx_type_params[approx_type][i][state_model.map_name_to_sorted_ind[param_name]] for i
                                in
                                range(len(approx_type_params[approx_type]))]
                        except:
                            pass

                    # try:
                    #     SM_vals = [SM_params[i][state_model.map_name_to_sorted_ind[param_name]] for i in
                    #                range(len(SM_params))]
                    # except:
                    #     pass

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

                    # try:
                    #     SM_vals = [state_model.extra_params[param_name](SM_params[i]) for i
                    #                in range(len(SM_params))]
                    # except:
                    #     pass

                    approx_type_vals = dict()
                    for approx_type in state_model.map_approx_type_to_model:
                        try:
                            approx_type_vals[approx_type] = [
                                state_model.extra_params[param_name](approx_type_params[approx_type][i]) for i
                                in range(len(approx_type_params[approx_type]))]
                        except:
                            pass

                    try:
                        PyMC3_vals = [
                            state_model.extra_params[param_name](state_model.extra_params[param_name](PyMC3_params[i]))
                            for i
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
                    offset = SM_acc_params[param_name]['offset']
                    SM_acc_mean = SM_acc_params[param_name]['slope2'] - SM_acc_params[param_name]['slope1']
                    SM_acc_std_err = np.sqrt(
                        SM_acc_params[param_name]['bse1'] ** 2 + SM_acc_params[param_name]['bse2'] ** 2)
                    SM_acc_model = sp_norm(loc=SM_acc_mean, scale=SM_acc_std_err)

                    dict_to_add.update({
                        'statsmodels_acc_mean': SM_acc_mean,
                        'statsmodels_acc_std_err': SM_acc_std_err,
                        'statsmodels_acc_fwhm': SM_acc_std_err * 2.355,
                        'statsmodels_acc_p50': SM_acc_model.ppf(0.5),
                        'statsmodels_acc_p5': SM_acc_model.ppf(0.05),
                        'statsmodels_acc_p95': SM_acc_model.ppf(0.95),
                        'statsmodels_acc_p25': SM_acc_model.ppf(0.25),
                        'statsmodels_acc_p75': SM_acc_model.ppf(0.75),
                    })

                    dict_to_add.update({
                        f'statsmodels_mean_offset_{offset}_days': SM_acc_params[param_name]['slope1'],
                        f'statsmodels_mean_std_err_offset_{offset}_days': SM_acc_params[param_name]['bse1'],
                        # 'statsmodels_mean': SM_acc_params[param_name]['slope2'],
                        # 'statsmodels_mean_std_err': SM_acc_params[param_name]['bse2'],
                        'statsmodels_acc_z_score': SM_acc_params[param_name]['z_score'],
                        'statsmodels_acc_p_value': SM_acc_params[param_name]['p_value']
                    })
                except:
                    pass

                try:
                    dict_to_add.update({
                        'bootstrap_mean': np.average(BS_vals),
                        'bootstrap_p50': np.percentile(BS_vals, 50),
                        'bootstrap_p25':
                            np.percentile(BS_vals, 25),
                        'bootstrap_p75':
                            np.percentile(BS_vals, 75),
                        'bootstrap_p5': np.percentile(BS_vals, 5),
                        'bootstrap_p95': np.percentile(BS_vals, 95)
                    })
                except:
                    pass
                try:
                    dict_to_add.update({
                        'random_walk_mean': np.average(MCMC_vals),
                        'random_walk_p50': np.percentile(MCMC_vals, 50),
                        'random_walk_p5': np.percentile(MCMC_vals, 5),
                        'random_walk_p95': np.percentile(MCMC_vals, 95),
                        'random_walk_p25':
                            np.percentile(MCMC_vals, 25),
                        'random_walk_p75':
                            np.percentile(MCMC_vals, 75)
                    })
                except:
                    pass
                try:
                    dict_to_add.update({
                        'likelihood_samples_mean': np.average(LS_vals),
                        'likelihood_samples_p50': np.percentile(LS_vals, 50),
                        'likelihood_samples_p5':
                            np.percentile(LS_vals, 5),
                        'likelihood_samples_p95':
                            np.percentile(LS_vals, 95),
                        'likelihood_samples_p25':
                            np.percentile(LS_vals, 25),
                        'likelihood_samples_p75':
                            np.percentile(LS_vals, 75)
                    })
                except:
                    pass

                if hasattr(state_model, 'map_param_name_to_statsmodels_norm_model'):
                    if param_name in state_model.map_param_name_to_statsmodels_norm_model:
                        use_model = state_model.map_param_name_to_statsmodels_norm_model[param_name]
                        dict_to_add.update({
                            'statsmodels_mean': use_model.mean(),
                            'statsmodels_std_err': use_model.std(),
                            'statsmodels_fwhm': use_model.std() * 2.355,
                            'statsmodels_p50': use_model.ppf(0.5),
                            'statsmodels_p5': use_model.ppf(0.05),
                            'statsmodels_p95': use_model.ppf(0.95),
                            'statsmodels_p25': use_model.ppf(0.25),
                            'statsmodels_p75': use_model.ppf(0.75)
                        })
                    else:
                        pass
                    
                else:
                    pass

                # 
                # try:
                #     dict_to_add.update({
                #         'statsmodels_mean': np.average(SM_vals),
                #         'statsmodels_std_err': np.std(SM_vals),
                #         'statsmodels_fwhm': np.std(SM_vals) * 2.355,
                #         'statsmodels_p50': np.percentile(SM_vals, 50),
                #         'statsmodels_p5':
                #             np.percentile(SM_vals, 5),
                #         'statsmodels_p95':
                #             np.percentile(SM_vals, 95),
                #         'statsmodels_p25':
                #             np.percentile(SM_vals, 25),
                #         'statsmodels_p75':
                #             np.percentile(SM_vals, 75)
                #     })
                # except:
                #     pass

                for approx_type in approx_type_vals:
                    try:
                        dict_to_add.update({
                            f'{approx_type.value[0]}_mean': np.average(approx_type_vals[approx_type]),
                            f'{approx_type.value[0]}_std_err': np.std(approx_type_vals[approx_type]),
                            f'{approx_type.value[0]}_p50': np.percentile(approx_type_vals[approx_type], 50),
                            f'{approx_type.value[0]}_p5':
                                np.percentile(approx_type_vals[approx_type], 5),
                            f'{approx_type.value[0]}_p95':
                                np.percentile(approx_type_vals[approx_type], 95),
                            f'{approx_type.value[0]}_p25':
                                np.percentile(approx_type_vals[approx_type], 25),
                            f'{approx_type.value[0]}_p75':
                                np.percentile(approx_type_vals[approx_type], 75)
                        })
                    except:
                        pass

                try:
                    dict_to_add.update({
                        'PyMC3_mean': np.average(PyMC3_vals),
                        'PyMC3_std_err': np.std(PyMC3_vals),
                        'PyMC3_p50': np.percentile(PyMC3_vals, 50),
                        'PyMC3_p5':
                            np.percentile(PyMC3_vals, 5),
                        'PyMC3_p95':
                            np.percentile(PyMC3_vals, 95),
                        'PyMC3_p25':
                            np.percentile(PyMC3_vals, 25),
                        'PyMC3_p75':
                            np.percentile(PyMC3_vals, 75)
                    })
                except:
                    pass

                state_report_as_list_of_dicts.append(dict_to_add)

    state_report = pd.DataFrame(state_report_as_list_of_dicts)
    
    if opt_save_to_csv:
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
        for approx_type in ApproxType:
            col = col.replace(approx_type.value[1], approx_type.value[0])
        new_col = col.replace('__', '_')
        new_cols.append(new_col)
    state_report.columns = new_cols

    if all([x.startswith('US: ') for x in state_report['state']]):
        state_report['state'] = [x[3:] for x in state_report['state']]

    return state_report


def run_everything(run_states,
                   model_class,
                   load_data,
                   override_max_date_str=None,
                   sorted_init_condit_names=None,
                   sorted_param_names=None,
                   extra_params=None,
                   logarithmic_params=list(),
                   plot_param_names=None,
                   opt_simplified=False,
                   opt_report=True,
                   **kwargs):
    # setting intermediate variables to global allows us to inspect these objects via monkey-patching
    global map_state_name_to_model, state_report

    map_state_name_to_model = dict()

    for state_ind, state in enumerate(run_states):

        if state not in load_data.map_state_to_current_case_cnt:
            print(f'{state} not in load_data.map_state_to_current_case_cnt')
            # print('\n'.join(sorted(load_data.map_state_to_current_case_cnt.keys())))
        else:
            print(
                f'\n----\n----\nProcessing {state} ({state_ind} of {len(run_states)}, current cases {load_data.map_state_to_current_case_cnt[state]:,})...\n----\n----\n')
            try:
                print('Building model with the following args...')
                for key in sorted(kwargs.keys()):
                    print(f'{key}: {kwargs[key]}')
                state_model = model_class(state,
                                          sorted_init_condit_names=sorted_init_condit_names,
                                          sorted_param_names=sorted_param_names,
                                          extra_params=extra_params,
                                          logarithmic_params=logarithmic_params,
                                          plot_param_names=plot_param_names,
                                          opt_simplified=opt_simplified,
                                          override_max_date_str=override_max_date_str,
                                          **kwargs
                                          )
                if opt_simplified:
                    state_model.run_fits_simplified()
                else:
                    state_model.run_fits()
                map_state_name_to_model[state] = state_model
    
            except:
                print("Error getting model for state", state)
    
            plot_subfolder = state_model.plot_subfolder

        if not opt_report:
            continue
            
        if opt_simplified:
            state_report_filename = path.join(plot_subfolder, f'simplified_state_report.joblib')
            state_prediction_filename = path.join(plot_subfolder, f'simplified_state_prediction.joblib')
            filename_format_str = path.join(plot_subfolder, f'simplified_boxplot_for_{{}}_{{}}.png')
            if state_ind in [0, 4, 49, 499, 999, 1999, 2999] or state_ind > len(run_states) - 2:
                print(f'Reporting at index {state_ind} and at the end')
                state_report = generate_state_report(map_state_name_to_model,
                                                     state_report_filename=state_report_filename,
                                                     report_names=plot_param_names)
                if state_model.opt_plot:
                    _ = generate_state_prediction(map_state_name_to_model,
                                                  override_max_date_str,
                                                  prediction_filename=state_prediction_filename)
                for param_name in state_model.plot_param_names:
                    if len(set(state_report['state'])) <= 65 and state_model.opt_plot:
                        render_whisker_plot_simplified(state_report,
                                                       plot_param_name=param_name,
                                                       output_filename_format_str=filename_format_str,
                                                       opt_log=param_name in logarithmic_params,
                                                       approx_types=state_model.model_approx_types)
                        if ApproxType.SM in state_model.model_approx_types:
                            render_whisker_plot_simplified(state_report,
                                                           plot_param_name=param_name,
                                                           output_filename_format_str=filename_format_str,
                                                           opt_log=param_name in logarithmic_params,
                                                           approx_types=[ApproxType.SM_acc])
        else:
            state_report_filename = path.join(plot_subfolder, 'state_report.csv')
            filename_format_str = path.join(plot_subfolder, 'boxplot_for_{}_{}.png')
            if state_ind in [0, 1, 4, 9, 19, 49] or state_ind == len(run_states) - 1:
                print('Reporting now and at the end')
                state_report = generate_state_report(map_state_name_to_model,
                                                     state_report_filename=state_report_filename)
                for param_name in state_model.plot_param_names:
                    render_whisker_plot_simplified(state_report,
                                                   plot_param_name=param_name,
                                                   output_filename_format_str=filename_format_str,
                                                   opt_log=param_name in logarithmic_params,
                                                   approx_types=state_model.model_approx_types)

    return plot_subfolder


def generate_plot_browser(plot_browser_dir, base_url_dir, github_url, full_report_filename, list_of_figures,
                          list_of_figures_full_report, regions_to_present):
    if not path.exists(plot_browser_dir):
        os.mkdir(plot_browser_dir)

    alphabetical_states = sorted(regions_to_present)  # load_data.map_state_to_current_case_cnt.keys())
    # alphabetical_states.remove('total')
    # alphabetical_states = ['total'] + alphabetical_states

    map_state_to_html = dict()
    for state in alphabetical_states:

        if state.lower().startswith('us:_'):
            print_state = state[4:]
        else:
            print_state = state
        print_state = print_state.title().replace('_', ' ').replace(' Of', ' of')

        state_lc = state.lower().replace(' ', '_')
        doc, tag, text = Doc(defaults={'title': f'Plots for {state}'}).tagtext()

        doc.asis('<!DOCTYPE html>')
        with tag('html'):
            with tag('head'):
                pass
            with tag('body'):
                with tag('div', id='photo-container'):
                    with tag('h2'):
                        text(print_state)
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
        state_lc = state.lower().replace(' ', '_').replace(':', '')
        if not path.exists(path.join(plot_browser_dir, state_lc)):
            os.mkdir(path.join(plot_browser_dir, state_lc))
        with open(path.join(plot_browser_dir, path.join(state_lc, 'index.html')), 'w') as f:
            f.write(map_state_to_html[state])

    #####
    # Generate state-report page
    #####

    with open(path.join(plot_browser_dir, full_report_filename), 'w') as f:
        doc, tag, text = Doc(defaults={'title': f'Plots for Full Report'}).tagtext()
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
                            text(f'Full Report')
                    for state in alphabetical_states:

                        if state.lower().startswith('us:_'):
                            print_state = state[4:]
                        else:
                            print_state = state
                        print_state = print_state.title().replace('_', ' ').replace(' Of', ' of')

                        state_lc = state.lower().replace(' ', '_').replace(':', '')
                        tmp_url = state_lc + '/index.html'
                        with tag('li'):
                            with tag("a", href=tmp_url):
                                text(print_state)
        f.write(doc.getvalue())
