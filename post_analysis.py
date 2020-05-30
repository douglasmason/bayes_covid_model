import os
import pandas as pd
import numpy as np
from scipy import stats

hyperparameter_strings = ['2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels',
                          '2020_05_28_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
                          '2020_05_28_date_smoothed_moving_window_21_days_US_counties_region_statsmodels']

for hyperparameter_str in hyperparameter_strings:
    params_filename = os.path.join('state_plots', hyperparameter_str, 'simplified_state_report.csv')
    print(f'Reading {params_filename}...')
    params = pd.read_csv(params_filename, encoding="ISO-8859-1")

    for col_std_err in params.columns:
        if 'mean_std_err' in col_std_err:
            col_std_err2 = col_std_err.replace('mean_std_err', 'std_err')
            params[col_std_err2] = params[col_std_err]
            del (params[col_std_err])

    map_col_std_err_to_p_value = dict()
    for col_std_err in params.columns:
        if 'std_err' in col_std_err:
            col_mean = col_std_err.replace('std_err', 'mean')
            for i in range(len(params)):
                col_distro = stats.norm(loc=params.iloc[i][col_mean], scale=params.iloc[i][col_std_err])
                col_p_value = col_distro.cdf(0)
                if col_p_value > 0.5:
                    col_p_value = 1 - col_p_value
                if col_std_err not in map_col_std_err_to_p_value:
                    map_col_std_err_to_p_value[col_std_err] = list()
                map_col_std_err_to_p_value[col_std_err].append(col_p_value)

    for col_std_err in map_col_std_err_to_p_value:
        col_p_value = col_std_err.replace('std_err', 'p_value')
        params[col_p_value] = map_col_std_err_to_p_value[col_std_err]

    param_name = 'positive_slope'
    param_ind = [i for i, x in enumerate(params['param']) if x == param_name]
    col_name = 'statsmodels_acc_p_value'
    col_name_mean = 'statsmodels_acc_mean'
    ilocs_ranked_by_p_val = sorted(param_ind,
                                   key=lambda x: params.iloc[x][col_name] * np.sign(params.iloc[x][col_name_mean]))
    ilocs_ranked_by_p_val = [i for i in ilocs_ranked_by_p_val if
                             params.iloc[i][col_name_mean] > 0 and params.iloc[i][col_name] < 0.1]

    params['statsmodels_p_value']

    sorted(params.columns)
    cols_to_show = ['state', 'statsmodels_mean', 'statsmodels_p_value', 'statsmodels_acc_mean',
                    'statsmodels_acc_p_value', ]
    print_params = params.iloc[ilocs_ranked_by_p_val][cols_to_show]
    print_params.index = list(range(len(print_params)))
    print_params.index += 1

    if 'US_counties' in hyperparameter_str:
        url = 'https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/'
    elif 'US_states' in hyperparameter_str:
        url = 'https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/'
    elif 'countries' in hyperparameter_str:
        url = 'https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/'

    print_params['state_github'] = [f'[{x}](' + url + x.lower().replace(" ", "_").replace(":", "") + '/index.html)'
                                    for x in print_params['state']]
    cols_to_show.remove('state')
    cols_to_show = ['state_github'] + cols_to_show
    print_params = print_params[cols_to_show]
    print(print_params.to_csv(sep='|', float_format='%.4g'))
