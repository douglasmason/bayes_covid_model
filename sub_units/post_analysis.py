import os
import pandas as pd
import numpy as np
from scipy import stats
import sub_units.load_data_country as load_data  # only want to load this once, so import as singleton pattern

scratchpad_filename = 'states_to_draw_figures_for.list'

hyperparameter_strings = [
    # '2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels',
    # '2020_05_28_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
    '2020_06_02_date_smoothed_moving_window_21_days_US_counties_region_statsmodels'
]

def post_process_state_reports(opt_acc=True):
    for hyperparameter_str in hyperparameter_strings:
        # hyperparameter_str = hyperparameter_strings[0]
        print(f'Processing {hyperparameter_str}...')
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

        params['new_positive_cnt_7_day_avg'] = [0 if state not in load_data.map_state_to_series else np.mean(load_data.map_state_to_series[state]["cases_diff"][-7:]) for state in params['state']]
        params['new_deceased_cnt_7_day_avg'] = [0 if state not in load_data.map_state_to_series else np.mean(load_data.map_state_to_series[state]["deaths_diff"][-7:]) for state in params['state']]
        
        param_name = 'positive_slope'
        param_ind = [i for i, x in enumerate(params['param']) if x == param_name]
        
        if opt_acc:
            col_name = 'statsmodels_acc_p_value'
            col_name_mean = 'statsmodels_acc_mean'
        else:
            col_name = 'statsmodels_p_value'
            col_name_mean = 'statsmodels_mean'
        
        ilocs_ranked_by_p_val = sorted(param_ind,
                                       key=lambda x: params.iloc[x][col_name] * np.sign(params.iloc[x][col_name_mean]))
        
        if 'counties' in hyperparameter_str:
            ilocs_ranked_by_p_val = [i for i in ilocs_ranked_by_p_val if
                                     params.iloc[i][col_name_mean] > 0 and \
                                     params.iloc[i][col_name] < 0.1 and \
                                     params.iloc[i]['new_positive_cnt_7_day_avg'] > 5]
        else:
            ilocs_ranked_by_p_val = [i for i in ilocs_ranked_by_p_val if
                                     params.iloc[i][col_name_mean] > 0 and \
                                     params.iloc[i][col_name] < 0.1]
        
        cols_to_show = ['state', 'statsmodels_mean_converted', 'statsmodels_acc_mean_converted',
                        'statsmodels_acc_p_value']
        params['statsmodels_mean_converted'] = [f'{(np.exp(x) - 1) * 100:.4g}%' for x in params['statsmodels_mean']]
        params['statsmodels_acc_mean_converted'] = [f'{(np.exp(x) - 1) * 100:.4g}%' for x in params['statsmodels_acc_mean']]
        print_params = params.iloc[ilocs_ranked_by_p_val]
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
        print_params['pretty_print_new_positive_cnt_7_day_avg'] = [f'{x:.1f}' for x in print_params['new_positive_cnt_7_day_avg']]
        print_params['pretty_print_new_deceased_cnt_7_day_avg'] = [f'{x:.1f}' for x in print_params['new_deceased_cnt_7_day_avg']]
        cols_to_show = ['state_github', 'pretty_print_new_positive_cnt_7_day_avg'] + cols_to_show
        print(print_params[cols_to_show].to_csv(sep='|', float_format='%.4g'))
        
        if 'counties' in hyperparameter_str:
            with open(scratchpad_filename, 'w') as f:
                for row_ind, row_dict in print_params.iterrows():
                    f.write(row_dict['state'] + '\n')
                    
            

######
# Counties Map
######

def choropleth_test():
    print('Choropleth test!')
    import plotly.io as pio
    
    pio.renderers
    pio.renderers.default = "browser"
    
    import plotly.figure_factory as ff
    
    fips = ['06021', '06023', '06027',
            '06029', '06033', '06059',
            '06047', '06049', '06051',
            '06055', '06061']
    values = range(len(fips))
    
    fig = ff.create_choropleth(fips=fips, values=values)
    fig.layout.template = None
    fig.show()
    
    fig.close()
    
    param_name = 'positive_slope'
    param_ind = [i for i, x in enumerate(params['param']) if x == param_name]
    col_name = 'statsmodels_acc_p_value'
    col_name_mean = 'statsmodels_acc_mean'
    
    # clean up fips
    fips = list()
    for x in params.iloc[param_ind]['state']:
        val = None
        if x in load_data.map_state_to_fips:
            if np.isfinite(load_data.map_state_to_fips[x]):
                val = str(int(load_data.map_state_to_fips[x]))
        fips.append(val)
    values = params.iloc[param_ind][col_name]
    good_ind = [i for i, x in enumerate(fips) if type(x) == str]
    
    fig = ff.create_choropleth(fips=np.array(fips)[good_ind], values=np.array(values)[good_ind])
    fig.layout.template = None
    fig.show()
