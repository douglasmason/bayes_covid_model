import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy import stats
import datetime


#####
# Load in file
#####

def pearsonr_ci(x, y, alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1 / np.sqrt(x.size - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


combined = pd.read_csv('test_combined.csv')
combined['fips'] = [str(int(x)) if np.isfinite(x) else '0' for x in combined['fips']]
combined['fips'] = [x if len(x) > 4 else '0' + x for x in combined['fips']]
combined['fips']

import sub_units.load_data_country as load_data
from sub_units import post_analysis

date = datetime.datetime.strptime('2020_06_09', '%Y_%m_%d')
date_str = datetime.datetime.strftime(date, '%Y_%m_%d')

post_analysis.hyperparameter_strings = [
    f'{date_str}_date_smoothed_moving_window_21_days_countries_region_statsmodels',
    f'{date_str}_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
    f'{date_str}_date_smoothed_moving_window_21_days_US_counties_region_statsmodels'
]

post_analysis.post_process_state_reports(opt_acc=False, opt_reset_tables_file=True)
post_analysis.post_process_state_reports(opt_acc=True)

hyperparameter_str = post_analysis.hyperparameter_strings[2]
params_df = post_analysis.map_hp_str_to_params_df[hyperparameter_str]
params_df['fips'] = [load_data.map_state_to_fips.get(x, None) for x in params_df['state']]
params_df['fips'] = [str(int(x)) if np.isfinite(x) else '0' for x in params_df['fips']]
params_df['fips'] = [x if len(x) > 4 else '0' + x for x in params_df['fips']]
params_df['fips']

final_merge = pd.merge(combined, params_df, on=['fips'])

final_merge.to_csv('final_merge.csv')

import statsmodels.formula.api as smf

param_name = 'positive_slope'

df = pd.read_csv('final_merge.csv')
param_ind = [i for i, x in enumerate(df['param']) if x == param_name]
df = df.iloc[param_ind]

######
# Step ??: Get all offset days so I can do time series on regression
######

# hyperparameter_str = '2020_06_09_date_smoothed_moving_window_21_days_US_counties_region_statsmodels'
# param_name = 'positive_slope'

print('Rendering Regression Results...')
map_offset_to_df = dict()
map_offset_to_results = dict()
map_offset_to_correlation_dict = dict()
for offset in tqdm(range(0, 3 * 28, 7)):
    filename = os.path.join('state_plots', hyperparameter_str,
                            f'simplified_state_report_offset_{offset:03d}_of_084.csv')
    # print(f'Reading in {filename}...')
    params_df = pd.read_csv(filename, encoding="ISO-8859-1")
    # print('...done!')

    params_df['fips'] = [load_data.map_state_to_fips.get(x, None) for x in params_df['state']]
    params_df['fips'] = [str(int(x)) if np.isfinite(x) else '0' for x in params_df['fips']]
    params_df['fips'] = [x if len(x) > 4 else '0' + x for x in params_df['fips']]

    param_ind = [i for i, x in enumerate(params_df['param']) if x == param_name]
    params_df = params_df.iloc[param_ind]
    for col_std_err in params_df.columns:
        if 'mean_std_err' in col_std_err:
            col_std_err2 = col_std_err.replace('mean_std_err', 'std_err')
            params_df[col_std_err2] = params_df[col_std_err]
            del (params_df[col_std_err])

    map_col_std_err_to_p_value = dict()
    for col_std_err in params_df.columns:
        if 'std_err' in col_std_err:
            col_mean = col_std_err.replace('std_err', 'mean')
            for i in range(len(params_df)):
                col_distro = stats.norm(loc=params_df.iloc[i][col_mean], scale=params_df.iloc[i][col_std_err])
                col_p_value = col_distro.cdf(0)
                if col_p_value > 0.5:
                    col_p_value = 1 - col_p_value
                if col_std_err not in map_col_std_err_to_p_value:
                    map_col_std_err_to_p_value[col_std_err] = list()
                map_col_std_err_to_p_value[col_std_err].append(col_p_value)

    for col_std_err in map_col_std_err_to_p_value:
        col_p_value = col_std_err.replace('std_err', 'p_value')
        params_df[col_p_value] = map_col_std_err_to_p_value[col_std_err]

    params_df['new_positive_cnt_7_day_avg'] = [0 if state not in load_data.map_state_to_series else np.mean(
        load_data.map_state_to_series[state]["cases_diff"][-7:]) for state in params_df['state']]
    params_df['new_deceased_cnt_7_day_avg'] = [0 if state not in load_data.map_state_to_series else np.mean(
        load_data.map_state_to_series[state]["deaths_diff"][-7:]) for state in params_df['state']]

    params_df['statsmodels_p_value'] = stats.norm.sf(
        abs(params_df['statsmodels_mean'] / params_df['statsmodels_std_err']))
    params_df['statsmodels_mean_converted'] = [f'{(np.exp(x) - 1) * 100:.4g}%' for x in params_df['statsmodels_mean']]
    # params_df['statsmodels_acc_mean_converted'] = [f'{(np.exp(x) - 1) * 100:.4g}%' for x in
    #                                             params_df['statsmodels_acc_mean']]

    df = pd.merge(combined, params_df, on=['fips'])

    filter_ind = list(range(len(df)))
    new_filter_ind = np.array([i for i, x in enumerate(df['statsmodels_p_value']) if x < 0.1])
    filter_ind = [i for i in filter_ind if i in new_filter_ind]

    new_filter_ind = np.array([i for i, x in enumerate(df['new_positive_cnt_7_day_avg']) if x > 3])
    filter_ind = [i for i in filter_ind if i in new_filter_ind]

    df['Mar_temp_minus_67'] = df['Mar_temp'] - 67

    model = smf.ols(
        formula='statsmodels_mean ~ Mar_temp_minus_67 + Mar_precip + frac_female + frac_black + frac_asian + frac_native + frac_hispanic + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)',
        data=df.iloc[filter_ind])
    results = model.fit()
    print(f'\n#######\n# Offset: {offset}\n#######')
    print(results.summary())

    map_offset_to_df[offset] = df
    map_offset_to_results[offset] = results

    df['log_pop_density'] = np.log(df['pop_density'])
    df['log_median_household_income'] = np.log(df['median_household_income'])

    corr_dict = dict()
    for plot_param_name in ['Mar_temp_minus_67', 'Mar_precip', 'frac_male', 'frac_female', 'frac_white', 'frac_black',
                            'frac_asian', 'frac_native', 'frac_hispanic', 'frac_under_25', 'frac_25_to_44',
                            'frac_45_to_64', 'frac_over_64', 'log_pop_density', 'unemployment_rate',
                            'log_median_household_income']:
        corr, pval, p5, p95 = pearsonr_ci(df.iloc[filter_ind][plot_param_name], df.iloc[filter_ind]['statsmodels_mean'],
                                          alpha=0.1)
        corr_dict[plot_param_name + '_corr'] = corr
        corr_dict[plot_param_name + '_pval'] = pval
        corr_dict[plot_param_name + '_p5'] = p5
        corr_dict[plot_param_name + '_p95'] = p95
        _, _, p25, p75 = pearsonr_ci(df.iloc[filter_ind][plot_param_name], df.iloc[filter_ind]['statsmodels_mean'],
                                     alpha=0.5)
        corr_dict[plot_param_name + '_p25'] = p25
        corr_dict[plot_param_name + '_p75'] = p75

    map_offset_to_correlation_dict[offset] = corr_dict.copy()

list_of_dicts = list()
for offset in list(reversed(sorted(map_offset_to_results))):
    offset_date_str = datetime.datetime.strftime(date - datetime.timedelta(days=offset), '%Y-%m-%d')
    print()
    print(offset_date_str)
    params = map_offset_to_results[offset].params.to_dict()
    bse = map_offset_to_results[offset].bse.to_dict()
    pvals = map_offset_to_results[offset].pvalues.to_dict()

    for tmp_param_name in pvals:
        if pvals[tmp_param_name] < 0.1:
            print(f'{tmp_param_name}: {params[tmp_param_name]:0.4g} (p-value {pvals[tmp_param_name]:0.4g}')

    tmp_dict = params.copy()

    for tmp_param_name in params:
        param_model = stats.norm(loc=params[tmp_param_name], scale=bse[tmp_param_name])
        tmp_dict[tmp_param_name + '_p5'] = param_model.ppf(0.05)
        tmp_dict[tmp_param_name + '_p25'] = param_model.ppf(0.25)
        tmp_dict[tmp_param_name + '_p50'] = param_model.ppf(0.50)
        tmp_dict[tmp_param_name + '_p75'] = param_model.ppf(0.75)
        tmp_dict[tmp_param_name + '_p95'] = param_model.ppf(0.95)

    tmp_dict.update({key+'_corr': val for key,val in map_offset_to_correlation_dict[offset].items()})

    tmp_dict.update({'date': date - datetime.timedelta(days=offset), 'offset': offset})
    list_of_dicts.append(tmp_dict)

plot_df = pd.DataFrame(list_of_dicts)

######
# Render plots
######

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import matplotlib.ticker as mtick

plt.style.use('seaborn-darkgrid')
matplotlib.use('Agg')

for plot_param_name in map_offset_to_results[list(map_offset_to_results.keys())[0]].params.to_dict():
    plt.close()
    fig, ax = plt.subplots()
    # plt.plot([(date-x).days for x in plot_df['date']], plot_df['np.log(pop_density)'])

    x_vals = plot_df['date']  # [(date - x).days for x in plot_df['date']]
    ax.fill_between(x_vals,
                    plot_df[f'{plot_param_name}_p5'],
                    plot_df[f'{plot_param_name}_p95'],
                    facecolor=matplotlib.colors.colorConverter.to_rgba('blue', alpha=0.3),
                    edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                    )
    ax.fill_between(x_vals,
                    plot_df[f'{plot_param_name}_p25'],
                    plot_df[f'{plot_param_name}_p75'],
                    facecolor=matplotlib.colors.colorConverter.to_rgba('blue', alpha=0.6),
                    edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                    )
    ax.plot(x_vals,
            plot_df[f'{plot_param_name}'],
            color='darkblue')

    fig.autofmt_xdate()
    # this removes the year from the x-axis ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.title(plot_param_name)
    plt.xlabel('Date')
    plt.ylabel('Coefficient')
    plt.axhline(linestyle='--', y=0, color='black')
    plt.savefig(os.path.join('parameter_regression_results', plot_param_name + '.png'))

for plot_param_name in sorted(['_'.join(x.split('_')[:-1]) for x in
                               map_offset_to_correlation_dict[list(map_offset_to_correlation_dict.keys())[0]].keys()]):
    plt.close()
    fig, ax = plt.subplots()
    # plt.plot([(date-x).days for x in plot_df['date']], plot_df['np.log(pop_density)'])

    x_vals = plot_df['date']  # [(date - x).days for x in plot_df['date']]
    ax.fill_between(x_vals,
                    plot_df[f'{plot_param_name}_p5_corr'],
                    plot_df[f'{plot_param_name}_p95_corr'],
                    facecolor=matplotlib.colors.colorConverter.to_rgba('blue', alpha=0.3),
                    edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                    )
    ax.fill_between(x_vals,
                    plot_df[f'{plot_param_name}_p25_corr'],
                    plot_df[f'{plot_param_name}_p75_corr'],
                    facecolor=matplotlib.colors.colorConverter.to_rgba('blue', alpha=0.6),
                    edgecolor=(0, 0, 0, 0)  # get rid of the darker edge
                    )
    ax.plot(x_vals,
            plot_df[f'{plot_param_name}_corr_corr'],
            color='darkblue')

    fig.autofmt_xdate()
    # this removes the year from the x-axis ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.title(plot_param_name)
    plt.xlabel('Date')
    plt.ylabel('Coefficient')
    plt.axhline(linestyle='--', y=0, color='black')
    plt.savefig(os.path.join('parameter_correlation_results', plot_param_name + '.png'))
    
######


plt.plot(df['statsmodels_mean'], df['frac_black'], '.')
plt.show()

filter_ind = list(range(len(df)))
new_filter_ind = np.array([i for i, x in enumerate(df['statsmodels_p_value']) if x < 0.1])
filter_ind = [i for i in filter_ind if i in new_filter_ind]

new_filter_ind = np.array([i for i, x in enumerate(df['new_positive_cnt_7_day_avg']) if x > 3])
filter_ind = [i for i in filter_ind if i in new_filter_ind]

for var in ['frac_white', 'frac_black', 'frac_asian', 'frac_native', 'frac_hispanic', 'frac_under_25', 'frac_25_to_44',
            'frac_45_to_64', 'frac_over_64', 'pop_density', 'unemployment_rate', 'median_household_income']:
    model = smf.ols(formula=f'statsmodels_mean ~ {var}',
                    data=df.iloc[filter_ind])
    results = model.fit()
    print(f'\n--- {var} ---')
    print(results.summary())

model = smf.ols(
    formula='statsmodels_mean ~ frac_male + frac_female + frac_white + frac_black + frac_asian + frac_native + frac_hispanic + frac_under_25 + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)',
    data=df.iloc[filter_ind])
results = model.fit()
results.summary()

model = smf.ols(
    formula='statsmodels_mean ~ frac_female + frac_black + frac_asian + frac_native + frac_hispanic + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)',
    data=df.iloc[filter_ind])
results = model.fit()
results.summary()

filter_ind = list(range(len(df)))
new_filter_ind = np.array([i for i, x in enumerate(df['statsmodels_acc_p_value']) if x < 0.1])
filter_ind = [i for i in filter_ind if i in new_filter_ind]

new_filter_ind = np.array([i for i, x in enumerate(df['new_positive_cnt_7_day_avg']) if x > 3])
filter_ind = [i for i in filter_ind if i in new_filter_ind]

model = smf.ols(
    formula='statsmodels_acc_mean ~ frac_female + frac_black + frac_asian + frac_native + frac_hispanic + frac_under_25 + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)',
    data=df.iloc[filter_ind])
results = model.fit()
results.summary()
