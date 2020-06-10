import os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy import stats

data_source = os.path.expanduser('~/Downloads/cc-est2018-alldata.csv')
census = pd.read_csv(data_source, encoding="ISO-8859-1")

year_ind = [i for i, x in enumerate(census['YEAR']) if x == 11]
census = census.iloc[year_ind]

census['fips_county'] = [str(int(x)) for x in census['COUNTY']]
census['fips_county'] = [x if len(x) == 3 else '0' + x for x in census['fips_county']]
census['fips_county'] = [x if len(x) == 3 else '0' + x for x in census['fips_county']]
census['fips_state'] = [str(int(x)) for x in census['STATE']]
census['fips_state'] = [x if len(x) == 2 else '0' + x for x in census['fips_state']]
census['fips'] = [y + x for x, y in zip(census['fips_county'], census['fips_state'])]

census[['fips', 'STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'YEAR', 'AGEGRP']].iloc[:10]

all_age_groups = census.groupby(['fips', 'STNAME', 'CTYNAME', 'STATE', 'COUNTY'], as_index=False).sum()

census['frac_male'] = census['TOT_MALE'] / census['TOT_POP']
census['frac_female'] = census['TOT_FEMALE'] / census['TOT_POP']
census['frac_hispanic'] = (census['H_MALE'] + census['H_FEMALE']) / (census['TOT_MALE'] + census['TOT_FEMALE'])
census['frac_white'] = (census['WAC_MALE'] + census['WAC_FEMALE']) / (census['TOT_MALE'] + census['TOT_FEMALE'])
census['frac_black'] = (census['BAC_MALE'] + census['BAC_FEMALE']) / (census['TOT_MALE'] + census['TOT_FEMALE'])
census['frac_asian'] = (census['AAC_MALE'] + census['AAC_FEMALE']) / (census['TOT_MALE'] + census['TOT_FEMALE'])
census['frac_native'] = (census['NAC_MALE'] + census['NAC_FEMALE'] + census['IAC_MALE'] + census['IAC_FEMALE']) / (
        census['TOT_MALE'] + census['TOT_FEMALE'])

census[['fips', 'STNAME', 'CTYNAME', 'AGEGRP', 'frac_white', 'frac_hispanic', 'frac_asian', 'frac_black',
        'frac_native']].iloc[:10]

all_age_groups['frac_male'] = all_age_groups['TOT_MALE'] / all_age_groups['TOT_POP']
all_age_groups['frac_female'] = all_age_groups['TOT_FEMALE'] / all_age_groups['TOT_POP']
all_age_groups['frac_hispanic'] = (all_age_groups['H_MALE'] + all_age_groups['H_FEMALE']) / (
        all_age_groups['TOT_MALE'] + all_age_groups['TOT_FEMALE'])
all_age_groups['frac_white'] = (all_age_groups['WAC_MALE'] + all_age_groups['WAC_FEMALE']) / (
        all_age_groups['TOT_MALE'] + all_age_groups['TOT_FEMALE'])
all_age_groups['frac_black'] = (all_age_groups['BAC_MALE'] + all_age_groups['BAC_FEMALE']) / (
        all_age_groups['TOT_MALE'] + all_age_groups['TOT_FEMALE'])
all_age_groups['frac_asian'] = (all_age_groups['AAC_MALE'] + all_age_groups['AAC_FEMALE']) / (
        all_age_groups['TOT_MALE'] + all_age_groups['TOT_FEMALE'])
all_age_groups['frac_native'] = (all_age_groups['NAC_MALE'] + all_age_groups['NAC_FEMALE'] + all_age_groups[
    'IAC_MALE'] + all_age_groups['IAC_FEMALE']) / (all_age_groups['TOT_MALE'] + all_age_groups['TOT_FEMALE'])

all_age_groups[
    ['fips', 'STNAME', 'CTYNAME', 'frac_white', 'frac_hispanic', 'frac_asian', 'frac_black', 'frac_native']].iloc[:10]

pivot_age_groups = pd.pivot_table(census, index=['fips', 'STNAME', 'CTYNAME'], values=['TOT_POP'], columns=['AGEGRP'],
                                  margins=True)
pivot_age_groups.columns = pivot_age_groups.columns.get_level_values(1)
pivot_age_groups.reset_index(level=0, inplace=True)
pivot_age_groups.reset_index(level=0, inplace=True)
pivot_age_groups.reset_index(level=0, inplace=True)
pivot_age_groups.reset_index(level=0, inplace=True)

for i in range(19):
    pivot_age_groups[f'frac_{i}'] = pivot_age_groups[i] / pivot_age_groups[0]

combined = pd.merge(all_age_groups, pivot_age_groups, on=['fips', 'STNAME', 'CTYNAME'])

for i in range(19):
    del combined[i]

combined[['fips', 'STNAME', 'CTYNAME', 'frac_1', 'frac_white']]
combined['fips'] = [str(int(x)) for x in combined['fips']]
combined['fips'] = [x if len(x) > 4 else '0' + x for x in combined['fips']]

# from https://github.com/camillol/cs424p3/blob/master/data/Population-Density%20By%20County.csv
pop_density = pd.read_csv(os.path.join('source_data', 'Population-Density By County.csv'))
pop_density['fips'] = [f'{x:05d}' for x in pop_density['GCT_STUB.target-geo-id2']]
pop_density['pop_density'] = pop_density['Density per square mile of land area']
pop_density = pop_density[['fips', 'pop_density']]

combined = pd.merge(combined, pop_density, on=['fips'])

# from https://www.ers.usda.gov/data-products/county-level-data-sets/download-data/
income = pd.read_csv(os.path.join('source_data', 'Unemployment.csv'))
income['fips'] = [f'{x:05d}' for x in income['FIPStxt']]
income['median_household_income'] = [int(x.replace(',', '')) if type(x) == str else 0 for x in
                                     income['Median_Household_Income_2018']]
income['unemployment_rate'] = income['Unemployment_rate_2019']
combined = pd.merge(combined, income, on=['fips'])

# from https://www.ncei.noaa.gov/news/noaa-offers-climate-data-counties
climate = pd.read_csv(os.path.join('source_data', 'climdiv-norm-tmpccy-v1.0.0-20200504'), delimiter='\s+',
                      dtype={'state_code': str,
                             'Jan': float,
                             'Feb': float,
                             'Mar': float,
                             'Apr': float,
                             'May': float,
                             'Jun': float,
                             'Jul': float,
                             'Aug': float,
                             'Sep': float,
                             'Oct': float,
                             'Nov': float,
                             'Dec': float})
climate.columns = ['state_code_temp', 'Jan_temp', 'Feb_temp', 'Mar_temp', 'Apr_temp', 'May_temp', 'Jun_temp',
                   'Jul_temp', 'Aug_temp', 'Sep_temp', 'Oct_temp', 'Nov_temp', 'Dec_temp']
climate['fips'] = [f'{int(x[:5]):05d}' for x in climate['state_code_temp']]
climate['year_code_temp'] = [x[-4:] for x in climate['state_code_temp']]
year_code_ind = [i for i, x in enumerate(climate['year_code_temp']) if x == '0009']
climate = climate.iloc[year_code_ind]
combined = pd.merge(combined, climate, on='fips')

climate = pd.read_csv(os.path.join('source_data', 'climdiv-norm-pcpncy-v1.0.0-20200504'), delimiter='\s+',
                      dtype={'state_code': str,
                             'Jan': float,
                             'Feb': float,
                             'Mar': float,
                             'Apr': float,
                             'May': float,
                             'Jun': float,
                             'Jul': float,
                             'Aug': float,
                             'Sep': float,
                             'Oct': float,
                             'Nov': float,
                             'Dec': float})
climate.columns = ['state_code_precip', 'Jan_precip', 'Feb_precip', 'Mar_precip', 'Apr_precip', 'May_precip',
                   'Jun_precip', 'Jul_precip', 'Aug_precip', 'Sep_precip', 'Oct_precip', 'Nov_precip', 'Dec_precip']
climate['fips'] = [f'{int(x[:5]):05d}' for x in climate['state_code_precip']]
climate['year_code_precip'] = [x[-4:] for x in climate['state_code_precip']]
year_code_ind = [i for i, x in enumerate(climate['year_code_precip']) if x == '0009']
climate = climate.iloc[year_code_ind]
combined = pd.merge(combined, climate, on='fips')

combined['frac_under_25'] = combined['frac_1'] + combined['frac_2'] + combined['frac_3'] + combined['frac_4'] + \
                            combined['frac_5']
combined['frac_25_to_44'] = combined['frac_6'] + combined['frac_7'] + combined['frac_8'] + combined['frac_9']
combined['frac_45_to_64'] = combined['frac_10'] + combined['frac_11'] + combined['frac_12'] + combined['frac_13']
combined['frac_over_64'] = combined['frac_14'] + combined['frac_15'] + combined['frac_16'] + combined['frac_17'] + \
                           combined['frac_18']

combined.to_csv('test_combined.csv')

#####
# Load in file
#####

combined = pd.read_csv('test_combined.csv')

import sub_units.load_data_country as load_data
from sub_units import post_analysis

post_analysis.hyperparameter_strings = [
    '2020_06_07_date_smoothed_moving_window_21_days_countries_region_statsmodels',
    '2020_06_07_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
    '2020_06_07_date_smoothed_moving_window_21_days_US_counties_region_statsmodels'
]

post_analysis.post_process_state_reports(opt_acc=False, opt_reset_tables_file=True)
post_analysis.post_process_state_reports(opt_acc=True)

hyperparameter_str = post_analysis.hyperparameter_strings[2]
params = post_analysis.map_hp_str_to_params_df[hyperparameter_str]
params['fips'] = [load_data.map_state_to_fips.get(x, None) for x in params['state']]
params['fips'] = [str(int(x)) if np.isfinite(x) else '0' for x in params['fips']]
params['fips'] = [x if len(x) > 4 else '0' + x for x in params['fips']]
params['fips']

final_merge = pd.merge(combined, params, on=['fips'])

final_merge.to_csv('final_merge.csv')

import statsmodels.formula.api as smf

df = pd.read_csv('final_merge.csv')
param_ind = [i for i, x in enumerate(df['param']) if x == 'positive_slope']
df = df.iloc[param_ind]

######
# Step ??: Get all offset days so I can do time series on regression
######

hyperparameter_str = '2020_06_07_date_smoothed_moving_window_21_days_US_counties_region_statsmodels'
param_name = 'positive_slope'
map_offset_to_df = dict()
for offset in tqdm(range(0, 3 * 28, 7)):
    filename = os.path.join('state_plots', hyperparameter_str,
                            f'simplified_state_report_offset_{offset:03d}_of_084.csv')
    params = pd.read_csv(filename, encoding="ISO-8859-1")

    params['fips'] = [load_data.map_state_to_fips.get(x, None) for x in params['state']]
    params['fips'] = [str(int(x)) if np.isfinite(x) else '0' for x in params['fips']]
    params['fips'] = [x if len(x) > 4 else '0' + x for x in params['fips']]
    
    param_ind = [i for i, x in enumerate(params['param']) if x == param_name]
    params = params.iloc[param_ind]
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

    params['new_positive_cnt_7_day_avg'] = [0 if state not in load_data.map_state_to_series else np.mean(
        load_data.map_state_to_series[state]["cases_diff"][-7:]) for state in params['state']]
    params['new_deceased_cnt_7_day_avg'] = [0 if state not in load_data.map_state_to_series else np.mean(
        load_data.map_state_to_series[state]["deaths_diff"][-7:]) for state in params['state']]

    params['statsmodels_mean_converted'] = [f'{(np.exp(x) - 1) * 100:.4g}%' for x in params['statsmodels_mean']]
    params['statsmodels_acc_mean_converted'] = [f'{(np.exp(x) - 1) * 100:.4g}%' for x in
                                                params['statsmodels_acc_mean']]

    df = pd.merge(combined, params, on=['fips'])

    filter_ind = list(range(len(df)))
    new_filter_ind = np.array([i for i, x in enumerate(df['statsmodels_acc_p_value']) if x < 0.1])
    filter_ind = [i for i in filter_ind if i in new_filter_ind]

    new_filter_ind = np.array([i for i, x in enumerate(df['new_positive_cnt_7_day_avg']) if x > 3])
    filter_ind = [i for i in filter_ind if i in new_filter_ind]

    model = smf.ols(
        formula='statsmodels_mean ~ frac_male + frac_female + frac_white + frac_black + frac_asian + frac_native + frac_hispanic + frac_under_25 + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)',
        data=df.iloc[filter_ind])
    results = model.fit()
    print(f'\n#######\n# Offset: {offset}\n#######')
    print(results.summary())

import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')
# matplotlib.use('TkAgg')

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
