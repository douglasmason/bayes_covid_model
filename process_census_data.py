import os
import pandas as pd
import numpy as np
from collections import Counter

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
pivot_age_groups

pivot_age_groups

for i in range(19):
    pivot_age_groups[f'frac_{i}'] = pivot_age_groups[i] / pivot_age_groups[0]

combined = pd.merge(all_age_groups, pivot_age_groups, on=['fips', 'STNAME', 'CTYNAME'])

for i in range(19):
    del combined[i]
    
combined[['fips', 'STNAME', 'CTYNAME', 'frac_1', 'frac_white']]
combined['fips'] = [str(int(x)) for x in combined['fips']]
combined['fips'] = [x if len(x) > 4 else '0' + x for x in combined['fips']]
combined.to_csv('test_combined.csv')

import sub_units.load_data_country as load_data
from sub_units import post_analysis

post_analysis.post_process_state_reports(opt_acc=False, opt_reset_tables_file=True)
post_analysis.post_process_state_reports(opt_acc=True)

hyperparameter_str = '2020_06_02_date_smoothed_moving_window_21_days_US_counties_region_statsmodels'
params = post_analysis.map_hp_str_to_params_df[hyperparameter_str]
params['fips'] = [load_data.map_state_to_fips.get(x, None) for x in params['state']]
params['fips'] = [str(int(x)) if np.isfinite(x) else '0' for x in params['fips']]
params['fips'] = [x if len(x) > 4 else '0' + x for x in params['fips']]
params['fips']

final_merge = pd.merge(combined, params, on=['fips'])

final_merge.to_csv('final_merge.csv')

#####
# Load in file
#####

import statsmodels.formula.api as smf

df = pd.read_csv('final_merge.csv')
param_ind = [i for i, x in enumerate(df['param']) if x=='positive_slope']
df = df.iloc[param_ind]

# from https://github.com/camillol/cs424p3/blob/master/data/Population-Density%20By%20County.csv
pop_density = pd.read_csv(os.path.join('source_data','Population-Density By County.csv'))
pop_density['fips'] = pop_density['GCT_STUB.target-geo-id2']
pop_density['pop_density'] = pop_density['Density per square mile of land area']
pop_density = pop_density[['fips', 'pop_density']]
df = pd.merge(df, pop_density, on=['fips'])

# from https://www.ers.usda.gov/data-products/county-level-data-sets/download-data/
income = pd.read_csv(os.path.join('source_data','Unemployment.csv'))
income['fips'] = income['FIPStxt']
income['median_household_income'] = [int(x.replace(',','')) if type(x)==str else 0 for x in income['Median_Household_Income_2018']]
income['unemployment_rate'] = income['Unemployment_rate_2019']
df = pd.merge(df, income, on=['fips'])



# from https://www.ncei.noaa.gov/news/noaa-offers-climate-data-counties
climate = pd.read_csv(os.path.join('source_data','climdiv-norm-tmpccy-v1.0.0-20200504'), delimiter='  ')


df['fips'] = [str(int(x)) if np.isfinite(x) else '0' for x in df['fips']]
df['fips'] = [x if len(x) > 4 else '0' + x for x in df['fips']]


df['frac_under_25'] = df['frac_1'] + df['frac_2'] + df['frac_3'] + df['frac_4'] + df['frac_5']
df['frac_25_to_44'] = df['frac_6'] + df['frac_7'] + df['frac_8'] + df['frac_9']
df['frac_45_to_64'] = df['frac_10'] + df['frac_11'] + df['frac_12'] + df['frac_13']
df['frac_over_64'] = df['frac_14'] + df['frac_15'] + df['frac_16'] + df['frac_17'] + df['frac_18']

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
#matplotlib.use('TkAgg')

plt.plot(df['statsmodels_mean'], df['frac_black'], '.')
plt.show()

filter_ind = list(range(len(df)))
new_filter_ind = np.array([i for i, x in enumerate(df['statsmodels_p_value']) if x < 0.1])
filter_ind = [i for i in filter_ind if i in new_filter_ind]

new_filter_ind = np.array([i for i, x in enumerate(df['new_positive_cnt_7_day_avg']) if x > 3])
filter_ind = [i for i in filter_ind if i in new_filter_ind]


for var in ['frac_white','frac_black','frac_asian','frac_native','frac_hispanic', 'frac_under_25', 'frac_25_to_44', 'frac_45_to_64', 'frac_over_64', 'pop_density', 'unemployment_rate', 'median_household_income']:
    model = smf.ols(formula=f'statsmodels_mean ~ {var}',
                    data=df.iloc[filter_ind])
    results = model.fit()
    print(f'\n--- {var} ---')
    print(results.summary())

model = smf.ols(formula='statsmodels_mean ~ frac_male + frac_female + frac_white + frac_black + frac_asian + frac_native + frac_hispanic + frac_under_25 + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)', 
                data=df.iloc[filter_ind])
results = model.fit()
results.summary()

model = smf.ols(formula='statsmodels_mean ~ frac_female + frac_black + frac_asian + frac_native + frac_hispanic + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)', 
                data=df.iloc[filter_ind])
results = model.fit()
results.summary()

filter_ind = list(range(len(df)))
new_filter_ind = np.array([i for i, x in enumerate(df['statsmodels_acc_p_value']) if x < 0.1])
filter_ind = [i for i in filter_ind if i in new_filter_ind]

new_filter_ind = np.array([i for i, x in enumerate(df['new_positive_cnt_7_day_avg']) if x > 3])
filter_ind = [i for i in filter_ind if i in new_filter_ind]

model = smf.ols(formula='statsmodels_acc_mean ~ frac_female + frac_black + frac_asian + frac_native + frac_hispanic + frac_under_25 + frac_25_to_44 + frac_45_to_64 + frac_over_64 + np.log(pop_density) + unemployment_rate + np.log(median_household_income)', 
                data=df.iloc[filter_ind])
results = model.fit()
results.summary()

