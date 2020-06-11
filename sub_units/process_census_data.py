import os
import pandas as pd

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
