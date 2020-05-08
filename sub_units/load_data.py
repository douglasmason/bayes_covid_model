import pandas as pd
import numpy as np
import os
import datetime
data_dir = 'source_data'

# NB: full_count_data is cumulative
full_count_data = pd.read_csv(os.path.join(data_dir,'counts.csv'))  # from https://github.com/nytimes/covid-19-data
full_count_data['date'] = full_count_data['date'].astype('datetime64[ns]')

date_range = pd.date_range(min(full_count_data['date']), max(full_count_data['date']))
map_state_to_series = dict()

# get totals across U.S.
list_of_dict_totals = list()
for date in sorted(set(full_count_data['date'])):
    date_iloc = [i for i, x in enumerate(full_count_data['date']) if x == date]
    sum_cases = sum(full_count_data.iloc[date_iloc]['cases'])
    sum_deaths = sum(full_count_data.iloc[date_iloc]['deaths'])
    list_of_dict_totals.append({'date': date, 'cases': sum_cases, 'deaths': sum_deaths, 'state': 'total'})

total_counts_data = pd.DataFrame(list_of_dict_totals)
full_count_data = full_count_data.append(total_counts_data, ignore_index=True)

# data munging gets daily-differences differences by state
for state in sorted(set(full_count_data['state'])):
    state_iloc = [i for i, x in enumerate(full_count_data['state']) if x == state]
    state_iloc = sorted(state_iloc, key=lambda x: full_count_data.iloc[x]['date'])

    cases_series = pd.Series({full_count_data.iloc[i]['date']: full_count_data.iloc[i]['cases'] for i in state_iloc})
    deaths_series = pd.Series({full_count_data.iloc[i]['date']: full_count_data.iloc[i]['deaths'] for i in state_iloc})

    cases_series.index = pd.DatetimeIndex(cases_series.index)
    deaths_series.index = pd.DatetimeIndex(deaths_series.index)

    cases_diff = cases_series.diff()
    deaths_diff = deaths_series.diff()

    map_state_to_series[state] = {'cases_series': cases_series,
                                  'deaths_series': deaths_series,
                                  'cases_diff': cases_diff,
                                  'deaths_diff': deaths_diff}

# get state populations from https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-detail.html
state_populations = pd.read_csv(os.path.join(data_dir,'state_population.csv'))
map_state_to_population = {
    state_populations.iloc[i]['state']: int(state_populations.iloc[i]['population'].replace(',', '')) for i in
    range(len(state_populations))}
map_state_to_population['total'] = sum(map_state_to_population.values())

# get shelter-in-place dates
SIP_dates = pd.read_csv(
    os.path.join(data_dir,'shelter_in_place_dates_by_state.csv'))  # from https://www.finra.org/rules-guidance/key-topics/covid-19/shelter-in-place
SIP_dates['sip_date'] = SIP_dates['sip_date'].astype('datetime64[ns]')
SIP_dates = SIP_dates.append(pd.DataFrame([{'state': 'total', 'sip_date': datetime.datetime.strptime('2020-03-20', '%Y-%m-%d')}]))

for i in range(len(SIP_dates)):
    state = SIP_dates.iloc[i]['state']
    SIP_date = SIP_dates.iloc[i]['sip_date']
    map_state_to_series[state]['sip_date'] = SIP_date

def get_state_data(state):
    
    # tmp = datetime.datetime.strptime('2020-01-21', '%Y-%m-%d')
    # tmp2 = datetime.datetime.strptime('2020-03-19', '%Y-%m-%d')
    # tmp2 - tmp = 58 days
    # NP: Cali shelter-in-place (SIP) March 19 Data starts at Jan. 21.
    
    population = map_state_to_population[state]
    count_data = map_state_to_series[state]['cases_series'].values
    n_count_data = np.prod(count_data.shape)
    print(f'# data points: {n_count_data}')
    
    min_date = min(list(map_state_to_series[state]['cases_series'].index))
    
    # format count_data into I and S values for SIR Model
    infected = [x for x in count_data]
    susceptible = [population - x for x in count_data]
    dead = [x for x in map_state_to_series[state]['deaths_series'].values]
    series_data = np.vstack([susceptible, infected, dead]).T
        
    if 'sip_date' in map_state_to_series:
        sip_date = map_state_to_series[state]['sip_date']
    else:
        sip_date = None
        
    return {'series_data': series_data,
            'population': population,
            'sip_date': sip_date,
            'min_date': min_date}
