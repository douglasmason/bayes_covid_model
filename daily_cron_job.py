import os
import shutil 
import requests
import glob
import logging
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import datetime
from sub_units.utils import Region
from sub_units import post_analysis
from sub_units import github_readme_components
from sub_units.utils import print_and_write

# TODO: re-implement sucking data from the internet by checking for all days
#   and sucking only what it needs and put that in the load_data module
#   so it automatically happens whenever you load the data, rather
#   than having to manually do it here.

#####
# Step 1a: Update counts data
#####

url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
r = requests.get(url, allow_redirects=True)
with open('source_data/states.csv', 'w') as f:
    f.write(r.content.decode("utf-8"))

url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
r = requests.get(url, allow_redirects=True)
with open('source_data/counties.csv', 'w') as f:
    f.write(r.content.decode("utf-8"))

print('Downloading last week of data')
for days_back in tqdm(range(0, 7)):
    date = datetime.date.today() - datetime.timedelta(days=days_back)
    date_str = date.strftime('%m-%d-%Y')
    url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{date_str}.csv"
    r = requests.get(url, allow_redirects=True)
    filename = f'source_data/csse_covid_19_daily_reports/{date_str}.csv'
    print(filename, len(r.content.decode("utf-8")))
    with open(filename, 'w') as f:
        f.write(r.content.decode("utf-8"))

#####
# Step 1b: Update load_data (this happens as soon as you import modules that use load_data)
#####

import datetime
import covid_moving_window as covid

#####
# Step 2: Run Update
#####

# get today's date
yesterdays_date_str = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
print(f'Yesterday: {yesterdays_date_str}')

covid.n_bootstraps = None
covid.n_likelihood_samples = None
covid.moving_window_size = 21  # three weeks
covid.override_max_date_str = yesterdays_date_str
covid.opt_force_calc = False
covid.opt_force_plot = False
covid.opt_simplified = True  # set to True to just do statsmodels as a simplified daily service

####
# Step 2a: do deep-dive run on top regions by current case count
####

covid.opt_plot = True
covid.opt_truncate = True
covid.opt_report = True
covid.override_region = None
covid.override_run_states = None

_ = covid.run_everything()

####
# Step 2b: do run on all regions, but only produce parameters...
####

covid.opt_plot = False
covid.opt_truncate = False
covid.opt_report = True
covid.override_region = None
covid.override_run_states = None

region_plot_subfolders = covid.run_everything()

print('region_plot_subfolders:')
print(region_plot_subfolders)

####
# Step 2c: post-process parameters to identify which regions to deep-dive into
####

post_analysis.hyperparameter_strings = list(region_plot_subfolders.values())

from sub_units import post_analysis

post_analysis.post_process_state_reports(opt_acc=True)

# re-run and produce all the plots for the top outbreak-y counties
covid.opt_plot = True
covid.opt_report = False
covid.override_region = Region.US_counties
covid.override_run_states = list()

with open(post_analysis.scratchpad_filename, 'r') as f:
    for line in f.readlines():
        covid.override_run_states.append(line.strip())

region_plot_subfolders = covid.run_everything()

####
# Step 2d: post-process parameters to identify which regions to deep-dive into
####

from sub_units import post_analysis

post_analysis.post_process_state_reports(opt_acc=False)

# re-run and produce all the plots for the top outbreak-y counties
covid.opt_plot = True
covid.opt_report = False
covid.override_region = Region.US_counties
covid.override_run_states = list()

with open(post_analysis.scratchpad_filename, 'r') as f:
    for line in f.readlines():
        covid.override_run_states.append(line.strip())

region_plot_subfolders = covid.run_everything()

# TODO: Make update to README automatic

#####
# Step 2e: Re-do post-analysis and write to github table file
#####

from sub_units import post_analysis
from sub_units import github_readme_components
from sub_units.utils import print_and_write
import shutil 

post_analysis.post_process_state_reports(opt_acc=False, opt_reset_tables_file=True)
post_analysis.post_process_state_reports(opt_acc=True)
print_and_write(github_readme_components.get_all(github_table_filename=post_analysis.github_table_filename),
                filename='github_README.txt', reset=True)
shutil.copyfile('github_README.txt', 'README.md')


# TODO: Copy 'github_README.txt' to 'README.md'

#####
# Step 3: Upload Figures to AWS
#####

# {<Region.US_counties: 'US_counties'>: 'state_plots/2020_06_02_date_smoothed_moving_window_21_days_US_counties_region_statsmodels', <Region.countries: 'countries'>: 'state_plots/2020_06_02_date_smoothed_moving_window_21_days_countries_region_statsmodels', <Region.US_states: 'US_states'>: 'state_plots/2020_06_02_date_smoothed_moving_window_21_days_US_states_region_statsmodels'}

# run in bash...
# TODO: get these bash scripts to run from Python
# HYP_STR=2020_06_02_date_smoothed_moving_window_21_days_countries_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/$HYP_STR/
# HYP_STR=2020_06_02_date_smoothed_moving_window_21_days_US_states_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/$HYP_STR/
# HYP_STR=2020_06_02_date_smoothed_moving_window_21_days_US_counties_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/$HYP_STR/
# # HYP_STR=2020_06_02_date_smoothed_moving_window_21_days_provinces_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/$HYP_STR/


######
# Step 3: Generate Figure Browser
######

import os
from sub_units.utils import Region
import generate_plot_browser_moving_window_statsmodels_only as generate_figure_browser

opt_file_check = False  # change to False after run on server

region_plot_subfolders = {
    # Region.provinces: 'state_plots/2020_05_02_date_smoothed_moving_window_21_days_provinces_region_statsmodels',
    Region.countries: 'state_plots/2020_06_02_date_smoothed_moving_window_21_days_countries_region_statsmodels',
    Region.US_states: 'state_plots/2020_06_02_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
    Region.US_counties: 'state_plots/2020_06_02_date_smoothed_moving_window_21_days_US_counties_region_statsmodels'
}

for region, plot_subfolder in region_plot_subfolders.items():
    hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder))
    print(hyperparamater_str)

    data_dir = plot_subfolder

    # only add regions with a valid directory and a figure to present inside
    if opt_file_check:
        regions_to_present = [f for f in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, f)) and \
                              os.path.exists(os.path.join(data_dir, f, 'statsmodels_growth_rate_time_series.png')) and \
                              os.path.exists(
                                  os.path.join(data_dir, f, 'statsmodels_solutions_filled_quantiles.png')) and \
                              os.path.exists(
                                  os.path.join(data_dir, f, 'statsmodels_solutions_cumulative_filled_quantiles.png'))]
    else:
        regions_to_present = [f for f in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, f))]

    print(sorted(regions_to_present))

    # Regenerate Figures
    generate_figure_browser.hyperparameter_str = hyperparamater_str + '/'
    generate_figure_browser.plot_browser_dir = f'plot_browser_moving_window_statsmodels_only_{region}'
    generate_figure_browser.regions_to_present = regions_to_present
    generate_figure_browser.generate_plot_browser(regions_to_present)

#####
# Step 4: Update static_figures
#####

import os
import shutil 
date_str = yesterdays_date_str

map_filename_to_github = {
    os.path.join('state_plots',
                 f'{date_str}_date_smoothed_moving_window_21_days_countries_region_statsmodels',
                 'us',
                 'statsmodels_solutions_filled_quantiles.png'
                 ):
        'statsmodels_solutions_filled_quantiles.png',
    os.path.join('state_plots',
                 f'{date_str}_date_smoothed_moving_window_21_days_countries_region_statsmodels',
                 'us',
                 'statsmodels_solutions_cumulative_filled_quantiles.png'
                 ):
        'statsmodels_solutions_cumulative_filled_quantiles.png',
    os.path.join('state_plots',
                 f'{date_str}_date_smoothed_moving_window_21_days_countries_region_statsmodels',
                 'us',
                 'statsmodels_growth_rate_time_series.png'
                 ):
        'statsmodels_growth_rate_time_series.png',
    os.path.join('state_plots',
                 f'{date_str}_date_smoothed_moving_window_21_days_countries_region_statsmodels',
                 'simplified_boxplot_for_positive_slope_statsmodels.png'
                 ):
        'intl_simplified_boxplot_for_positive_slope_statsmodels.png',
    os.path.join('state_plots',
                 f'{date_str}_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
                 'simplified_boxplot_for_positive_slope_statsmodels.png'
                 ):
        'simplified_boxplot_for_positive_slope_statsmodels.png',
}

for in_file, out_file in map_filename_to_github.items():
    out_file = os.path.join('static_figures', out_file)
    print(f'Copying {in_file} to {out_file}...')
    dest_filename = shutil.copyfile(in_file, out_file)

######
# Step 5: Push to Github
######

# Make sure all the folders are in the github repo
# git add plot_browser_moving_window_statsmodels_only_countries/*
# git add plot_browser_moving_window_statsmodels_only_US_counties/*
# git add plot_browser_moving_window_statsmodels_only_US_states/* 


# Do this by hand
# TODO: Figure out how to update git repo automatically instead of by hand
print('Now add, commit, and push to Github!')
