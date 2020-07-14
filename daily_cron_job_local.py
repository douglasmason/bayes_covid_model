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
import datetime
import covid_moving_window as covid
from sub_units import post_analysis

# get today's date -- note this is TWO days behind!
yesterdays_date_str = (datetime.date.today() - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
date_str = yesterdays_date_str
print(f'Yesterday: {yesterdays_date_str}')

####
# Step 2c: post-process parameters to identify which regions to deep-dive into
####

# Copy from AWS to local directory...
# cd /Users/kayote/PycharmProjects/im_so_pissed/state_plots
# HYP_STR=2020_07_05_date_smoothed_moving_window_21_days_US_counties_region_statsmodels; mkdir $HYP_STR; aws s3 cp --recursive s3://covid-figures/$HYP_STR/ $HYP_STR
# HYP_STR=2020_07_05_date_smoothed_moving_window_21_days_US_states_region_statsmodels; mkdir $HYP_STR; aws s3 cp --recursive s3://covid-figures/$HYP_STR/ $HYP_STR
# HYP_STR=2020_07_05_date_smoothed_moving_window_21_days_countries_region_statsmodels; mkdir $HYP_STR; aws s3 cp --recursive s3://covid-figures/$HYP_STR/ $HYP_STR
# HYP_STR=2020_07_05_date_smoothed_moving_window_21_days_provinces_region_statsmodels; mkdir $HYP_STR; aws s3 cp --recursive s3://covid-figures/$HYP_STR/ $HYP_STR

region_plot_subfolders = {
    Region.US_counties: '2020_07_12_date_smoothed_moving_window_21_days_US_counties_region_statsmodels',
    Region.countries:   '2020_07_12_date_smoothed_moving_window_21_days_countries_region_statsmodels',
    Region.US_states:   '2020_07_12_date_smoothed_moving_window_21_days_US_states_region_statsmodels'}

post_analysis.hyperparameter_strings = list(region_plot_subfolders.values())

######
# Step 3: Generate Figure Browser
######

import generate_plot_browser_moving_window_statsmodels_only as generate_figure_browser

opt_file_check = False  # change to False after run on server

for region, plot_subfolder in region_plot_subfolders.items():

    hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder))
    print(hyperparamater_str)

    data_dir = os.path.join('state_plots', plot_subfolder)

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
    # generate_figure_browser.hyperparameter_str = hyperparamater_str.replace(date_str, 'current') + '/'
    generate_figure_browser.hyperparameter_str = hyperparamater_str + '/'
    generate_figure_browser.plot_browser_dir = f'plot_browser_moving_window_statsmodels_only_{region}'
    generate_figure_browser.regions_to_present = regions_to_present
    generate_figure_browser.generate_plot_browser(regions_to_present)

#####
# Step 4: Update static_figures
#####

import os
import shutil

date_str = yesterdays_date_str.replace('-', '_')

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
# Step 5: Update Choropleths
#         re-do post-analysis and write to github table file, even if you downloaded results from the server
######

from sub_units import post_analysis

date_str = yesterdays_date_str

post_analysis.hyperparameter_strings = [os.path.basename(os.path.normpath(x)).format(date_str=date_str) for x in
                                        region_plot_subfolders.values()]
post_analysis.post_process_state_reports(opt_acc=False, opt_reset_tables_file=True)
post_analysis.post_process_state_reports(opt_acc=True)
print_and_write(github_readme_components.get_all(github_table_filename=post_analysis.github_table_filename),
                filename='github_README.txt', reset=True)
shutil.copyfile('github_README.txt', 'README.md')

post_analysis.countries_hp_str = [x for x in post_analysis.hyperparameter_strings if 'countries' in x][0]
post_analysis.counties_hp_str = [x for x in post_analysis.hyperparameter_strings if 'US_counties' in x][0]
post_analysis.states_hp_str = [x for x in post_analysis.hyperparameter_strings if 'US_states' in x][0]

post_analysis.choropleth_test()

######
# Step 6: Push to Github
######

# Make sure all the folders are in the github repo
# git add source_data/csse_covid_19_daily_reports/*
# git add plot_browser_moving_window_statsmodels_only_countries/*
# git add plot_browser_moving_window_statsmodels_only_US_counties/*
# git add plot_browser_moving_window_statsmodels_only_US_states/* 
# for f in plot_browser_moving_window_statsmodels_only_US_counties/*; do echo git add $f; done

# Do this by hand
# TODO: Figure out how to update git repo automatically instead of by hand
print('Now add, commit, and push to Github!')
