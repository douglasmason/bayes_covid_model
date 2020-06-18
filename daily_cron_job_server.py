import os
import requests
import glob
import logging
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import datetime
from sub_units.utils import Region
from sub_units import post_analysis


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


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
covid.opt_plot = True
covid.opt_truncate = False
covid.opt_report = True
covid.opt_simplified = True  # set to True to just do statsmodels as a simplified daily service
covid.override_run_states = None

region_plot_subfolders = covid.run_everything()

print('region_plot_subfolders:')
print(region_plot_subfolders)

#####
# Step 3: Upload Figures to AWS
#####

from sub_units.utils import Region

region_plot_subfolders = {
    Region.US_counties: 'state_plots/{date_str}_date_smoothed_moving_window_21_days_US_counties_region_statsmodels',
    Region.countries: 'state_plots/{date_str}_date_smoothed_moving_window_21_days_countries_region_statsmodels',
    Region.US_states: 'state_plots/{date_str}_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
    # Region.provinces: 'state_plots/{date_str}_date_smoothed_moving_window_21_days_provinces_region_statsmodels'
}

# Or... 
# run in bash...
# TODO: get this part working using these fast CLIs rather than boto3 as below
# HYP_STR=2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/current_date_smoothed_moving_window_21_days_countries_region_statsmodels/
# HYP_STR=2020_05_28_date_smoothed_moving_window_21_days_US_states_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/current_date_smoothed_moving_window_21_days_countries_region_statsmodels/
# HYP_STR=2020_05_28_date_smoothed_moving_window_21_days_US_counties_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/current_date_smoothed_moving_window_21_days_countries_region_statsmodels/
# HYP_STR=2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels; aws s3 cp --recursive state_plots/$HYP_STR s3://covid-figures/current_date_smoothed_moving_window_21_days_countries_region_statsmodels/

for region, plot_subfolder in region_plot_subfolders.items():
    hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder))
    print(hyperparamater_str)

    first_level_files = glob.glob(plot_subfolder + '/*.*', recursive=True)
    second_level_files = glob.glob(plot_subfolder + '/*/*.*', recursive=True)
    files = first_level_files + second_level_files

    # This is really inefficient -- takes 20-30 minutes! Mainly issue is that 
    #   `aws s3 cp --recursive ...` uses parallel threads, and this is a serial upload
    for file in tqdm(list(files)):
        relative_filename = '/'.join(file.split('/')[1:])
        print(f'Uploading {relative_filename}...')
        upload_file(file, 'covid-figures', object_name=relative_filename)

######
# Step 4: Generate Figure Browser
######

import os
import generate_plot_browser_moving_window_statsmodels_only as generate_figure_browser

for region, plot_subfolder in region_plot_subfolders.items():
    hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder)).format(date_str='2020_06_07')
    data_dir = plot_subfolder.format(date_str='2020_06_07')
    regions_to_present = [f for f in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, f))]
    print(sorted(regions_to_present))

    # Regenerate Figures
    hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder)).format(date_str='current')
    generate_figure_browser.hyperparameter_str = hyperparamater_str + '/'
    generate_figure_browser.plot_browser_dir = f'plot_browser_moving_window_statsmodels_only_{region}'
    generate_figure_browser.regions_to_present = regions_to_present
    generate_figure_browser.generate_plot_browser(regions_to_present)

######
# Step 5: Push to Github
######

# TODO: Update static_figures with most recent version
# TODO: Update README.md to link to new CSV files

# Do this by hand
# TODO: Figure out how to do this automatically instead of by hand
print('Now add, commit, and push to Github!')
