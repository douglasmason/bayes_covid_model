import os
import requests
from sub_units.bayes_model import ApproxType
from sub_units.utils import Region
import glob
import logging
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import datetime

'''
* Download the latest data to a local directory using curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv
* Update covid_moving_window.py to today's max_date_str (maybe I should set this up as argparse)
* Execute "import covid_moving_window as x; x.opt_simplified=True #monkey-patching; x.run_everything()"
* Upload the resulting directory (hyperparameter string) added to directory state_plots to AWS
* Re-run generate_plot_browser_moving_window_statsmodels_only.py with the updated hyperparameter string
* Push updated plot_browser_moving_window_statsmodels_only directory to GitHub
'''


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
# Step 1a: Update counts data
#####

url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
r = requests.get(url, allow_redirects=True)
with open('source_data/states.csv', 'w') as f:
    f.write(r.content.decode("utf-8") )

url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
r = requests.get(url, allow_redirects=True)
with open('source_data/counties.csv', 'w') as f:
    f.write(r.content.decode("utf-8") )

# # from https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports


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
covid.override_run_states = None # ['Spain', 'Iceland']#covid.load_data.current_cases_ranked_non_us_states[:50] # if you want to do countries instead

region_plot_subfolders = covid.run_everything()

#####
# Step 3: Upload Figures to AWS
#####

for region, plot_subfolder in region_plot_subfolders.items():
    hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder))
    print(hyperparamater_str)

    first_level_files = glob.glob(plot_subfolder + '/*.*', recursive=True)
    second_level_files = glob.glob(plot_subfolder + '/*/*.*', recursive=True)
    files = first_level_files + second_level_files

    # TODO: This is really inefficient -- takes 20-30 minutes! Mainly issue is that 
    #   `aws s3 cp --recursive ...` uses parallel threads, and this is a serial upload
    for file in tqdm(list(files)):
        relative_filename = '/'.join(file.split('/')[1:])
        print(f'Uploading {relative_filename}...')
        upload_file(file, 'covid-figures', object_name=relative_filename)


######
# Step 5: Generate Figure Browser
######

import os
from sub_units.utils import Region
import generate_plot_browser_moving_window_statsmodels_only as generate_figure_browser

region_plot_subfolders = {Region.US_states: 'state_plots/2020_05_25_date_smoothed_moving_window_21_days_US_states_region_statsmodels',
 Region.countries: 'state_plots/2020_05_25_date_smoothed_moving_window_21_days_countries_region_statsmodels'}


# import importlib
# importlib.reload(generate_figure_browser)

for region, plot_subfolder in region_plot_subfolders.items():
    hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder))
    print(hyperparamater_str)

    data_dir = plot_subfolder
    regions_to_present = [f for f in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, f))]
    print(regions_to_present)

    # Regenerate Figures
    generate_figure_browser.hyperparameter_str = hyperparamater_str + '/'
    generate_figure_browser.plot_browser_dir = f'plot_browser_moving_window_statsmodels_only_{region}'
    generate_figure_browser.regions_to_present = regions_to_present
    generate_figure_browser.generate_plot_browser(regions_to_present)

######
# Step 6: Push to Github
######

# TODO: Update static_figures with most recent version
# TODO: Update README.md to link to new CSV file

# Do this by hand
# TODO: Figure out how to do this automatically instead of by hand
print('Now add, commit, and push to Github!')
