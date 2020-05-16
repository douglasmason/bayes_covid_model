import os
import requests
import covid_moving_window as covid
from sub_units.bayes_model import ApproxType
import glob
import logging
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import datetime
import generate_plot_browser_moving_window_statsmodels_only as generate_figure_browser

'''
* Download the latest data to a local directory using curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv
* Update covid_moving_window.py to today's max_date_str (maybe I should set this up as argparse)
* Execute "import covid_moving_window as x; x.opt_simplified=True #monkey-patching; x.run_everything()"
* Upload the resulting directory (hyperparameter string) added to directory state_plots to AWS
* Re-run generate_plot_browser_moving_window_statsmodels_only.py with the updated hyperparameter string
* Push updated plot_browser_moving_window_statsmodels_only directory to GitHub
'''

#####
# Step 1: Update counts data
#####

url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
r = requests.get(url, allow_redirects=True)
with open('source_data/counts.csv', 'w') as f:
    f.write(r.content.decode("utf-8") )

#####
# Step 2: Run Update
#####

# get today's date
yesterdays_date_str = (datetime.date.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d')
print(f'Yesterday: {yesterdays_date_str}')

covid.n_bootstraps = 10
covid.n_likelihood_samples = 10000
covid.moving_window_size = 21  # three weeks
covid.max_date_str = yesterdays_date_str
covid.opt_force_calc = False
covid.opt_force_plot = True
covid.opt_simplified = True  # set to True to just do statsmodels as a simplified daily service
covid.override_run_states = ['Utah']

plot_subfolder = covid.run_everything()
hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder))
print(hyperparamater_str)

#####
# Step 3: Upload Figures to AWS
#####

plot_subfolder = 'state_plots/2020_05_15_date_smoothed_moving_window_21_days_statsmodels_only'
hyperparamater_str = os.path.basename(os.path.normpath(plot_subfolder))

first_level_files = glob.glob(plot_subfolder+'/*.*', recursive=True)
second_level_files = glob.glob(plot_subfolder+'/*/*.*', recursive=True)
files = first_level_files + second_level_files

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

# TODO: THis is really inefficient -- takes 30 minutes! Mainly issue is that 
#   `aws s3 cp --recursive ...` uses parallel threads, and this is a serial upload
for file in tqdm(list(files)):
    relative_filename = '/'.join(file.split('/')[1:])
    print(f'Uploading {relative_filename}...')
    upload_file(file, 'covid-figures', object_name=relative_filename)

######
# Step 4: Regenerate Figures
######

generate_figure_browser.hyperparameter_str = hyperparamater_str + '/'
generate_figure_browser.generate_plot_browser()

######
# Step 5: Push to Github
######

# Do this by hand
# TODO: Figure out how to do this automatically instead of by hand
print('Now add, commit, and push to Github!')