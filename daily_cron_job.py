'''
* Download the latest data to a local directory using curl https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv
* Update covid_moving_window.py to today's max_date_str (maybe I should set this up as argparse)
* Execute "import covid_moving_window as x; x.opt_simplified=True #monkey-patching; x.run_everything()"
* Upload the resulting directory (hyperparameter string) added to directory state_plots to AWS
* Re-run generate_plot_browser_moving_window_statsmodels_only.py with the updated hyperparameter string
* Push updated plot_browser_moving_window_statsmodels_only directory to GitHub
'''

import covid_moving_window as covid

covid.n_bootstraps = 100
covid.n_likelihood_samples = 100000
covid.moving_window_size = 21  # three weeks
covid.max_date_str = '2020-05-14'
covid.opt_force_calc = False
covid.opt_force_plot = False
covid.opt_simplified = False  # set to True to just do statsmodels as a simplified daily service
covid.override_run_states = None

covid.run_everything()
