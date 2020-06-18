import sub_units.load_data_country as load_data
from sub_units.utils import generate_plot_browser as generate_plot_browser_imported
from sub_units.utils import Region

#####
# Setup
#####
base_url_dir = 'https://covid-figures.s3-us-west-2.amazonaws.com/'
hyperparameter_str = 'current/'
github_url = 'https://github.com/douglasmason/covid_model'
plot_browser_dir = 'plot_browser_moving_window_statsmodels_only'
full_report_filename = 'full_report.html'

list_of_figures = [
    # 'statsmodels_param_distro_without_priors.png',
    # 'statsmodels_solutions_discrete.png',
    'statsmodels_growth_rate_time_series.png',
    'statsmodels_solutions_filled_quantiles.png',
    # 'statsmodels_solutions_cumulative_discrete.png',
    'statsmodels_solutions_cumulative_filled_quantiles.png',
]

list_of_figures_full_report = [
    'simplified_boxplot_for_positive_slope_statsmodels.png',
    'simplified_boxplot_for_deceased_slope_statsmodels.png',
    'simplified_boxplot_for_positive_slope_statsmodels_acc.png',
    'simplified_boxplot_for_deceased_slope_statsmodels_acc.png',
    # 'simplified_boxplot_for_positive_intercept_statsmodels.png',
    # 'simplified_boxplot_for_deceased_intercept_statsmodels.png'
]
regions_to_present = ['total', 'California', 'New York']


#####
# Execute
#####

def generate_plot_browser(regions_to_present):
    full_url_dir = base_url_dir + hyperparameter_str

    generate_plot_browser_imported(plot_browser_dir,
                                   full_url_dir,
                                   github_url,
                                   full_report_filename,
                                   list_of_figures,
                                   list_of_figures_full_report,
                                   regions_to_present)


if __name__ == '__main__':
    generate_plot_browser
