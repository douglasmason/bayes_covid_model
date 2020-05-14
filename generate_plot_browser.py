import sub_units.load_data as load_data
from sub_units.utils import generate_plot_browser

#####
# Setup
#####

base_url_dir = 'https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_06_date_100_bootstraps_100000_likelihood_samples/'
github_url = 'https://github.com/douglasmason/covid_model'
plot_browser_dir = 'plot_browser'
full_report_filename = 'full_us_report.html'

population_ranked_state_names = sorted(load_data.map_state_to_population.keys(),
                                       key=lambda x: -load_data.map_state_to_population[x])
alphabetical_states = sorted(load_data.map_state_to_population.keys())
alphabetical_states.remove('total')
alphabetical_states = ['total'] + alphabetical_states

list_of_figures = [
    'all_data_solution.png',
    'bootstrap_solutions.png',
    'bootstrap_param_distro_without_priors.png',
    'MVN_samples_actual_vs_predicted_vals.png',
    'mean_of_MVN_samples_solution.png',
    'MVN_samples_correlation_matrix.png',
    'likelihood_sample_param_distro_without_priors.png',
    'MVN_random_walk_actual_vs_predicted_vals.png',
    'mean_of_MVN_random_walk_solution.png',
    'MVN_random_walk_correlation_matrix.png',
    'random_walk_param_distro_without_priors.png',
]

list_of_figures_full_report = [
    'boxplot_for_I_0_with_direct_samples.png',
    'boxplot_for_I_0_without_direct_samples.png',
    'boxplot_for_alpha_1_with_direct_samples.png',
    'boxplot_for_alpha_1_without_direct_samples.png',
    'boxplot_for_alpha_2_with_direct_samples.png',
    'boxplot_for_alpha_2_without_direct_samples.png',
    'boxplot_for_contagious_to_deceased_delay_with_direct_samples.png',
    'boxplot_for_contagious_to_deceased_delay_without_direct_samples.png',
    'boxplot_for_contagious_to_deceased_mult_with_direct_samples.png',
    'boxplot_for_contagious_to_deceased_mult_without_direct_samples.png',
    'boxplot_for_contagious_to_positive_delay_with_direct_samples.png',
    'boxplot_for_contagious_to_positive_delay_without_direct_samples.png',
    'boxplot_for_positive_to_deceased_delay_with_direct_samples.png',
    'boxplot_for_positive_to_deceased_delay_without_direct_samples.png',
    'boxplot_for_positive_to_deceased_mult_with_direct_samples.png',
    'boxplot_for_positive_to_deceased_mult_without_direct_samples.png',
]

#####
# Execute
#####

generate_plot_browser(plot_browser_dir, load_data, base_url_dir, github_url, full_report_filename, list_of_figures, list_of_figures_full_report)
