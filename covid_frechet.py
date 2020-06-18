from sub_units.bayes_model_implementations.frechet_model import \
    FrechetModel  # want to make an instance of this class for each state / set of params
from sub_units.utils import run_everything as run_everything_imported  # for plotting the report across all states
from sub_units.utils import Region
import sub_units.load_data_country as load_data  # only want to load this once, so import as singleton pattern
import datetime

#####
# Set up model
#####

n_bootstraps = 100
n_likelihood_samples = 100000
opt_force_plot = False
opt_force_calc = False
override_run_states = None
# ['Kansas', 'New York', 'Alaska', 'South Dakota', 'Wyoming', 'Arkansas', 'Arizona', 'Virginia']  #['New York', 'total'] #['North Carolina', 'Michigan', 'Georgia']
# ['total', 'Virginia', 'Arkansas', 'Connecticut', 'Alaska', 'South Dakota', 'Hawaii', 'Vermont', 'Wyoming'] # None
override_max_date_str = None


####
# Make whisker plots
####

def run_everything():
    if override_run_states is not None:
        countries_plot_subfolder = _run_everything_sub(region=Region.countries, override_run_states=override_run_states)
        return {Region.countries: countries_plot_subfolder}
    else:
        countries_plot_subfolder = _run_everything_sub(region=Region.countries)
        us_states_plot_subfolder = _run_everything_sub(region=Region.US_states)
        us_counties_plot_subfolder = _run_everything_sub(region=Region.US_counties)
        provinces_plot_subfolder = _run_everything_sub(region=Region.provinces)
        return {
            Region.US_states: us_states_plot_subfolder,
            Region.countries: countries_plot_subfolder,
            Region.US_counties: us_counties_plot_subfolder,
            Region.provinces: provinces_plot_subfolder
        }


def _run_everything_sub(region=Region.US_states, override_run_states=None):
    if type(load_data.current_cases_ranked_us_counties) == tuple:
        load_data.current_cases_ranked_us_counties = load_data.current_cases_ranked_us_counties[0]
    if type(load_data.current_cases_ranked_non_us_provinces) == tuple:
        load_data.current_cases_ranked_non_us_provinces = load_data.current_cases_ranked_non_us_provinces[0]

    load_data.current_cases_ranked_non_us_provinces = [x for x in load_data.current_cases_ranked_non_us_provinces \
                                                       if not x.startswith('US:')]

    # Remove provinces without enough data
    new_provinces = list()
    for province in load_data.current_cases_ranked_non_us_provinces:
        tmp_dict = load_data.get_state_data(province)
        if tmp_dict['series_data'].shape[1] < 3 or tmp_dict['series_data'].shape[0] < 30 or province.startswith(
                'China:'):
            print(f'Removing province {province}')
            if tmp_dict['series_data'].shape[1] >= 3:
                print(f'  with tmp_dict["series_data"].shape = {tmp_dict["series_data"].shape}')
        else:
            print(f'Keeping province {province}')
            print(f'  with tmp_dict["series_data"].shape = {tmp_dict["series_data"].shape}')
            new_provinces.append(province)
    load_data.current_cases_ranked_non_us_provinces = new_provinces

    print('load_data.current_cases_ranked_non_us_provinces')
    [print(x) for x in sorted(load_data.current_cases_ranked_non_us_provinces)]

    if override_run_states is None:
        if region == Region.US_states:
            override_run_states = load_data.current_cases_ranked_us_states
        elif region == Region.US_counties:
            override_run_states = load_data.current_cases_ranked_us_counties[:70]
        elif region == Region.countries:
            override_run_states = load_data.current_cases_ranked_non_us_states[:70]
        elif region == Region.provinces:
            override_run_states = load_data.current_cases_ranked_non_us_provinces[:70]

        override_run_states = [x for x in override_run_states if not x.startswith(' ')]

    print('Gonna run these states:')
    [print(x) for x in sorted(override_run_states)]

    model_type_name = f'frechet_{region}_region'

    if override_max_date_str is None:
        hyperparameter_max_date_str = datetime.datetime.today().strftime('%Y-%m-%d')
    else:
        hyperparameter_max_date_str = override_max_date_str

    state_models_filename = f'state_models_smoothed_frechet_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{hyperparameter_max_date_str.replace("-", "_")}_max_date.joblib'
    state_report_filename = f'state_report_smoothed_frechet_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{hyperparameter_max_date_str.replace("-", "_")}_max_date.joblib'

    # fixing parameters I don't want to train for saves a lot of computer power
    static_params = dict() #{'mult_positive': 3400,
                           #'mult_deceased': 3400})
    logarithmic_params = ['alpha_positive',
                          's_positive',
                          'mult_positive',
                          'alpha_deceased',
                          's_deceased',
                          'mult_deceased',
                          'sigma_positive',
                          'sigma_deceased',
                          ]

    sorted_init_condit_names = list()
    sorted_param_names = ['alpha_positive',
                          's_positive',
                          'm_positive',
                          'mult_positive',
                          'alpha_deceased',
                          's_deceased',
                          'm_deceased',
                          'mult_deceased',
                          'sigma_positive',
                          'sigma_deceased',
                          ]

    plot_param_names = ['alpha_positive',
                        's_positive',
                        'm_positive',
                        'mult_positive',
                        'alpha_deceased',
                        's_deceased',
                        'm_deceased',
                        'mult_deceased',
                        'sigma_positive',
                        'sigma_deceased',
                        ]

    extra_params = dict()

    curve_fit_bounds = {'alpha_positive': (1e-8, 1000),
                        's_positive': (1e-3, 1e8),
                        'm_positive': (-1000, 1000),
                        'mult_positive': (1e-8, 1e12),
                        'alpha_deceased': (1e-8, 1000),
                        's_deceased': (1e-3, 1e8),
                        'm_deceased': (-1000, 1000),
                        'mult_deceased': (1e-8, 1e12),
                        'sigma_positive': (0, 100),
                        'sigma_deceased': (0, 100),
                        }

    test_params = {'alpha_positive': 8,
                   's_positive': 200,
                   'm_positive': -50,
                   'mult_positive': 2e6,
                   'alpha_deceased': 7,
                   's_deceased': 140,
                   'm_deceased': -30,
                   'mult_deceased': 1e5,
                   'sigma_positive': 0.5,
                   'sigma_deceased': 0.3,
                   }

    # uniform priors with bounds:
    priors = curve_fit_bounds

    plot_subfolder = run_everything_imported(override_run_states,
                                             FrechetModel,
                                             load_data,
                                             model_type_name=model_type_name,
                                             state_models_filename=state_models_filename,
                                             state_report_filename=state_report_filename,
                                             n_bootstraps=n_bootstraps,
                                             n_likelihood_samples=n_likelihood_samples,
                                             load_data_obj=load_data,
                                             sorted_param_names=sorted_param_names,
                                             sorted_init_condit_names=sorted_init_condit_names,
                                             curve_fit_bounds=curve_fit_bounds,
                                             priors=priors,
                                             test_params=test_params,
                                             static_params=static_params,
                                             opt_force_calc=opt_force_calc,
                                             opt_force_plot=opt_force_plot,
                                             logarithmic_params=logarithmic_params,
                                             extra_params=extra_params,
                                             plot_param_names=plot_param_names,
                                             opt_statsmodels=True,
                                             override_max_date_str=override_max_date_str,
                                             opt_convert_growth_rate_to_percent=False
                                             )

    return plot_subfolder


if __name__ == '__main__':
    run_everything()
