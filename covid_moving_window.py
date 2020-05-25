from sub_units.bayes_model_implementations.moving_window_model import \
    MovingWindowModel  # want to make an instance of this class for each state / set of params
from sub_units.utils import run_everything as run_everything_imported  # for plotting the report across all states
from sub_units.utils import Region
import sub_units.load_data_country as load_data  # only want to load this once, so import as singleton pattern
import datetime

#####
# Set up model
#####

n_bootstraps = 100
n_likelihood_samples = 100000
moving_window_size = 21  # three weeks
opt_force_calc = False
opt_force_plot = False
opt_simplified = False  # set to True to just do statsmodels as a simplified daily service
override_run_states = None
override_max_date_str = None


# ['total', 'Virginia', 'Arkansas', 'Connecticut', 'Alaska', 'South Dakota', 'Hawaii', 'Vermont', 'Wyoming'] # None

###
# Execute
###

def run_everything():

    if override_run_states is not None:
        countries_plot_subfolder = _run_everything_sub(region=Region.countries, override_run_states=override_run_states)
        return {Region.countries: countries_plot_subfolder}
    else:
        countries_plot_subfolder = _run_everything_sub(region=Region.countries)
        us_states_plot_subfolder = _run_everything_sub(region=Region.US_states)
        us_counties_plot_subfolder = _run_everything_sub(region=Region.US_counties)
        provinces_plot_subfolder = _run_everything_sub(region=Region.provinces)
        return {Region.US_states: us_states_plot_subfolder, Region.countries: countries_plot_subfolder,
            Region.US_counties: us_counties_plot_subfolder, Region.provinces: provinces_plot_subfolder}


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
        if tmp_dict['series_data'].shape[1] < 3 or tmp_dict['series_data'].shape[0] < 30 or province.startswith('China:'):
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
            override_run_states = load_data.current_cases_ranked_us_counties[:50]
        elif region == Region.countries:
            override_run_states = load_data.current_cases_ranked_non_us_states[:50]
        elif region == Region.provinces:
            override_run_states = load_data.current_cases_ranked_non_us_provinces[:50]
    
        override_run_states = [x for x in override_run_states if not x.startswith(' ')]

    print('Gonna run these states:')
    [print(x) for x in sorted(override_run_states)]

    model_type_name = f'moving_window_{moving_window_size}_days_{region}_region'

    if override_max_date_str is None:
        hyperparameter_max_date_str = datetime.datetime.today().strftime('%Y-%m-%d')
    else:
        hyperparameter_max_date_str = override_max_date_str

    state_models_filename = f'state_models_smoothed_moving_window_{region}_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{hyperparameter_max_date_str.replace("-", "_")}_max_date.joblib'
    state_report_filename = f'state_report_smoothed_moving_window_{region}_{n_bootstraps}_bootstraps_{n_likelihood_samples}_likelihood_samples_{hyperparameter_max_date_str.replace("-", "_")}_max_date.joblib'

    # fixing parameters I don't want to train for saves a lot of computer power
    extra_params = dict()
    static_params = {'day0_positive_multiplier': 1,
                     'day0_deceased_multiplier': 1}
    logarithmic_params = ['positive_intercept',
                          'deceased_intercept',
                          'sigma_positive',
                          'sigma_deceased',
                          # 'day0_positive_multiplier',
                          'day1_positive_multiplier',
                          'day2_positive_multiplier',
                          'day3_positive_multiplier',
                          'day4_positive_multiplier',
                          'day5_positive_multiplier',
                          'day6_positive_multiplier',
                          # 'day0_deceased_multiplier',
                          'day1_deceased_multiplier',
                          'day2_deceased_multiplier',
                          'day3_deceased_multiplier',
                          'day4_deceased_multiplier',
                          'day5_deceased_multiplier',
                          'day6_deceased_multiplier',
                          ]
    plot_param_names = ['positive_slope',
                        'positive_intercept',
                        'deceased_slope',
                        'deceased_intercept',
                        'sigma_positive',
                        'sigma_deceased'
                        ]
    if opt_simplified:
        plot_param_names = ['positive_slope',
                            'deceased_slope',
                            'positive_intercept',
                            'deceased_intercept', ]
    sorted_init_condit_names = list()
    sorted_param_names = ['positive_slope',
                          'positive_intercept',
                          'deceased_slope',
                          'deceased_intercept',
                          'sigma_positive',
                          'sigma_deceased',
                          # 'day0_positive_multiplier',
                          'day1_positive_multiplier',
                          'day2_positive_multiplier',
                          'day3_positive_multiplier',
                          'day4_positive_multiplier',
                          'day5_positive_multiplier',
                          'day6_positive_multiplier',
                          # 'day0_deceased_multiplier',
                          'day1_deceased_multiplier',
                          'day2_deceased_multiplier',
                          'day3_deceased_multiplier',
                          'day4_deceased_multiplier',
                          'day5_deceased_multiplier',
                          'day6_deceased_multiplier']

    curve_fit_bounds = {'positive_slope': (-10, 10),
                        'positive_intercept': (0, 1000000),
                        'deceased_slope': (-10, 10),
                        'deceased_intercept': (0, 1000000),
                        'sigma_positive': (0, 100),
                        'sigma_deceased': (0, 100),
                        # 'day0_positive_multiplier': (0, 10),
                        'day1_positive_multiplier': (0, 10),
                        'day2_positive_multiplier': (0, 10),
                        'day3_positive_multiplier': (0, 10),
                        'day4_positive_multiplier': (0, 10),
                        'day5_positive_multiplier': (0, 10),
                        'day6_positive_multiplier': (0, 10),
                        # 'day0_deceased_multiplier': (0, 10),
                        'day1_deceased_multiplier': (0, 10),
                        'day2_deceased_multiplier': (0, 10),
                        'day3_deceased_multiplier': (0, 10),
                        'day4_deceased_multiplier': (0, 10),
                        'day5_deceased_multiplier': (0, 10),
                        'day6_deceased_multiplier': (0, 10)
                        }
    test_params = {'positive_slope': 0,
                   'positive_intercept': 2500,
                   'deceased_slope': 0,
                   'deceased_intercept': 250,
                   'sigma_positive': 0.05,
                   'sigma_deceased': 0.1,
                   # 'day0_positive_multiplier': 1,
                   'day1_positive_multiplier': 1,
                   'day2_positive_multiplier': 1,
                   'day3_positive_multiplier': 1,
                   'day4_positive_multiplier': 1,
                   'day5_positive_multiplier': 1,
                   'day6_positive_multiplier': 1,
                   # 'day0_deceased_multiplier': 1,
                   'day1_deceased_multiplier': 1,
                   'day2_deceased_multiplier': 1,
                   'day3_deceased_multiplier': 1,
                   'day4_deceased_multiplier': 1,
                   'day5_deceased_multiplier': 1,
                   'day6_deceased_multiplier': 1
                   }

    # uniform priors with bounds:
    priors = curve_fit_bounds

    plot_subfolder = run_everything_imported(override_run_states,
                                             MovingWindowModel,
                                             load_data,
                                             model_type_name=model_type_name,
                                             state_models_filename=state_models_filename,
                                             state_report_filename=state_report_filename,
                                             moving_window_size=moving_window_size,
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
                                             opt_simplified=opt_simplified,
                                             override_max_date_str=override_max_date_str,
                                             )

    return plot_subfolder


if __name__ == '__main__':
    run_everything()
