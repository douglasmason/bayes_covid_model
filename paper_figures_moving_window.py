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
