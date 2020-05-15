import covid_convolve as covid

covid.n_bootstraps = 100
covid.n_likelihood_samples = 100000
covid.max_date_str = '2020-05-14'
covid.opt_force_plot = False
covid.opt_force_calc = False
covid.override_run_states = None

covid.run_everything()
