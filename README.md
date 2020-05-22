# covid_model

**Full write-up at [PDF version](https://covid-figures.s3-us-west-2.amazonaws.com/covid.pdf)**

**Latest per-state results and three-month projections provided in the [US Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_states/index.html) and [International Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_countries/index.html) and tabulated within the [US CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_21_date_smoothed_moving_window_21_days_US_states_region_statsmodels/simplified_state_prediction.csv) and [International CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_21_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_prediction.csv)**

~~*We are looking for more compute resources so we can provide results for each state on a daily basis at high fidelity, requiring 10x more samples. If you would like to help please contact the contributors.*~~

*Thank You to Digital Ocean and Black Sails Consulting for access to compute resources for this project.*

*We've  implemented a simple photo browser to help readers deep-dive. If you would like to design an interface for people to beautifully and easily find their state and the metrics that matter to them, please contact the contributors.*

*To run the code that generates the paper figures, clone repo and execute `python paper_figures_convolution.py; python paper_figures_moving_window.py`*

*To run the code that generates the daily updates, clone repo and execute `python daily_cron_job.py`.

We model universal curves of reported COVID-19 daily reported infections and related deaths using a modified epidemiological Susceptible-Exposed-Infectious- Recovered (SEIR) Model[4, 1, 5]. Using currently available data, we determine optimized constants and apply this framework to reproducing the infection and death curves for California (the state with the largest population), New York (the state with highest population density), U.S. totals, and supplimentary results for the remaining 50 states and Washington D.C.

![boxplot](/static_figures/statsmodels_solutions_filled_quantiles.png?)
**Figure 1a:** Moving-Window Linear Regression model curves and COVID-19 Daily Reported Cases and Related Deaths in the U.S. 

![boxplot](/static_figures/statsmodels_solutions_cumulative_filled_quantiles.png?)
**Figure 1b:** Moving-Window Linear Regression model curves and COVID-19 Daily Reported Cases and Related Deaths in the U.S. 

![boxplot](/static_figures/simplified_boxplot_for_positive_slope_statsmodels.png?)
**Figure 2:** Model parameter estimates for the current growth rate of COVID- 19 for each of 50 U.S. states, Washington D.C., and U.S. totals with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. We see strongest growth in Minnesota, Main, and South Dakota, and lowest growth in Montana, Arkansas, and Virginia.

[1] R.M. Anderson and R.M. May. Infectious Diseases of Humans: Dynamics and Control. Dynamics and Control. OUP Oxford, 1992.

[4] William Ogilvy Kermack, A. G. McKendrick, and Gilbert Thomas Walker. A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London. Series A, Containing Papers of a Mathematical and Physical Character, 115(772):700–721, 1927.

[5] Eduardo Massad. An introduction to infectious diseases modelling. by E. Vyn-Nycky and R. White. Oxford university press, 2010. Epidemiology and Infection, 139(7):1126– 1126, 2011.

This work has not been peer reviewed.
