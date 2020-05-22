# covid_model

**Latest per-state results and three-month projections provided in the [US Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_states/index.html) and [International Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_countries/index.html)**

**Tabulated projections can be found within the [US CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_21_date_smoothed_moving_window_21_days_US_states_region_statsmodels/simplified_state_prediction.csv) and [International CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_21_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_prediction.csv)**

*Thank You to Digital Ocean and Black Sails Consulting for access to compute resources for this project.*

*We've  implemented a simple photo browser to help readers deep-dive. If you would like to design an interface for people to beautifully and easily find their nation, province, or county, and the metrics that matter to them, please contact the contributors.*

*To run the code that generates the paper figures, clone repo and execute `python paper_figures_convolution.py; python paper_figures_moving_window.py`.*

*To run the code that generates the daily updates, clone repo and execute `python daily_cron_job.py`.*

We model universal curves of reported COVID-19 daily  infections and related deaths using a linear regression with standard errors and a weekly profile in the log space (making it an exponential regression in linear space). Using currently available data from [N. Y. Times](https://github.com/nytimes/covid-19-data) and [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19), we fit our model parameters to the most recent three weeks and provide projections for the next three months, assuming that the same growth rate continues during that time.

![boxplot](/static_figures/statsmodels_solutions_filled_quantiles.png?)
**Figure 1a:** Three-week moving-window model prediction curves for three months and COVID-19 Daily Reported Cases and Related Deaths in the U.S. This is what we predict would happen if the trend from the last three weeks continued for the next three months.

![boxplot](/static_figures/statsmodels_solutions_cumulative_filled_quantiles.png?)
**Figure 1b:** Three-week moving-window model prediction curves for three months and COVID-19 Comulative Reported Cases and Related Deaths in the U.S. This is what we predict would happen if the trend from the last three weeks continued for the next three months.

![boxplot](/static_figures/simplified_boxplot_for_positive_slope_statsmodels.png?)
**Figure 2a:** Model parameter estimates for the current growth rate of COVID- 19 for each of 50 U.S. states, Washington D.C., and U.S. totals with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. 

![boxplot](/static_figures/intl_simplified_boxplot_for_positive_slope_statsmodels.png?)
**Figure 2b:** Model parameter estimates for the current growth rate of COVID- 19 for each of the top 50 nations by current number of cases, with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. 

This work has not been peer reviewed.
