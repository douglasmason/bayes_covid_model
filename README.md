# covid_model

**Latest results and three-month projections provided in** 
* [US States Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_states/index.html)
* [US Counties Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_counties/index.html) 
* [International Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_countries/index.html)

**Tabulated projections can be found at**
* [US States Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_states_region_statsmodels/simplified_state_prediction.csv)
* [US Counties Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_counties_region_statsmodels/simplified_state_prediction.csv)  
* [International Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_prediction.csv)

**Parameter estimates can be found at**
* [US States Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_states_region_statsmodels/simplified_state_report.csv)
* [US Counties Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_counties_region_statsmodels/simplified_state_report.csv)
* [International Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_report.csv)

*Thank You to Digital Ocean and Black Sails Consulting for access to compute resources for this project.*

*We've  implemented a simple photo browser to help readers deep-dive. If you would like to design an interface for people to beautifully and easily find their nation, province, or county, and the metrics that matter to them, please contact the contributors.*

*To run the code that generates the paper figures, clone repo and execute `python paper_figures_convolution.py; python paper_figures_moving_window.py`.*

*To run the code that generates the daily updates, clone repo and execute `python daily_cron_job.py`.*

We model universal curves of reported COVID-19 daily infections and related deaths using a linear regression with standard errors and a weekly profile in the log space (making it an exponential regression in linear space). Using currently available data from [N. Y. Times](https://github.com/nytimes/covid-19-data) and [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19), we fit our model parameters to the most recent three weeks and provide projections for the next three months, assuming that the same growth rate continues during that time. In addition, we provide a time-series of growth rates for each region, as well estimates of the current week-over-week change in growth rate and its statistical significance, an indicator of where new waves or outbreaks may be occuring.

![boxplot](/static_figures/country_name_field.png?)
**Figure 1:** Week-over-week change in daily growth rate vs. daily growth rate among nations, filtered to nations where the likelihood to have this magnitude of change in daily growth rate or greater (the p-value) is less than 10%. Nations in the top-right are likely to be accelerating an already high growth rate, in the top-left are likely to be reversing negative growth rate, in the bottom-left are likely to be accelerating an already strongly negative growth rate, and in the bottom-right are likely to be reversing a positive growth rate. This plot is generated from the [International Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_21_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_report.csv).

![boxplot](/static_figures/statsmodels_solutions_filled_quantiles.png?)
**Figure 2a:** Three-week moving-window model prediction curves for three months and COVID-19 Daily Reported Cases and Related Deaths in the U.S. This is what we predict would happen if the trend from the last three weeks continued for the next three months. 5th-95th percentile and 25th-75th percentile regions are displayed in light and dark colors, respectively.

![boxplot](/static_figures/statsmodels_solutions_cumulative_filled_quantiles.png?)
**Figure 2b:** Three-week moving-window model prediction curves for three months and COVID-19 Cumulative Reported Cases and Related Deaths in the U.S. This is what we predict would happen if the trend from the last three weeks continued for the next three months. 5th-95th percentile and 25th-75th percentile regions are displayed in light and dark colors, respectively

![boxplot](/static_figures/statsmodels_growth_rate_time_series.png?)
**Figure 2c:** Three-week moving-window model growth rate curves for three months and COVID-19 Cumulative Reported Cases and Related Deaths in the U.S.

![boxplot](/static_figures/intl_simplified_boxplot_for_positive_slope_statsmodels.png?)
**Figure 3a:** Model parameter estimates for the current growth rate of COVID- 19 for each of the top 50 nations by current number of cases, with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. 

![boxplot](/static_figures/intl_simplified_boxplot_for_positive_slope_statsmodels_acc.png?)
**Figure 3b:** Model parameter estimates for the week-over-week change in growth rate of COVID- 19 for each of the top 50 nations by current number of cases, with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. States  higher in the list are candidates for new waves or outbreaks.

More figures can be found in the Figure Browser links at the top of this page. This work has not yet been peer reviewed.
