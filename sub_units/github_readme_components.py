hp_str = '2020_06_02_date_smoothed_moving_window_21_days_US_states_region_statsmodels'

header_fmt_str = f'''
# bayes_covid_model

**[Leading Countries, U.S. States, and U.S. Counties by Likelihood of Positive Exponential Growth in Infections](#countries-with-highest-likelihood-of-exponential-growth-in-infections)**

**[Leading Countries, U.S. States, and U.S. Counties by Likelihood of Case Acceleration](#countries-with-highest-likelihood-of-case-acceleration)**

**[Example Figures](#example-figures)**

**Latest results and three-month projections provided in** 
* [International Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_countries/index.html)
* [US States Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_states/index.html)
* [US Counties Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_counties/index.html) 


**Tabulated projections can be found at**
* [International Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/{hp_str}/simplified_state_prediction.csv)
* [US States Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/{hp_str}/simplified_state_prediction.csv)
* [US Counties Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/{hp_str}/simplified_state_prediction.csv)  

**Parameter estimates can be found at**
* [International Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/{hp_str}/simplified_state_report.csv)
* [US States Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/{hp_str}/simplified_state_report.csv)
* [US Counties Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/{hp_str}/simplified_state_report.csv)

*We gratefully acknowledge Digital Ocean and Black Sails Consulting for access to compute resources for this project.*

*We've  implemented a simple photo browser to help readers deep-dive. If you would like to design an interface for people to beautifully and easily find their nation, province, or county, and the metrics that matter to them, please contact the contributors.*

*To run the code that generates the paper figures, clone repo and execute `python paper_figures_convolution.py; python paper_figures_moving_window.py`, for the daily updates execute `python daily_cron_job.py`, and to refresh the tables execute `python post_analysis.py`*

We model universal curves of reported COVID-19 daily infections and related deaths using a linear regression with standard errors and a weekly profile in the log space (making it an exponential regression in linear space). Using currently available data from [N.Y. Times](https://github.com/nytimes/covid-19-data) and [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19), we fit our model parameters to the most recent three weeks and provide projections for the next three months, assuming no new mitigating factors and the same growth rate continues during that time. In addition, we provide a time-series of growth rates for each locale, as well estimates for the current week-to-week change in growth rate and its statistical significance, an indicator of where changinge behaviors may be increasing spread (when the new daily case count is high) or where new outbreaks may be occuring (when the new daily case count is low). In our tables published in this README file (but not in the CSVs), we elimiante any region in which the 7-day average of new infections daily is less than 5, since measuring relative growth for such low numbers becomes spurious.
'''

example_figures = '''
## Example Figures

![boxplot](/static_figures/acceleration_diagram.png?)
**Figure 1:** Schematic diagram demonstrating how the Week-over-Week Change in 3-Week Avg. Daily Relative Growth Rate is computed.

![boxplot](/static_figures/statsmodels_solutions_filled_quantiles.png?)
**Figure 2a:** Three-week moving-window model prediction curves for three months and COVID-19 Daily Reported Cases and Related Deaths in the U.S. This is what we predict would happen if the trend from the last three weeks continued for the next three months. 5th-95th percentile and 25th-75th percentile regions are displayed in light and dark colors, respectively.

![boxplot](/static_figures/statsmodels_solutions_cumulative_filled_quantiles.png?)
**Figure 2b:** Three-week moving-window model prediction curves for three months and COVID-19 Cumulative Reported Cases and Related Deaths in the U.S. This is what we predict would happen if the trend from the last three weeks continued for the next three months. 5th-95th percentile and 25th-75th percentile regions are displayed in light and dark colors, respectively

![boxplot](/static_figures/statsmodels_growth_rate_time_series.png?)
**Figure 2c:** Three-week moving-window model growth rate curves for three months and COVID-19 Cumulative Reported Cases and Related Deaths in the U.S.

![boxplot](/static_figures/intl_simplified_boxplot_for_positive_slope_statsmodels.png?)
**Figure 3b:** Model parameter estimates for the current growth rate of COVID- 19 for each of the top 50 nations by current number of cases, with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. 

![boxplot](/static_figures/simplified_boxplot_for_positive_slope_statsmodels.png?)
**Figure 3c:** Model parameter estimates for the current growth rate of COVID- 19 for the United States, with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. 

More figures can be found in the Figure Browser links at the top of this page.
'''

footer = '''
This work is provided by [Koyote Science, LLC](http://www.koyotescience.com) and in collaboration with [Nexus iR&D Laboratory, LLC](http://www.nexusilab.com), and has not been peer reviewed.
'''

def get_header():
    return header_fmt_str.format(hp_str=hp_str)

def get_all(github_table_filename=None):
    
    table_text = ''
    with open(github_table_filename, 'r') as f:
        for line in f.readlines():
            table_text += line
    
    return get_header() + table_text + example_figures + footer