# bayes_covid_model

**[Leading Countries, U.S. States, and U.S. State Counties by Likelihood of Case Acceleration](#countries-with-highest-likelihood-of-case-acceleration)**

**Latest results and three-month projections provided in** 
* [International Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_countries/index.html)
* [US States Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_states/index.html)
* [US Counties Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only_US_counties/index.html) 


**Tabulated projections can be found at**
* [International Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_prediction.csv)
* [US States Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_states_region_statsmodels/simplified_state_prediction.csv)
* [US Counties Projection CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_counties_region_statsmodels/simplified_state_prediction.csv)  

**Parameter estimates can be found at**
* [International Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_report.csv)
* [US States Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_states_region_statsmodels/simplified_state_report.csv)
* [US Counties Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_28_date_smoothed_moving_window_21_days_US_counties_region_statsmodels/simplified_state_report.csv)

*We gratefully acknowledge Digital Ocean and Black Sails Consulting for access to compute resources for this project.*

*We've  implemented a simple photo browser to help readers deep-dive. If you would like to design an interface for people to beautifully and easily find their nation, province, or county, and the metrics that matter to them, please contact the contributors.*

*To run the code that generates the paper figures, clone repo and execute `python paper_figures_convolution.py; python paper_figures_moving_window.py`, for the daily updates execute `python daily_cron_job.py`, and to refresh the tables execute `python post_analysis.py`*

We model universal curves of reported COVID-19 daily infections and related deaths using a linear regression with standard errors and a weekly profile in the log space (making it an exponential regression in linear space). Using currently available data from [N.Y. Times](https://github.com/nytimes/covid-19-data) and [Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19), we fit our model parameters to the most recent three weeks and provide projections for the next three months, assuming no new mitigating factors and the same growth rate continues during that time. In addition, we provide a time-series of growth rates for each locale, as well estimates for the current week-to-week change in growth rate and its statistical significance, an indicator of where new clusters or outbreaks may be occuring.

## Example Figures

![boxplot](/static_figures/country_name_field.png?)
**Figure 1:** Week-to-week change in daily growth rate vs. daily growth rate among nations, filtered to nations where the likelihood to have this magnitude of change in daily growth rate or greater (the p-value) is less than 10%. Nations in the top-right are likely to be accelerating an already high growth rate, in the top-left are likely to be reversing negative growth rate, in the bottom-left are likely to be accelerating an already strongly negative growth rate, and in the bottom-right are likely to be reversing a positive growth rate. This plot is generated from the [International Parameters CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_21_date_smoothed_moving_window_21_days_countries_region_statsmodels/simplified_state_report.csv).

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

More figures can be found in the Figure Browser links at the top of this page. 

## Countries with Highest Likelihood of Case Acceleration
Rank|Country|Infections Growth Rate|p-value|Week-to-Week Change|p-value
-|-|-|-|-|-
1|[Iraq](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/iraq/index.html)|0.06718|1.164e-21|0.04007|6.024e-06
2|[Malaysia](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/malaysia/index.html)|0.03432|0.03329|0.08575|1.837e-05
3|[Netherlands](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/netherlands/index.html)|-0.0227|0.000155|0.0279|2.75e-05
4|[Sri Lanka](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/sri_lanka/index.html)|0.07299|0.0001027|0.08605|5.734e-05
5|[Canada](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/canada/index.html)|-0.01282|0|0.0148|0.0001135
6|[Panama](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/panama/index.html)|0.0226|0.01916|0.0443|0.0001369
7|[Cuba](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/cuba/index.html)|-0.0262|0.0002659|0.04499|0.0001861
8|[Norway](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/norway/index.html)|-0.03375|8.384e-06|0.03816|0.0007276
9|[Armenia](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/armenia/index.html)|0.06931|1.667e-63|0.02143|0.002121
10|[Israel](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/israel/index.html)|-0.0398|0.008287|0.05125|0.003912
11|[Philippines](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/philippines/index.html)|0.01087|0.04006|0.01899|0.006799
12|[Serbia](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/serbia/index.html)|-0.01761|0.03975|0.05936|0.009622
13|[Iceland](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/iceland/index.html)|0.01837|0.2172|0.08662|0.01015
14|[Kazakhstan](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/kazakhstan/index.html)|0.05878|3.823e-07|0.04355|0.01022
15|[Honduras](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/honduras/index.html)|0.0541|5.363e-07|0.0396|0.0145
16|[US](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/us/index.html)|-0.006491|8.903e-06|0.005307|0.01564
17|[Kenya](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/kenya/index.html)|0.06793|1.478e-08|0.03323|0.03585
18|[Egypt](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/egypt/index.html)|0.04671|1.337e-11|0.01257|0.0768
19|[Afghanistan](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/afghanistan/index.html)|0.05997|2.821e-34|0.00946|0.08886
20|[Chile](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_countries/chile/index.html)|0.06894|4.322e-67|0.008531|0.0969


## U.S. States with Highest Likelihood of Case Acceleration
Rank|State|Infections Growth Rate|p-value|Week-to-Week Change|p-value
-|-|-|-|-|-
1|[US: Wisconsin](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_wisconsin/index.html)|0.02899|9.347e-28|0.02669|1.04e-06
2|[US: Vermont](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_vermont/index.html)|0.02526|0.07554|0.09725|2.256e-05
3|[US: West Virginia](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_west_virginia/index.html)|0.0535|1.002e-06|0.05416|0.000119
4|[US: Missouri](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_missouri/index.html)|0.01135|0.07167|0.04095|0.000157
5|[US: Mississippi](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_mississippi/index.html)|0.01492|1.877e-08|0.0182|0.0003864
6|[US: South Carolina](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_south_carolina/index.html)|0.01925|2.176e-08|0.01375|0.001381
7|[US: New York](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_new_york/index.html)|-0.03081|0|0.01007|0.001829
8|[US: Iowa](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_iowa/index.html)|-0.006295|0.1153|0.02452|0.002592
9|[US: Puerto Rico](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_puerto_rico/index.html)|0.02813|5.085e-06|0.02648|0.01715
10|[US: total](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_total/index.html)|-0.006226|1.817e-05|0.00555|0.0181
11|[US: Alabama](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_alabama/index.html)|0.03135|8.287e-06|0.02027|0.02034
12|[US: California](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_california/index.html)|0.01554|5.189e-07|0.009453|0.03244
13|[US: Tennessee](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_tennessee/index.html)|0.007519|0.124|0.03216|0.03422
14|[US: Georgia](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_georgia/index.html)|0.002501|0.3425|0.01507|0.03438
15|[US: Guam](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_states/us_guam/index.html)|0.06501|0.02727|0.0728|0.06838

## U.S. Counties with Highest Likelihood of Case Acceleration, Among Top 500 by Current Case Count
Rank|State:County|7-Day Avg. New Daily Infections|Infections Day-over-Day Growth Rate|Week-to-Week Change in Day-over-Day Growth Rate|p-value
-|-|-|-|-|-
1|[US: Georgia: Hall](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_georgia_hall/index.html)|2425|0.44550665061711126%|13.2785697686999%|7.721e-09
2|[US: Texas: Galveston](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_texas_galveston/index.html)|786|6.302683206964121%|16.6143201508395%|2.732e-08
3|[US: New York: Nassau](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_nassau/index.html)|40140|-4.187621952135601%|1.8945312195434827%|1.047e-07
4|[US: Missouri: Kansas City](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_missouri_kansas_city/index.html)|1108|3.0653576780688807%|6.7368962854851056%|1.525e-07
5|[US: Tennessee: Rutherford](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_tennessee_rutherford/index.html)|1121|1.7823777592386403%|4.440839467425395%|4.607e-06
6|[US: New York: Erie](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_erie/index.html)|5881|0.8664281183732525%|3.8967150899967162%|7.126e-06
7|[US: Minnesota: Nobles](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_minnesota_nobles/index.html)|1497|-5.2018500333035345%|5.684430927103068%|8.415e-06
8|[US: California: Los Angeles](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_california_los_angeles/index.html)|50360|2.477041680456882%|2.3470814643357762%|1.165e-05
9|[US: Georgia: DeKalb](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_georgia_dekalb/index.html)|3588|4.7292995030758345%|5.56735430141575%|1.955e-05
10|[US: New Jersey: Middlesex](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_middlesex/index.html)|15656|-3.153835077289824%|3.560004721053578%|1.994e-05
11|[US: Louisiana: St. Tammany](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_louisiana_st._tammany/index.html)|1701|0.6597348764091349%|6.762937123928414%|2.084e-05
12|[US: California: Santa Clara](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_california_santa_clara/index.html)|2704|4.419048444890961%|4.7580265718993875%|2.198e-05
13|[US: Tennessee: Shelby](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_tennessee_shelby/index.html)|4726|1.899313022192728%|2.635771807609566%|0.0001379
14|[US: New Jersey: Burlington](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_burlington/index.html)|4479|-1.4056647424016133%|3.8056073045124927%|0.0001522
15|[US: Ohio: Summit](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_ohio_summit/index.html)|1336|2.431067571665335%|6.284021421143682%|0.0001883
16|[US: California: San Mateo](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_california_san_mateo/index.html)|2025|2.730805543583825%|3.5219510776513863%|0.0002256
17|[US: Pennsylvania: Lancaster](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_pennsylvania_lancaster/index.html)|3052|2.400482457110864%|2.8962481265883744%|0.0002304
18|[US: Iowa: Dallas](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_iowa_dallas/index.html)|882|-3.9152333002233197%|7.5359402768440775%|0.0004479
19|[US: New Mexico: Bernalillo](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_mexico_bernalillo/index.html)|1415|-0.47145348816035604%|3.6394656455323027%|0.0004769
20|[US: Oklahoma: Tulsa](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_oklahoma_tulsa/index.html)|967|3.597989399835222%|5.7396952265653%|0.0005783
21|[US: New York: Oneida](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_oneida/index.html)|961|2.2742015108432545%|4.7830898056848925%|0.0006966
22|[US: New York: New York City](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_new_york_city/index.html)|205819|-3.1017096519655007%|1.7791554961117617%|0.0007389
23|[US: Pennsylvania: York](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_pennsylvania_york/index.html)|971|0.46997883748030045%|3.0580371938629725%|0.000821
24|[US: Virginia: Loudoun](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_virginia_loudoun/index.html)|2336|6.40799066368416%|5.552918475455781%|0.0009831
25|[US: New Jersey: Bergen](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_bergen/index.html)|18107|-0.298423883982013%|7.117931263644528%|0.0009989
26|[US: Georgia: Clayton](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_georgia_clayton/index.html)|1193|2.2989643381853053%|9.657792274296128%|0.001404
27|[US: Florida: Broward](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_florida_broward/index.html)|6917|-0.7325668014260711%|2.554603461443672%|0.001416
28|[US: Georgia: Gwinnett](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_georgia_gwinnett/index.html)|3555|5.219067749365802%|9.714464966235425%|0.001582
29|[US: New Jersey: Passaic](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_passaic/index.html)|15963|-3.825543710948409%|3.6130983443672937%|0.001963
30|[US: Utah: Utah](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_utah_utah/index.html)|1803|-0.1808352867827212%|3.93619691929592%|0.002458
31|[US: Florida: Hillsborough](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_florida_hillsborough/index.html)|2045|4.066413860305507%|4.318108346287475%|0.002501
32|[US: Missouri: St. Charles](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_missouri_st._charles/index.html)|761|0.538969954638624%|7.174378042968255%|0.002503
33|[US: New York: Westchester](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_westchester/index.html)|33269|-3.839525664696153%|1.1277214674052072%|0.0034
34|[US: Ohio: Mahoning](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_ohio_mahoning/index.html)|1388|-1.4694559515511685%|4.41934146748606%|0.003595
35|[US: New York: Rockland](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_rockland/index.html)|13073|-3.0011551621623056%|2.450247682772244%|0.003709
36|[US: Tennessee: Trousdale](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_tennessee_trousdale/index.html)|1392|-5.811134931409912%|25.250564013253474%|0.003798
37|[US: Virginia: Harrisonburg city](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_virginia_harrisonburg_city/index.html)|755|2.0329606427072378%|7.219794013501257%|0.004107
38|[US: California: Kings](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_california_kings/index.html)|765|4.131743040884817%|7.65595422309826%|0.004429
39|[US: Connecticut: New London](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_connecticut_new_london/index.html)|1065|2.2391369138764716%|7.216325144844449%|0.005065
40|[US: Georgia: Fulton](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_georgia_fulton/index.html)|4358|1.6966571944947884%|8.015725428997333%|0.005243
41|[US: Pennsylvania: Lebanon](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_pennsylvania_lebanon/index.html)|941|-2.3072627891155273%|6.55356831391618%|0.005591
42|[US: Florida: Volusia](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_florida_volusia/index.html)|705|4.224412792146115%|5.630722973627633%|0.005815
43|[US: Indiana: Allen](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_indiana_allen/index.html)|1440|3.3541388783324466%|2.309559896127289%|0.006697
44|[US: Washington: Snohomish](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_washington_snohomish/index.html)|3316|-1.520461191001632%|4.826141209979173%|0.01108
45|[US: Indiana: Cass](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_indiana_cass/index.html)|1588|-11.053669295684188%|8.811482622963783%|0.01162
46|[US: New York: Orange](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_orange/index.html)|10342|-4.413894109630501%|0.8254967736097019%|0.01328
47|[US: Mississippi: Hinds](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_mississippi_hinds/index.html)|951|1.3808506953578714%|3.7213592975042786%|0.01746
48|[US: Washington: Benton](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_washington_benton/index.html)|914|-1.356482267679715%|5.618928547883506%|0.01896
49|[US: Missouri: St. Louis city](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_missouri_st._louis_city/index.html)|1901|-0.5432049211658629%|4.273679918682927%|0.01931
50|[US: New York: Putnam](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_york_putnam/index.html)|1234|0.34517016713679904%|3.6679920422655554%|0.01972
51|[US: North Carolina: Durham](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_north_carolina_durham/index.html)|1444|6.685019089336142%|3.6289389847162346%|0.01976
52|[US: New Jersey: Warren](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_warren/index.html)|1147|-3.2588464102311865%|5.162426614385285%|0.0203
53|[US: Louisiana: Ouachita](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_louisiana_ouachita/index.html)|1168|6.299157944219913%|6.003923529856525%|0.02033
54|[US: Virginia: Henrico](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_virginia_henrico/index.html)|1641|4.563699724004744%|2.884726657742731%|0.02164
55|[US: Louisiana: Orleans](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_louisiana_orleans/index.html)|7062|0.7028934278240806%|3.2296550098102283%|0.02191
56|[US: Puerto Rico: Unknown](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_puerto_rico_unknown/index.html)|3515|2.8230485483598633%|2.5480448694846114%|0.02314
57|[US: New Jersey: Morris](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_morris/index.html)|6342|-2.687835725806631%|2.9235632005526924%|0.02416
58|[US: Pennsylvania: Philadelphia](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_pennsylvania_philadelphia/index.html)|22166|-2.022638867729254%|2.011321732872662%|0.02576
59|[US: Rhode Island: Providence](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_rhode_island_providence/index.html)|10887|18.149021856475844%|45.18862807132278%|0.03012
60|[US: Tennessee: Davidson](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_tennessee_davidson/index.html)|5112|1.2965233466617931%|2.800848445473214%|0.03192
61|[US: Nevada: Washoe](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_nevada_washoe/index.html)|1502|3.332147560395615%|3.0330808905820517%|0.03512
62|[US: Iowa: Black Hawk](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_iowa_black_hawk/index.html)|1719|-3.2700183514388192%|4.437476562856513%|0.03519
63|[US: Maryland: Harford](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_maryland_harford/index.html)|832|-1.283076355758106%|2.4344157638035435%|0.03639
64|[US: New Jersey: Sussex](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_sussex/index.html)|1099|-4.6700164385806815%|3.721821737245312%|0.03887
65|[US: California: Contra Costa](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_california_contra_costa/index.html)|1389|3.7197238293543267%|2.8950274414003996%|0.04041
66|[US: Tennessee: Sumner](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_tennessee_sumner/index.html)|854|3.5947132768465284%|2.677727472607838%|0.04265
67|[US: Texas: Fort Bend](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_texas_fort_bend/index.html)|1788|-0.5689866718205705%|3.400558174252555%|0.04331
68|[US: Alabama: Mobile](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_alabama_mobile/index.html)|2126|0.8903509378424657%|1.6288246821120511%|0.05145
69|[US: Wisconsin: Brown](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_wisconsin_brown/index.html)|2289|-7.157926989043705%|2.8005593139360796%|0.05506
70|[US: Kansas: Wyandotte](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_kansas_wyandotte/index.html)|1365|-4.713832271466744%|2.9080849055997726%|0.0566
71|[US: Michigan: Ingham](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_michigan_ingham/index.html)|722|-1.7961685605873745%|1.7519449156216949%|0.06215
72|[US: Ohio: Franklin](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_ohio_franklin/index.html)|5583|-1.4817908804169888%|1.1343078272185947%|0.06431
73|[US: Georgia: Cobb](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_georgia_cobb/index.html)|2932|-0.29782763991467176%|3.5662750593197456%|0.06526
74|[US: New Jersey: Mercer](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_mercer/index.html)|6693|-2.0389337632383375%|2.2022132190078425%|0.06549
75|[US: Maryland: Frederick](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_maryland_frederick/index.html)|1827|2.2051571955585825%|1.7771550185435636%|0.06681
76|[US: California: Alameda](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_california_alameda/index.html)|3146|3.5177861858110004%|1.2903024965981835%|0.07811
77|[US: Rhode Island: Kent](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_rhode_island_kent/index.html)|1036|9.332066149687467%|23.982138226969663%|0.08073
78|[US: Wisconsin: Racine](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_wisconsin_racine/index.html)|1580|3.8760022971163766%|1.9660696605246386%|0.08468
79|[US: New Jersey: Ocean](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_ocean/index.html)|8579|-2.8519897286810147%|1.4861566106885382%|0.08686
80|[US: New Jersey: Somerset](https://htmlpreview.github.io/?https://raw.githubusercontent.com/douglasmason/covid_model/master/plot_browser_moving_window_statsmodels_only_US_counties/us_new_jersey_somerset/index.html)|4528|-3.851808793199263%|1.93549276991285%|0.09089



This work is provided by [Koyote Science, LLC](http://www.koyotescience.com) and [Nexus iR&D Laboratory, LLC](http://www.nexusilab.com), and has not been peer reviewed.
