# covid_model

**Full write-up at [PDF version](https://covid-figures.s3-us-west-2.amazonaws.com/covid.pdf)**

**Per-state results provided at [Figure Browser](https://htmlpreview.github.io/?https://github.com/douglasmason/covid_model/blob/master/plot_browser_moving_window_statsmodels_only/index.html) and tabulated within the [CSV](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_12_date_moving_window_21_days_statsmodels_only/simplified_state_report.csv)**

*We are looking for more compute resources so we can provide results for each state on a daily basis at high fidelity, requiring 10x more samples. If you would like to help please contact the contributors.*

*We've currently implemented a simple photo browser to help readers deep-dive. If you would like to design an interface for people to easily find their state and the metrics that matter to them, please contact the contributors.*

*To run the code, clone repo and execute `import covid_convolve as covid; covid.run_everything()` or `import covid_moving_window as covid; covid.run_everything()`*

We model universal curves of reported COVID-19 daily reported infections and related deaths using a modified epidemiological Susceptible-Exposed-Infectious- Recovered (SEIR) Model[4, 1, 5]. Using currently available data, we determine optimized constants and apply this framework to reproducing the infection and death curves for California (the state with the largest population), New York (the state with highest population density), U.S. totals, and supplimentary results for the remaining 50 states and Washington D.C.

![boxplot](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_13_date_smoothed_moving_window_21_days_statsmodels_only/california/statsmodels_solutions_filled_quantiles.png)
**Figure 1:** Moving-Window Linear Regression model curves and COVID-19 Daily Reported Cases and Related Deaths in the U.S. 

![boxplot](https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_13_date_smoothed_moving_window_21_days_statsmodels_only/simplified_boxplot_for_positive_slope__statsmodels.png)
**Figure 2:** Model parameter estimates for the current growth rate of COVID- 19 for each of 50 U.S. states, Washington D.C., and U.S. totals with 5%, 25%, 50%, 75%, and 95% percentiles, ranked from highest to lowest median. We see strongest growth in Minnesota, Main, and South Dakota, and lowest growth in Montana, Arkansas, and Virginia.

[1] R.M. Anderson and R.M. May. Infectious Diseases of Humans: Dynamics and Control. Dynamics and Control. OUP Oxford, 1992.

[4] William Ogilvy Kermack, A. G. McKendrick, and Gilbert Thomas Walker. A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London. Series A, Containing Papers of a Mathematical and Physical Character, 115(772):700–721, 1927.

[5] Eduardo Massad. An introduction to infectious diseases modelling. by E. Vyn-Nycky and R. White. Oxford university press, 2010. Epidemiology and Infection, 139(7):1126– 1126, 2011.

This work has not been peer reviewed.
