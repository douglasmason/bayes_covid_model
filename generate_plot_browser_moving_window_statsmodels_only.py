from yattag import Doc
import sub_units.load_data as load_data
from os import path
import os

#####
# Setup
#####

base_url_dir = 'https://covid-figures.s3-us-west-2.amazonaws.com/2020_05_12_date_moving_window_21_days_statsmodels_only/'
github_url = 'https://github.com/douglasmason/covid_model'
plot_browser_dir = 'plot_browser_moving_window_statsmodels_only'
full_report_filename = 'full_us_report.html'

list_of_figures = [
    'statsmodels_param_distro_without_priors.png',
    'statsmodels_solutions_discrete.png',
    'statsmodels_solutions_filled_quantiles.png',
]


list_of_figures_full_report = [
    'simplified_boxplot_for_positive_slope__statsmodels.png',
    'simplified_boxplot_for_deceased_slope__statsmodels.png'
]

#####
# Execute
#####

if not path.exists(plot_browser_dir):
    os.mkdir(plot_browser_dir)

population_ranked_state_names = sorted(load_data.map_state_to_population.keys(),
                                       key=lambda x: -load_data.map_state_to_population[x])
alphabetical_states = sorted(load_data.map_state_to_population.keys())
alphabetical_states.remove('total')
alphabetical_states = ['total'] + alphabetical_states


map_state_to_html = dict()
for state in alphabetical_states:

    state_lc = state.lower().replace(' ', '_')
    doc, tag, text = Doc(defaults={'title': f'Plots for {state}'}).tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            pass
        with tag('body'):
            with tag('div', id='photo-container'):
                with tag('ul'):
                    with tag('li'):
                        with tag('a', href='../index.html'):
                            text('<-- Back')
                    for figure_name in list_of_figures:
                        tmp_url = base_url_dir + state_lc + '/' + figure_name
                        with tag("a", href=tmp_url):
                            doc.stag('img', src=tmp_url, klass="photo", height="300", width="400")
                        with tag('li'):
                            with doc.tag("a", href=tmp_url):
                                doc.text(figure_name)
                        with tag('hr'):
                            pass

    result = doc.getvalue()
    map_state_to_html[state] = result

for state in map_state_to_html:
    state_lc = state.lower().replace(' ', '_')
    if not path.exists(path.join(plot_browser_dir, state_lc)):
        os.mkdir(path.join(plot_browser_dir, state_lc))
    with open(path.join(plot_browser_dir, path.join(state_lc, 'index.html')), 'w') as f:
        f.write(map_state_to_html[state])

#####
# Generate state-report page
#####


with open(path.join(plot_browser_dir, full_report_filename), 'w') as f:
    doc, tag, text = Doc(defaults={'title': f'Plots for Full U.S. Report'}).tagtext()
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            pass
        with tag('body'):
            with tag('div', id='photo-container'):
                with tag('ul'):
                    with tag('li'):
                        with tag('a', href='index.html'):
                            text('<-- Back')
                    for figure_name in list_of_figures_full_report:
                        tmp_url = base_url_dir + figure_name
                        with tag("a", href=tmp_url):
                            doc.stag('img', src=tmp_url, klass="photo", height="400", width="300")
                        with tag('li'):
                            with doc.tag("a", href=tmp_url):
                                doc.text(figure_name)
                        with tag('hr'):
                            pass
    f.write(doc.getvalue())

#####
# Generate landing page
#####

with open(path.join(plot_browser_dir, 'index.html'), 'w') as f:
    doc, tag, text = Doc(defaults={'title': f'Plots for {state}'}).tagtext()

    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            pass
        with tag('body'):
            with tag('ul'):
                with tag('li'):
                    with tag("a", href=github_url):
                        text('<-- Back to Repository')
                with tag('li'):
                    with tag("a", href=full_report_filename):
                        text('Full U.S. Report')
                for state in alphabetical_states:
                    state_lc = state.lower().replace(' ', '_')
                    tmp_url = state_lc + '/index.html'
                    with tag('li'):
                        with tag("a", href=tmp_url):
                            text(state)
    f.write(doc.getvalue())
