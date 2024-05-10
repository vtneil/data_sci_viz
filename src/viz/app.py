import sys

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objs as go
import dash
import dash_deck
import dash_bootstrap_components as dbc
import networkx as nx

from dash import html, dcc, Input, Output, ctx
from pydeck.types import String as PdkString

from viz.deps.types import Paper, PaperFactory
from viz.deps.utils import benchmark

# App attributes
APP_NAME = 'Scopus Visualization'
APP_TITLE = 'Scopus Visualization (Chulalongkorn University)'
USE_DEBUG = False

# API key
MAPBOX_API_KEY = 'pk.eyJ1IjoibmVpbDQ4ODQiLCJhIjoiY2txZmZqbXk3MXR4aTJzcXRtanZvbmRhYSJ9.XSlym9F6sbOtwX4P2r-vrw'

# Colors
GREEN_RGB = [0, 240, 100, 40]
RED_RGB = [240, 100, 0, 40]

SIDEBAR_SIZE = 20  # rem

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': f'{SIDEBAR_SIZE}rem',
    'padding': '2rem 1rem',
    'background-color': '#f8f9fa',
}

CONTENT_STYLE = {
    'margin-left': f'{SIDEBAR_SIZE + 2}rem',
}


class App:
    app = dash.Dash(
        APP_NAME,
        meta_tags=[{'name': 'viewport',
                    'content': 'width=device-width, initial-scale=1'}],
        prevent_initial_callbacks=True,
        suppress_callback_exceptions=True,
        title=APP_TITLE,
        update_title=None,
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )

    def __init__(self, path: str) -> None:
        self.first_time = True

        print('Setting up data...')
        self.raw_data = PaperFactory.many_from_json(path)
        self.__setup_data()
        self.__setup_network()
        print('Setting up components...')
        self.__setup_components()
        print('Setting up Dash...')
        self.__setup_dash()
        print('Setting up callbacks...')
        self.__setup_callbacks()
        print('App is set up!')

    @benchmark
    def __setup_data(self) -> None:
        categories = set()
        flatten_rows = []
        heatmap_rows = []
        author_rows = []

        row_titles = ['id', 'title', 'year']

        for paper_entry in self.raw_data:
            sid = paper_entry['SCOPUSID']
            title = paper_entry['title']
            year = int(paper_entry['publish_year'])
            affiliations = paper_entry['affiliations']
            row_data = (sid, title, year)
            cat = paper_entry['abbrevs']

            # Category
            if not cat or not paper_entry['cited_by']:
                continue  # todo

            categories.update(cat)

            # Flatten data
            flatten_rows.append([
                *row_data,
                int(paper_entry['cited_by']),
                tuple(cat),
            ])

            # Heatmap data
            heatmap_rows.extend([
                [
                    *row_data,
                    aff['name'],
                    aff['city'],
                    aff['country'],
                    float(aff['location']['lat']),
                    float(aff['location']['lng']),
                    1,
                    tuple(paper_entry['abbrevs']),
                    tuple(paper_entry['authors'])
                ]
                for aff in affiliations
            ])

            num_coauthor = len(paper_entry['authors'])

            author_rows.extend([
                [
                    *row_data,
                    tuple(paper_entry['abbrevs']),
                    auth,
                    num_coauthor
                ]
                for auth in paper_entry['authors']
            ])

        self.df_flatten = pd.DataFrame(
            flatten_rows,
            columns=[*row_titles, 'cited', 'categories']
        ).dropna().set_index('id')

        self.df_heatmap = pd.DataFrame(
            heatmap_rows,
            columns=[*row_titles, 'name', 'city', 'country',
                     'lat', 'lng', 'weight', 'categories', 'authors']
        ).dropna()

        self.df_author = pd.DataFrame(
            author_rows,
            columns=[*row_titles, 'categories', 'author', 'num_authors']
        ).dropna()

        # Arc data
        num_head = 5
        grouped: pd.DataFrame = self.df_heatmap.groupby(['year', 'name']).size().reset_index(name='count')
        largest: pd.DataFrame = (
            grouped
            .sort_values(['count'], ascending=False)
            .groupby('year')
            .head(num_head)
            .reset_index(drop=True)
        )
        self.df_top = largest.merge(
            self.df_heatmap[['name', 'city', 'country', 'lat', 'lng']].drop_duplicates(subset='name', keep='first'),
            on='name', how='left'
        )

        top_affiliations = set(self.df_top['name'])
        arc_rows = {}
        arc_lookup: set[tuple[str, str, int]] = set()

        for paper_entry in self.raw_data:
            sid = paper_entry['SCOPUSID']
            title = paper_entry['title']
            year = int(paper_entry['publish_year'])
            affiliations = paper_entry['affiliations']
            row_data = (sid, title, year)

            # Arc data
            for i in range(len(affiliations)):
                aff1 = affiliations[i]
                if aff1['name'] not in top_affiliations:
                    continue

                for j in range(i, len(affiliations)):
                    aff2 = affiliations[j]
                    if aff2['name'] not in top_affiliations:
                        continue

                    if aff1['name'] == aff2['name']:
                        continue

                    loc_f = (aff1['name'], aff2['name'], year)
                    loc_b = (aff2['name'], aff1['name'], year)

                    if loc_f not in arc_lookup:
                        arc_lookup.add(loc_f)
                        arc_lookup.add(loc_b)

                        d = [
                            *row_data,
                            aff1['name'],
                            aff1['city'],
                            aff1['country'],
                            float(aff1['location']['lat']),
                            float(aff1['location']['lng']),
                            aff2['name'],
                            aff2['city'],
                            aff2['country'],
                            float(aff2['location']['lat']),
                            float(aff2['location']['lng']),
                            tuple(paper_entry['abbrevs']),
                            0
                        ]

                        arc_rows[loc_f] = d
                        arc_rows[loc_b] = d

                    if arc_rows[loc_f][-1] < 10:
                        arc_rows[loc_f][-1] += 1

        self.df_arc = pd.DataFrame(
            arc_rows.values(),
            columns=[*row_titles,
                     'name1', 'city1', 'country1', 'lat1', 'lng1',
                     'name2', 'city2', 'country2', 'lat2', 'lng2',
                     'categories', 'weight'
                     ]
        ).dropna()

        self.categories: list[str] = list(categories)

    @benchmark
    def __setup_network(self) -> None:
        coauthor_graph = nx.Graph()
        institution_graph = nx.Graph()
        country_graph = nx.Graph()

        self.graphs = {
            'coauthor': coauthor_graph,
            'institution': institution_graph,
            'country': country_graph
        }

    @benchmark
    def __setup_components(self) -> None:
        self.view_state = pdk.ViewState(
            longitude=100.532300,
            latitude=13.736567,
            zoom=6,
            pitch=40.5)

        self.map_layers: dict[str, pdk.Layer] = {
            'map_heatmap_2d': pdk.Layer(
                'HeatmapLayer',
                data=self.df_heatmap[['lat', 'lng', 'weight']],
                opacity=0.9,
                get_position='[lng, lat]',
                get_weight='weight',
                aggregation=PdkString('MEAN')
            ),
            'map_arc': pdk.Layer(
                'ArcLayer',
                data=self.df_arc[['lat1', 'lng1', 'lat2', 'lng2', 'weight']],
                get_source_position=['lng1', 'lat1'],
                get_target_position=['lng2', 'lat2'],
                get_width='weight',
                get_tilt=0,
                get_source_color=RED_RGB,
                get_target_color=GREEN_RGB,
                pickable=True,
                auto_highlight=True
            )
        }

        self.init_enable_layers = ['map_heatmap_2d', 'map_arc']

    @benchmark
    def __setup_dash(self) -> None:
        a, b = self.df_flatten['year'].min(), self.df_flatten['year'].max()
        countries = sorted(self.df_heatmap['country'].unique())
        row_height = '720px'
        row2_height = '600px'

        def map_container():
            return html.Div(
                dash_deck.DeckGL(
                    id='map',
                    data=self.get_render_map(self.init_enable_layers, (a, b), [], []),
                    mapboxKey=MAPBOX_API_KEY
                ),
                style={
                    'height': row_height,
                    'width': '100%',
                    'position': 'relative'
                }
            )

        def layer_selector_container():
            return html.Div(
                dbc.Checklist(
                    id='map-selector',
                    options=[
                        {'label': 'Heatmap', 'value': 'map_heatmap_2d'},
                        {'label': 'Arc Plot', 'value': 'map_arc'}
                    ],
                    value=self.init_enable_layers
                )
            )

        def year_selector_container():
            return html.Div(
                dcc.RangeSlider(
                    id='year-selector',
                    min=a,
                    max=b,
                    step=None,
                    marks={
                        v: f'{v}'
                        for v in list(range(a, b + 1))
                    },
                    value=[a, b]
                )
            )

        def country_selector_container():
            return html.Div(
                dcc.Dropdown(
                    id='country-selector',
                    placeholder='(All)',
                    options=countries,
                    multi=True
                )
            )

        def category_selector_container():
            return html.Div(
                dcc.Dropdown(
                    id='category-selector',
                    placeholder='(All)',
                    options=self.categories,
                    multi=True
                )
            )

        def num_collab_year_container():
            return html.Div(
                dcc.Graph(id='collaborations-year',
                          figure=self.get_collaborations_year([], []),
                          style={
                              'height': '100%'
                          }),
                style={
                    'height': row2_height
                }
            )

        def num_pub_year_container():
            return html.Div(
                dcc.Graph(id='publications-year',
                          figure=self.get_publications_year([], []),
                          style={
                              'height': '100%'
                          }),
                style={
                    'height': row2_height
                }
            )

        def sunburst_container():
            return html.Div(
                dcc.Graph(id='publications-sunburst',
                          figure=self.get_sunburst((a, b), [], []),
                          style={
                              'height': '100%'
                          }),
                style={
                    'height': row_height
                }
            )

        def network_country_container():
            return html.Div(
                dcc.Graph(id='network-country',
                          figure=self.get_network_graph(
                              self.update_country_network((a, b), [], [])
                          ),
                          style={
                              'height': '100%'
                          }, config={'scrollZoom': True}),
                style={
                    'height': row2_height
                }
            )

        def histogram_container():
            return html.Div(
                dcc.Graph(id='citation-histogram',
                          figure=self.get_cite_hist((a, b), []),
                          style={
                              'height': '100%'
                          }),
                style={
                    'height': row_height
                }
            )

        def heatmap_container():
            return html.Div(
                dcc.Graph(id='author-heatmap',
                          figure=self.get_author_heat((a, b), []),
                          style={
                              'height': '100%'
                          }),
                style={
                    'height': row_height
                }
            )

        sidebar = html.Div(
            [
                html.H2('Data and Layers'),
                html.Hr(),
                html.Div([
                    html.Strong('Layer selector'),
                    layer_selector_container(),
                    html.Br(),
                    html.Strong('Filter by year'),
                    year_selector_container(),
                    html.Br(),
                    html.Strong('Filter by countries'),
                    country_selector_container(),
                    html.Br(),
                    html.Strong('Filter by categories'),
                    category_selector_container(),
                ])
            ],
            style=SIDEBAR_STYLE,
        )

        content = html.Div(
            dbc.Container([
                # Title
                dbc.Row([
                    html.Center(html.H1(html.Strong(APP_TITLE))),
                ]),
                html.Br(),

                # Row 1: Selector/filter, Map, Free Panel
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Map Visualization')),
                            dbc.CardBody([
                                html.P('Heatmap: by filter, Arc Plot: top and filtered top'),
                                html.P('Red: source, Green: target'),
                                dcc.Loading(map_container())
                            ]),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Number of publications by institution')),
                            dbc.CardBody([
                                html.P('Shows only top if country filter is not selected'),
                                html.P('This is a sunburst graph.'),
                                dcc.Loading(sunburst_container())
                            ]),
                        ]),
                        width=4
                    ),
                ]),
                html.Br(),

                # Row 2:
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Number of publications over time')),
                            dbc.CardBody([
                                html.P('Shows only top if country filter is not selected'),
                                dcc.Loading(num_pub_year_container())
                            ]),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Number of international collaborations over time')),
                            dbc.CardBody([
                                html.P('Always shows only top'),
                                dcc.Loading(num_collab_year_container())
                            ]),
                        ]),
                    ),
                ]),
                html.Br(),

                # Row 3:
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Country network')),
                            dbc.CardBody([
                                html.P('Shows only top if country filter is not selected'),
                                dcc.Loading(network_country_container())
                            ]),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Citation distribution')),
                            dbc.CardBody([
                                html.P('Showing all by default.'),
                                dcc.Loading(histogram_container())
                            ]),
                        ]),
                    ),
                ]),
                html.Br(),
                # Row 3:
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Author publications distribution')),
                            dbc.CardBody([
                                dcc.Loading(heatmap_container())
                            ]),
                        ]),
                    ),
                ]),
                html.Br(),
            ], fluid=True, style={
                'width': '100%'
            }), style={
                **CONTENT_STYLE,
                'padding': '36px',
                'font-size': '20px'
            }
        )

        self.sidebar = sidebar
        self.content = content

        self.app.layout = html.Div(
            id='main-content',
            children=[
                self.sidebar,
                self.content
            ]
        )

    @benchmark
    def __setup_callbacks(self) -> None:
        @self.app.callback(
            Output(component_id='map',
                   component_property='data'),
            Output(component_id='collaborations-year',
                   component_property='figure'),
            Output(component_id='publications-year',
                   component_property='figure'),
            Output(component_id='publications-sunburst',
                   component_property='figure'),
            Output(component_id='network-country',
                   component_property='figure'),
            Output(component_id='citation-histogram',
                   component_property='figure'),
            Output(component_id='author-heatmap',
                   component_property='figure'),
            Input(component_id='map-selector',
                  component_property='value'),
            Input(component_id='year-selector',
                  component_property='value'),
            Input(component_id='country-selector',
                  component_property='value'),
            Input(component_id='category-selector',
                  component_property='value')
        )
        def update_data(selected_layers: list[str],
                        year_range: tuple[int, int],
                        countries: list[str],
                        categories: list[str]):
            print('Updating data...')

            self.map_rendered = self.get_render_map(selected_layers, year_range, countries, categories)

            if self.first_time:
                self.collaborations_year = self.get_collaborations_year(countries, categories)
                self.publications_year = self.get_publications_year(countries, categories)
                self.sunburst = self.get_sunburst(year_range, countries, categories)
                self.network = self.get_network_graph(self.update_country_network(year_range, countries, categories))
                self.citation = self.get_cite_hist(year_range, categories)
                self.author_hm = self.get_author_heat(year_range, categories)

                self.first_time = False
            else:
                trigger = ctx.triggered_id
                if trigger in ['counter-selector', 'category-selector']:
                    self.collaborations_year = self.get_collaborations_year(countries, categories)
                    self.publications_year = self.get_publications_year(countries, categories)
                if trigger in ['year-selector', 'counter-selector', 'category-selector']:
                    self.sunburst = self.get_sunburst(year_range, countries, categories)
                    self.network = self.get_network_graph(
                        self.update_country_network(year_range, countries, categories)
                    )
                if trigger in ['year-selector', 'category-selector']:
                    self.citation = self.get_cite_hist(year_range, categories)
                    self.author_hm = self.get_author_heat(year_range, categories)

            print('Data updated!')

            return (self.map_rendered, self.collaborations_year, self.publications_year,
                    self.sunburst, self.network, self.citation, self.author_hm)

    def get_render_map(self, layers: list[str],
                       year_range: tuple[int, int],
                       countries: list[str],
                       categories: list[str]) -> str:
        for name, layer in self.map_layers.items():
            if name in layers:
                layer.visible = True
            else:
                layer.visible = False

        # Heatmap
        filter_pos = self.df_heatmap['year'].between(year_range[0], year_range[1])
        if countries:
            filter_pos = filter_pos & self.df_heatmap['country'].isin(countries)
        if categories:
            filter_pos = filter_pos & self.df_heatmap['categories'].apply(
                lambda x: any(item in categories for item in x)
            )
        self.map_layers['map_heatmap_2d'].data = self.df_heatmap[filter_pos][['lat', 'lng', 'weight']]

        # Arc Plot
        filter_pos = self.df_arc['year'].between(year_range[0], year_range[1])
        if countries:
            filter_pos = filter_pos & (
                    self.df_arc['country1'].isin(countries) | self.df_arc['country2'].isin(countries)
            )
        if categories:
            filter_pos = filter_pos & self.df_arc['categories'].apply(
                lambda x: any(item in categories for item in x)
            )
        self.map_layers['map_arc'].data = self.df_arc[filter_pos][['lat1', 'lng1', 'lat2', 'lng2', 'weight']]

        r = pdk.Deck(
            layers=list(self.map_layers.values()),
            initial_view_state=self.view_state,
            map_style='light'
        )

        return r.to_json()

    def get_collaborations_year(self, _: list[str],
                                categories: list[str]):
        if categories:
            filtered_df = self.df_arc[self.df_arc['categories'].apply(
                lambda x: any(item in categories for item in x)
            )]
        else:
            filtered_df = self.df_arc

        filtered_df = filtered_df.melt(
            id_vars=['year', 'id'],
            value_vars=['country1', 'country2'],
            var_name='ct', value_name='country'
        ).dropna().drop(columns=['ct'])

        filtered_df = filtered_df.groupby('id').apply(
            lambda x: x.drop_duplicates(subset=['country']),
            include_groups=False
        ).reset_index(drop=True)

        df_count = filtered_df.groupby(['year', 'country']).size().reset_index(name='collaborations')

        fig = px.area(df_count, x='year', y='collaborations',
                      color='country', line_group='country')
        fig.update_layout(margin=dict(b=5, l=5, r=5, t=5))

        return fig

    def get_publications_year(self, countries: list[str],
                              categories: list[str]):
        if countries:
            filtered_df = self.df_heatmap[self.df_heatmap['country'].isin(countries)]
        else:
            filtered_df = self.df_heatmap[self.df_heatmap['country'].isin(self.df_top['country'])]

        if categories:
            filtered_df = filtered_df[filtered_df['categories'].apply(
                lambda x: any(item in categories for item in x)
            )]

        filtered_df = filtered_df[['id', 'year', 'country']]

        filtered_df = filtered_df.groupby('id').apply(
            lambda x: x.drop_duplicates(subset=['country']),
            include_groups=False
        ).reset_index(drop=True)

        df_count = filtered_df.groupby(['year', 'country']).size().reset_index(name='publications')
        fig = px.area(df_count, x='year', y='publications',
                      color='country', line_group='country')
        fig.update_layout(margin=dict(b=5, l=5, r=5, t=5))

        return fig

    def get_sunburst(self, year_range: tuple[int, int],
                     countries: list[str],
                     categories: list[str]):
        filter_pos = self.df_heatmap['year'].between(year_range[0], year_range[1])

        if countries:
            filter_pos = filter_pos & self.df_heatmap['country'].isin(countries)
        else:
            filter_pos = filter_pos & self.df_heatmap['country'].isin(self.df_top['country'])

        if categories:
            filter_pos = filter_pos & self.df_heatmap['categories'].apply(
                lambda x: any(item in categories for item in x)
            )

        filtered_df = self.df_heatmap[filter_pos]

        filtered_df = filtered_df[['id', 'year', 'name', 'city', 'country']]
        df_count = filtered_df.groupby(['country', 'city', 'name']).size().reset_index(name='publications')

        fig = px.sunburst(df_count, path=['country', 'city', 'name'], values='publications')
        fig.update_layout(margin=dict(b=5, l=5, r=5, t=5))

        return fig

    def update_country_network(self, year_range: tuple[int, int],
                               countries: list[str],
                               categories: list[str]) -> nx.Graph:
        if not countries:
            countries = set(self.df_top['country'])

        self.graphs['country'] = nx.Graph()

        for (k, paper_entry) in enumerate(self.raw_data):
            paper_entry: Paper
            if not (year_range[0] <= int(paper_entry['publish_year']) <= year_range[1]):
                continue

            if not paper_entry['abbrevs']:
                continue

            aff_countries = list(set([aff['country'] for aff in paper_entry['affiliations']]))

            for i in range(len(aff_countries)):
                for j in range(i + 1, len(aff_countries)):
                    if all([aff_countries[i],
                            aff_countries[j],
                            aff_countries[i] in countries or aff_countries[j] in countries,
                            any(item in categories if categories else True for item in paper_entry['abbrevs'])]):
                        self.graphs['country'].add_edge(aff_countries[i], aff_countries[j])

        return self.graphs['country']

    @staticmethod
    def get_network_graph(graph: nx.Graph):
        pos = nx.kamada_kawai_layout(graph)
        edge_x = []
        edge_y = []

        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(f'{adjacencies[0]} has {len(adjacencies[1])} connections.')

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=5, l=5, r=5, t=5),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

    def get_author_heat(self, year_range: tuple[int, int],
                        categories: list[str]):

        filter_pos = self.df_author['year'].between(year_range[0], year_range[1])

        if categories:
            filter_pos = filter_pos & self.df_author['categories'].apply(
                lambda x: any(item in categories for item in x)
            )

        filtered_df = self.df_author[filter_pos].reset_index(drop=True)

        paper_count = filtered_df['author'].value_counts().reset_index()
        paper_count.columns = ['author', 'publication_count']

        grouped = filtered_df.groupby(['author', 'id']).first().reset_index()
        avg_authors = grouped.groupby('author')['num_authors'].mean().reset_index()
        avg_authors.columns = ['author', 'avg_num_authors']

        author_info: pd.DataFrame = avg_authors.merge(paper_count, on='author').drop_duplicates()
        author_info['publication_count'] = np.log10(author_info['publication_count'])
        author_info['avg_num_authors'] = np.log10(author_info['avg_num_authors'])

        fig = px.density_heatmap(author_info,
                                 x='publication_count',
                                 y='avg_num_authors',
                                 marginal_x='histogram',
                                 marginal_y='histogram',
                                 nbinsx=20,
                                 nbinsy=40)
        fig.update_layout(xaxis_title='Publication Count per Author (log)',
                          yaxis_title='Average Number of Authors per Paper Involved (log)',
                          margin=dict(b=5, l=5, r=5, t=5))

        return fig

    def get_cite_hist(self, year_range: tuple[int, int],
                      categories: list[str]):

        filter_pos = self.df_flatten['year'].between(year_range[0], year_range[1])

        if categories:
            filter_pos = filter_pos & self.df_flatten['categories'].apply(
                lambda x: any(item in categories for item in x)
            )

        filtered_df = self.df_flatten[filter_pos].reset_index(drop=True)

        fig = px.histogram(filtered_df,
                           x='cited',
                           log_y=True,
                           nbins=60)
        fig.update_layout(xaxis_title='Cited count',
                          margin=dict(b=5, l=5, r=5, t=5))

        return fig

    def start(self) -> None:
        self.app.run(host='localhost',
                     port=8050,
                     debug=USE_DEBUG,
                     use_reloader=USE_DEBUG)

    def stop(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == '__main__':
    with App('./data/final_paper_format.json') as app:
        app.start()
