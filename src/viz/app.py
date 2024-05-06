import itertools
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objs as go
import dash
import dash_deck
import dash_bootstrap_components as dbc
import networkx as nx

from dash import html, dcc, Input, Output
from matplotlib import pyplot as plt
from pydeck.types import String as PdkString

from src.viz.deps.types import Paper, PaperFactory
from src.viz.deps.utils import benchmark

# App attributes
APP_NAME = 'SCOPUS Visualization'
APP_TITLE = 'SCOPUS Visualization'
USE_DEBUG = True

# API key
MAPBOX_API_KEY = 'pk.eyJ1IjoibmVpbDQ4ODQiLCJhIjoiY2txZmZqbXk3MXR4aTJzcXRtanZvbmRhYSJ9.XSlym9F6sbOtwX4P2r-vrw'

# Colors
GREEN_RGB = [0, 255, 0, 40]
RED_RGB = [240, 100, 0, 40]

SIDEBAR_SIZE = 24  # rem

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
        print('Setting up data...')
        self.raw_data = PaperFactory.many_from_json(path, 10_000)
        self.__setup_data(path)
        self.__setup_network()
        print('Setting up components...')
        self.__setup_components()
        print('Setting up Dash...')
        self.__setup_dash()
        print('Setting up callbacks...')
        self.__setup_callbacks()
        print('App is set up!')

    @benchmark
    def __setup_data(self, path: str) -> None:
        flatten_rows = []
        heatmap_rows = []
        arc_rows = []
        arc_lookup: set[tuple[str, str, int]] = set()

        row_titles = ['id', 'title', 'year']

        for paper_entry in self.raw_data:
            sid = paper_entry['SCOPUSID']
            title = paper_entry['title']
            year = int(paper_entry['publish_year'])
            affiliations = paper_entry['affiliations']
            row_data = (sid, title, year)

            # Flatten data
            flatten_rows.append(row_data)

            # Heatmap data
            heatmap_rows.extend([
                [
                    *row_data,
                    aff['name'],
                    aff['city'],
                    aff['country'],
                    float(aff['location']['lat']),
                    float(aff['location']['lng']),
                    1
                ]
                for aff in affiliations
            ])

            # Arc data
            for aff1, aff2 in itertools.combinations(affiliations, 2):
                if aff1['name'] == aff2['name']:
                    continue

                loc_f = (aff1['name'], aff2['name'], year)
                loc_b = (aff2['name'], aff1['name'], year)

                if loc_f in arc_lookup:
                    continue

                arc_lookup.add(loc_f)
                arc_lookup.add(loc_b)

                arc_rows.append(
                    [
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
                    ]
                )

        self.df_flatten = pd.DataFrame(
            flatten_rows,
            columns=row_titles
        ).set_index('id')

        self.df_heatmap = pd.DataFrame(
            heatmap_rows,
            columns=[*row_titles, 'name', 'city', 'country', 'lat', 'lng', 'weight']
        ).dropna()

        self.df_arc = pd.DataFrame(
            arc_rows,
            columns=[*row_titles,
                     'name1', 'city1', 'country1', 'lat1', 'lng1',
                     'name2', 'city2', 'country2', 'lat2', 'lng2'
                     ]
        ).dropna()

    @benchmark
    def __setup_network(self) -> None:
        institution_graph = nx.Graph()
        country_graph = nx.Graph()

        # for (k, paper_entry) in enumerate(self.raw_data):
        #     institutions = [aff['name'] for aff in paper_entry['affiliations']]
        #     for i in range(len(institutions)):
        #         for j in range(i + 1, len(institutions)):
        #             if institutions[i] and institutions[j]:
        #                 institution_graph.add_edge(institutions[i], institutions[j])

        for (k, paper_entry) in enumerate(self.raw_data):
            countries = list(set([aff['country'] for aff in paper_entry['affiliations']]))
            for i in range(len(countries)):
                for j in range(i + 1, len(countries)):
                    if countries[i] and countries[j]:
                        country_graph.add_edge(countries[i], countries[j])

        self.graphs = {
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
                data=self.df_heatmap,
                opacity=0.9,
                get_position='[lng, lat]',
                get_weight='weight',
                aggregation=PdkString('MEAN')
            ),
            'map_arc': pdk.Layer('ArcLayer'),
            'map_arc0': pdk.Layer(
                'ArcLayer',
                data=self.df_arc,
                get_source_position=['lng1', 'lat1'],
                get_target_position=['lng2', 'lat2'],
                get_width='10',
                get_tilt=0,
                get_source_color=RED_RGB,
                get_target_color=GREEN_RGB,
                pickable=True,
                auto_highlight=True
            )
        }

        self.init_enable_layers = ['map_heatmap_2d']

    @benchmark
    def __setup_dash(self) -> None:
        a, b = self.df_flatten['year'].min(), self.df_flatten['year'].max()
        countries = sorted(self.df_heatmap['country'].unique())
        row_height = '720px'

        def map_container():
            return html.Div(
                dash_deck.DeckGL(
                    id='map',
                    data=self.get_render_map(self.init_enable_layers, (a, b), []),
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

        def data_container():
            return html.Div(
                style={
                    'height': row_height
                }
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

        def num_collab_year_container():
            return html.Div(
                dcc.Graph(id='collaborations-year',
                          figure=self.get_collaborations_year([]),
                          style={
                              'height': '100%'
                          }),
                style={
                    'height': row_height
                }
            )

        def num_pub_year_container():
            return html.Div(
                dcc.Graph(id='publications-year',
                          figure=self.get_publications_year([]),
                          style={
                              'height': '100%'
                          }),
                style={
                    'height': row_height
                }
            )

        def sunburst_container():
            return html.Div(
                dcc.Graph(id='publications-sunburst',
                          figure=self.get_sunburst((a, b), []),
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
                          figure=self.get_network_graph(self.graphs['country']),
                          style={
                              'height': '100%'
                          }, config={'scrollZoom': True}),
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
                ], style={
                    'height': row_height
                })
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
                            dbc.CardBody(map_container()),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Number of publications by institution')),
                            dbc.CardBody(sunburst_container()),
                        ]),
                        width=4
                    ),
                ]),
                html.Br(),

                # Row 2:
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Country network')),
                            dbc.CardBody(network_country_container()),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Number of publications over time')),
                            dbc.CardBody(num_pub_year_container()),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Number of international collaborations over time')),
                            dbc.CardBody(num_collab_year_container()),
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

        self.app.layout = html.Div([
            # Sidebar
            sidebar,

            # Content
            content
        ])

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
            Input(component_id='map-selector',
                  component_property='value'),
            Input(component_id='year-selector',
                  component_property='value'),
            Input(component_id='country-selector',
                  component_property='value')
        )
        def update_data(selected_layers: list[str],
                        year_range: tuple[int, int],
                        countries: list[str]):
            print('Updating data...')

            map_rendered = self.get_render_map(selected_layers, year_range, countries)
            collaborations_year = self.get_collaborations_year(countries)
            publications_year = self.get_publications_year(countries)
            sunburst = self.get_sunburst(year_range, countries)

            print('Data updated!')
            return map_rendered, collaborations_year, publications_year, sunburst

    def get_render_map(self, layers: list[str],
                       year_range: tuple[int, int],
                       countries: list[str]) -> str:
        for name, layer in self.map_layers.items():
            if name in layers:
                layer.visible = True
            else:
                layer.visible = False

        filter_pos = self.df_heatmap['year'].between(year_range[0], year_range[1])
        if countries:
            filter_pos = filter_pos & self.df_heatmap['country'].isin(countries)
        self.map_layers['map_heatmap_2d'].data = self.df_heatmap[filter_pos]

        filter_pos = self.df_arc['year'].between(year_range[0], year_range[1])
        if countries:
            filter_pos = filter_pos & (
                    self.df_arc['country1'].isin(countries) | self.df_arc['country2'].isin(countries)
            )
        self.map_layers['map_arc'].data = self.df_arc[filter_pos]

        r = pdk.Deck(
            layers=[self.map_layers[k] for k in self.init_enable_layers],
            # layers=list(self.map_layers.values()), #  todo: FIX!
            initial_view_state=self.view_state,
            map_style='light'
        )

        return r.to_json()

    def get_collaborations_year(self, countries: list[str]):
        if countries:
            filtered_df = self.df_arc[self.df_arc['country1'].isin(countries) | self.df_arc['country2'].isin(countries)]
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
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return fig

    def get_publications_year(self, countries: list[str]):
        if countries:
            filtered_df = self.df_heatmap[self.df_heatmap['country'].isin(countries)]
        else:
            filtered_df = self.df_heatmap

        filtered_df = filtered_df[['id', 'year', 'country']]

        filtered_df = filtered_df.groupby('id').apply(
            lambda x: x.drop_duplicates(subset=['country']),
            include_groups=False
        ).reset_index(drop=True)

        df_count = filtered_df.groupby(['year', 'country']).size().reset_index(name='publications')
        fig = px.area(df_count, x='year', y='publications',
                      color='country', line_group='country')
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return fig

    def get_sunburst(self, year_range: tuple[int, int], countries: list[str]):
        filter_pos = self.df_heatmap['year'].between(year_range[0], year_range[1])
        if countries:
            filter_pos = filter_pos & self.df_heatmap['country'].isin(countries)
        filtered_df = self.df_heatmap[filter_pos]

        filtered_df = filtered_df[['id', 'year', 'name', 'city', 'country']]
        df_count = filtered_df.groupby(['country', 'city', 'name']).size().reset_index(name='publications')

        fig = px.sunburst(df_count, path=['country', 'city', 'name'], values='publications')
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))

        return fig

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
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig

    def start(self) -> None:
        self.app.run(host='localhost',
                     port=8050,
                     debug=USE_DEBUG)

    def stop(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == '__main__':
    with App('./data/papers.json') as app:
        app.start()
