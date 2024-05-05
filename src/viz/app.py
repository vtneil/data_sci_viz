import itertools
import pandas as pd
import pydeck as pdk
import dash
import dash_deck
import dash_bootstrap_components as dbc

from dash import html, dcc, Input, Output
from pydeck.types import String as PdkString

from src.viz.deps.types import Paper, PaperFactory

# App attributes
APP_NAME = 'SCOPUS Visualization'
APP_TITLE = 'SCOPUS Visualization'
USE_DEBUG = True

# API key
MAPBOX_API_KEY = 'pk.eyJ1IjoibmVpbDQ4ODQiLCJhIjoiY2txZmZqbXk3MXR4aTJzcXRtanZvbmRhYSJ9.XSlym9F6sbOtwX4P2r-vrw'

# Colors
GREEN_RGB = [0, 255, 0, 40]
RED_RGB = [240, 100, 0, 40]


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
        self.__setup_data(path)
        print('Setting up components...')
        self.__setup_components()
        print('Setting up Dash...')
        self.__setup_dash()
        print('Setting up callbacks...')
        self.__setup_callbacks()
        print('App is set up!')

    def __setup_data(self, path: str) -> None:
        self.raw_data = PaperFactory.many_from_json(path, 10)

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
                (
                    *row_data,
                    aff['country'],
                    float(aff['location']['lat']),
                    float(aff['location']['lng']),
                    1
                )
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
                    (
                        year,
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
                    )
                )

        self.df_flatten = pd.DataFrame(
            flatten_rows,
            columns=row_titles
        ).set_index('id')

        self.df_heatmap = pd.DataFrame(
            heatmap_rows,
            columns=[*row_titles, 'country', 'lat', 'lng', 'weight']
        ).set_index('id')

        self.df_arc = pd.DataFrame(
            arc_rows,
            columns=['year',
                     'name1', 'city1', 'country1', 'lat1', 'lng1',
                     'name2', 'city2', 'country2', 'lat2', 'lng2'
                     ]
        )

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
            'map_arc': pdk.Layer(
                'ArcLayer',
                data=self.df_arc,
                get_source_position=["lng1", "lat1"],
                get_target_position=["lng2", "lat2"],
                get_width='10',
                get_tilt=0,
                get_source_color=RED_RGB,
                get_target_color=GREEN_RGB,
                pickable=True,
                auto_highlight=True
            )
        }

        self.init_enable_layers = ['map_heatmap_2d']

    def __setup_dash(self) -> None:
        a, b = self.df_flatten['year'].min(), self.df_flatten['year'].max()
        row_height = '720px'

        def map_container():
            return html.Div(
                dash_deck.DeckGL(
                    id='map',
                    data=self.render_map(self.init_enable_layers, (a, b), []),
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
                    options=self.df_heatmap['country'].unique(),
                    multi=True
                )
            )

        self.app.layout = html.Div(
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
                            dbc.CardHeader(html.Strong('Data and Layers')),
                            dbc.CardBody(
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
                            ),
                        ]), width=2,
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Map Visualization')),
                            dbc.CardBody(map_container()),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Free Panel')),
                            dbc.CardBody(data_container()),
                        ]),
                        width=3
                    ),
                ]),
                html.Br(),

                # Row 2:
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Free Panel')),
                            dbc.CardBody(),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Free Panel')),
                            dbc.CardBody(),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Free Panel')),
                            dbc.CardBody(),
                        ]),
                    ),
                ]),
                html.Br(),

                # Row 3:
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Free Panel')),
                            dbc.CardBody(),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Free Panel')),
                            dbc.CardBody(),
                        ]),
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Free Panel')),
                            dbc.CardBody(),
                        ]),
                    ),
                ]),
                html.Br(),
            ], fluid=True, style={
                'width': '100%'
            }), style={
                'padding': '36px',
                'font-size': '20px'
            }
        )

    def __setup_callbacks(self) -> None:
        @self.app.callback(
            Output(component_id='map',
                   component_property='data'),
            Input(component_id='map-selector',
                  component_property='value'),
            Input(component_id='year-selector',
                  component_property='value'),
            Input(component_id='country-selector',
                  component_property='value')
        )
        def update_map(selected_layers: list[str],
                       year_range: tuple[int, int],
                       countries: list[str]):
            print('Updating map...')
            m = self.render_map(selected_layers, year_range, countries)
            print('Map updated!')
            return m

    def render_map(self, layers: list[str],
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
            layers=list(self.map_layers.values()),
            initial_view_state=self.view_state,
            map_style='light'
        )

        return r.to_json()

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
