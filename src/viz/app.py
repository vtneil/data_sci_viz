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
        external_stylesheets=[dbc.themes.DARKLY]
    )

    def __init__(self, path: str) -> None:
        self.__setup_data(path)
        self.__setup_components()
        self.__setup_dash()
        self.__setup_callbacks()

    def __setup_data(self, path: str) -> None:
        self.raw_data = PaperFactory.many_from_json(path)

        heatmap_rows = []
        arc_rows = []

        for sid, data in self.raw_data.items():
            affiliations = data['affiliations']

            heatmap_rows.extend([(
                float(aff['location']['lat']),
                float(aff['location']['lng']),
                1
            ) for aff in affiliations])

            arc_rows.extend([(
                float(aff1['location']['lat']),
                float(aff1['location']['lng']),
                float(aff2['location']['lat']),
                float(aff2['location']['lng']),
            ) for (aff1, aff2) in itertools.combinations(affiliations, 2)])

        self.df_heatmap = pd.DataFrame(heatmap_rows, columns=['lat', 'lng', 'weight'])
        self.df_arc = pd.DataFrame(arc_rows, columns=['lat1', 'lng1', 'lat2', 'lng2'])

    def __setup_components(self) -> None:
        self.view_state = pdk.ViewState(
            longitude=100.532300,
            latitude=13.736567,
            zoom=6,
            min_zoom=2,
            max_zoom=15,
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
        row_height = '720px'

        def map_container():
            return html.Div(
                dash_deck.DeckGL(
                    id='map',
                    data=self.render_map(self.init_enable_layers),
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
                ),
                style={
                    'height': row_height
                }
            )

        def year_selector_container():
            a, b = 2014, 2024
            return html.Div(
                dcc.RangeSlider(
                    min=a,
                    max=b,
                    step=None,
                    marks={
                        v: f'{v}'
                        for v in list(range(a, b + 1))
                    },
                    value=[2019, 2023]
                )
            )

        self.app.layout = html.Div(
            dbc.Container([
                dbc.Row([
                    html.Center(html.H1(APP_TITLE)),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Layer Selector')),
                            dbc.CardBody(layer_selector_container()),
                        ]),
                        width=4
                    ),
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Map Visualization')),
                            dbc.CardBody(map_container()),
                        ]),
                    ),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.Strong('Select the data range by year')),
                            dbc.CardBody(year_selector_container()),
                        ]),
                    ),
                ]),
            ]), style={
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
        )
        def update_map(selected_layers: list[str]):
            return self.render_map(selected_layers)

    def render_map(self, layers: list[str]) -> str:
        for name, layer in self.map_layers.items():
            if name in layers:
                layer.visible = True
            else:
                layer.visible = False

        r = pdk.Deck(
            layers=list(self.map_layers.values()),
            initial_view_state=self.view_state
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
    with App('./data/sample.json') as app:
        app.start()
