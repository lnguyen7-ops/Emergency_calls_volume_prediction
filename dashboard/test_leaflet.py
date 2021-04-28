import json
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_leaflet as dl
from dash.dependencies import Output, Input
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire

import vaex

app = dash.Dash()


# Import data using vaex
vaex_df = vaex.open('../data/processed/911_Calls_for_dashboard.csv.hdf5')
heatmap_limits_initial = [[-83.305680, -82.896303], [42.239087, 42.471065]]
xarray = vaex_df.count(binby=[vaex_df['longitude'], vaex_df['latitude']],
            selection=None,
            limits=heatmap_limits_initial, # coordinates of view
            shape=256, # Number of pixel (i.e. aggregate groups) in view.
            array_type='xarray') # Return xarray for easier plotting.
xarray = xarray.T.astype('uint32')
img = tf.shade(xarray, cmap=fire, how='eq_hist', alpha=255, min_alpha=40)[::-1].to_pil()

left = xarray['longitude'].min().values
right = xarray['longitude'].max().values
bottom = xarray['latitude'].min().values
top = xarray['latitude'].max().values

image_bounds = [[bottom, left],
				[top, right]]

coordinates = [[left, bottom],
               [right, bottom],
                [right, top],
                [left, top]]



traces = []
# mapbox
mapbox = go.Scattermapbox()
traces.append(mapbox)

# center of map (Detroit)
lat_c = 42.355753
lng_c = -83.076760
# layout
margin = go.layout.Margin(l=0, r=0, b=0, t=0)
layout = go.Layout(#height=height,
					autosize=True,
                   font={'size': 14},
                   margin=margin,
                   mapbox_style="carto-darkmatter",
                   mapbox_layers=[{
                       "sourcetype": "image",
                       "source": img,
                       "coordinates": coordinates
                   }],
                   mapbox_center={"lat": lat_c, "lon": lng_c},
                   mapbox_zoom=10
                  )
fig = go.Figure(data=traces, layout=layout)



app.layout = html.Div([
	dl.Map(dl.LayerGroup(dcc.Graph(id='calls_map', figure=fig, responsive=True)), id="map", center=[lat_c, lng_c], style={'width': '100%', 'height': '50vh'}),
	html.Div(id="log")
])

# @app.callback(
# 	Output('calls_map', 'figure')
# 	)
# def create_map():
# 	traces = []
# 	# mapbox
# 	mapbox = go.Scattermapbox()
# 	traces.append(mapbox)

# 	# center of map (Detroit)
# 	lat = 42.355753
# 	lng = -83.076760
# 	# layout
# 	margin = go.layout.Margin(l=0, r=0, b=0, t=0)
# 	layout = go.Layout(#height=height,
# 	                   font={'size': 14},
# 	                   #margin=margin,
# 	                   mapbox_style="carto-darkmatter",
# 	                   mapbox_layers=[{
# 	                       "sourcetype": "image",
# 	                       "source": img,
# 	                       "coordinates": coordinates
# 	                   }],
# 	                   mapbox_center={"lat": lat_c, "lon": lng_c},
# 	                   mapbox_zoom=10
# 	                  )
# 	fig = go.Figure(data=traces, layout=layout)
# 	return fig

@app.callback(Output("log", "children"), [Input("map", "bounds")])
def log_bounds(bounds):
	print(bounds)
	return json.dumps(bounds)


if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8077, debug='True')