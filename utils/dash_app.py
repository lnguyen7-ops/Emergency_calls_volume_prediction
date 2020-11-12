import pandas as pd
import json
import plotly.graph_objs as go 

import dash
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output

# to be used with date_picker
from datetime import datetime as dt

#app = dash.Dash(__name__)
app = dash.Dash("utils")
#app.css.append_css({"external_url": "/static/reset.css"})
#app.server.static_folder = "static"
#dcc._css_dist[0]['relative_package_path'].append('static/reset.css')

# ---------------------------------------------
# Read in csv data file
df = pd.read_csv("../data/forecast.csv")
# Get list of FIDs (maintain order same as df)
FIDs = [text.split(",")[0] for text in df.columns[:-2]] # skip last 2 columns
# Get list of neighborhoood names (maintain order same as df)
nhood_names = [text.split(",")[1] for text in df.columns[:-2]] # skip last 2 columns
# Read in geo-json file
# read in Detroit neighborhood geojson.
with open("../data/detroit_geo.json") as json_file:
    detroit_geo = json.load(json_file)

#-------------------------------------------------
# DASH APP
# Background color
colors = {
    'background': '#006e99',
    'text': '#7FDBFF'
}
# App layout (Dash components all go inside layout)
app.layout = html.Div(
	#style={"backgroundColor": colors["background"], "pad": 0},
	children = [
		# Title of the web-app
		html.H1("Detroit City", style={"text-align": "center"}),
		html.H2("Police Attention Level Needed", style={"text-align": "center"}),

		# Date picker
		html.Div(
			[dcc.DatePickerSingle(id="dt_pick_single", 
								date=dt(2020,11,13),
								min_date_allowed=dt(2020,11,7),
								max_date_allowed=dt(2020,11,13))],
			style={"padding-left": "25%"}
		),

		# Element to show selected date as text
		html.H2(html.Div(id="output_container", children=[], style={"padding-left":"25%"})),
		html.Br(),

		# Choropleth map
		html.Div(
			[dcc.Graph(id="att_level_map", figure={})],
			style={"padding-left":"25%", "padding-right": "25%"})
	])

#------------------------------------------------
# Connect Plotly graphs with Dash components

@app.callback(
	[Output(component_id="output_container", component_property="children"),
	Output(component_id="att_level_map", component_property="figure")],
	[Input(component_id="dt_pick_single", component_property="date")]
	)
def update_graph(date_picked):
	print(date_picked)
	print(type(date_picked))
	# Print Date selected
	container = f"Date selected: {date_picked[:10]}"

	# Because date_picked is of type str, need to conver to int
	date_picked = int(date_picked[8:10])

	# FIGURE 
	# center of map
	lat = 42.355753
	lng = -83.076760
	# Filter data for selected date
	data = df[df["Day"]==date_picked]
	data = data.iloc[:,:-2].to_numpy() # DONT keep the Day and Hour columns
	# colorscale min and max
	thres_min = data.min()
	thres_max = data.max()
	# time of day
	times = [f"<b>Hour<br>{i}:00</b>" for i in range(0, 24, 3)]
	times_slider = [f"<b>Hour {i}:00</b>" for i in range(0, 24, 3)]
	fig = go.Figure()
	# add choropleth traces to figure
	for i in range(data.shape[0]):
	    dff = pd.DataFrame({"FID": FIDs,
	                   "Neighborhood": nhood_names,
	                    "att_level": data[i,:]})
	    
	    fig.add_trace(go.Choroplethmapbox(geojson=detroit_geo,
	                                       locations=dff["FID"],
	                                       z=dff["att_level"],
	                                       featureidkey="properties.FID",
	                                       zmin=thres_min,
	                                       zmax=thres_max,
	                                       text=dff["Neighborhood"],
	                                       hovertemplate="<b>%{text}</b><br>Attention level: %{z:.2f}",
	                                       name=times[i],
	                                       marker_opacity=1))
	
	active = 0 # initial active trace
	for i in range(len(fig.data)):
		fig.data[i].visible = False
	fig.data[active].visible = True # set the active trace as visible.
	# create slider
	steps = []
	for i in range(len(fig.data)):
	    step = {"method": "update",
	            "label": times_slider[i],
	            "args": [{"visible": [True if j==i else False for j in range(len(fig.data))]}]# set i'th trace to visible 
	           }
	    steps.append(step)
	sliders = [{"active": active,
	            "currentvalue": {"prefix": ""},
	            "transition": {"duration": 100},
	            "pad": {"t": 5, "l":25, "b":10},
	            "steps": steps}]

	fig.update_layout(mapbox_style="open-street-map",
	                  mapbox_zoom=10,
	                  mapbox_center={"lat": lat, "lon": lng},
	                  sliders=sliders)
	fig.update_layout(autosize=True,
						height=700,
						margin={"r":0, "t":0, "l":0, "b":0},
	                  font={"size": 15})
	return container, fig

# ------------------------------------------------
if __name__ == "__main__":
	app.run_server(debug=True)