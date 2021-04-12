import pandas as pd
import numpy as np
import pathlib
import json
import plotly.graph_objs as go 

import dash
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output

# to be used with date_picker
from datetime import datetime as dt

from utils import process_data_dashboard

# Using style css sheet by chriddyp
# https://codepen.io/chriddyp/pen/bWLwgP
# This is done by having the css file inside the 'assets' directory.
app = dash.Dash(__name__)

#################################################################
################ Read in csv data file #########################
##################################################################
# 911 call records
# Check if processed data exists
directory = pathlib.Path('../data/processed/')
# define the filename pattern
name_pattern = f"911_Calls_for_dashboard.csv"
if len(directory.glob(name_pattern)) == 0:
	# Select years of data to include
	years = [2021]
	process_data_dashboard(years)
df = pd.read_csv('../data/processed/911_Calls_for_dashboard.csv', 
	thousands=",", dtype={'priority':'str'})


#print(df.info())
#print(df['call_timestamp_EST'].head(5))
# print(df['zip_code'].unique())
# print(df['priority'].unique())

'''
df = pd.read_csv("../data/forecast.csv")
# Get list of FIDs (maintain order same as df)
FIDs = [text.split(",")[0] for text in df.columns[:-2]] # skip last 2 columns
# Get list of neighborhoood names (maintain order same as df)
nhood_names = [text.split(",")[1] for text in df.columns[:-2]] # skip last 2 columns
'''

# Read in geo-json file
# read in Detroit neighborhood geojson.
with open("../data/raw/neighborhoods_info_2021-04-09.geojson") as geojson_file:
    nhoods_geo = json.load(geojson_file)
# read in Detroit city blocks geojson.
with open("../data/raw/city_blocks_info_2021-04-09.geojson") as geojson_file:
    blocks_geo = json.load(geojson_file)

##################################################################
############### DASH APP ###############################
##########################################################
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
		html.H2("911 Emergency Call Records", style={"text-align": "center"}),

		# Date picker
		# html.Div(
		# 	[dcc.DatePickerSingle(id="dt_pick_single", 
		# 						date=dt(2020,11,13),
		# 						min_date_allowed=dt(2020,11,7),
		# 						max_date_allowed=dt(2020,11,13))],
		# 	style={"padding-left": "25%"}
		# ),

		# Element to show selected date as text
		# html.H2(html.Div(id="output_container", children=[], style={"padding-left":"25%"})),
		# html.Br(),

		# GRID
		# Based on external css. Check css doc link above for more info.
		html.Div(className='container',
			children=[
				# Filters
				html.Div(className='two columns',
					children=[
						html.H3('Filters'),
						# Responding agency
						dcc.Dropdown(
							id = 'agency',
							options=[{'label': x, 'value': x} for x in np.sort(df['agency'].unique())],
							multi=True,
							placeholder='Responding agency'),
						# zip_code
						dcc.Dropdown(
							id = 'zip_code',
							options=[{'label': x, 'value': x} for x in np.sort(df['zip_code'].unique())],
							multi=True,
							placeholder='Zip code'),
						# Call priority
						dcc.Dropdown(
							id = 'priority',
							options=[{'label': x, 'value': x} for x in np.sort(df['priority'].unique())],
							multi=True,
							placeholder='Call priority'),
						# Call description
						dcc.Dropdown(
							id = 'calldescription',
							options=[{'label': x, 'value': x} for x in np.sort(df['calldescription'].unique())],
							multi=True,
							placeholder='Call description')
						]),

				# Choropleth map
				html.Div(className='ten columns',
					children=[dcc.Graph(id="calls_map", figure={})],
					style={"padding-left":"25%", "padding-right": "25%"})
			])
	])

#------------------------------------------------
# Connect Plotly graphs with Dash components

@app.callback(
	[Output(component_id="output_container", component_property="children"),
	Output(component_id="calls_map", component_property="figure")],
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
	app.run_server(host='0.0.0.0', debug=True)