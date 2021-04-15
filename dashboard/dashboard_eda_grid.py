import pandas as pd
import numpy as np
import pathlib
import json
import plotly.graph_objs as go 

import dash
import dash_core_components as dcc 
import dash_ui as dui
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

# to be used with date_picker
from datetime import datetime as dt

from utils import preprocessing

# Using style css sheet by chriddyp
# https://codepen.io/chriddyp/pen/bWLwgP
# This is done by having the css file inside the 'assets' directory.
app = dash.Dash(__name__)
my_css_urls = ["https://codepen.io/rmarren1/pen/mLqGRg.css"]

for url in my_css_urls:
    app.css.append_css({
        "external_url": url
    })
#################################################################
################ Read in csv data file #########################
##################################################################
# 911 call records
# Check if processed data exists
directory = pathlib.Path('../data/processed/')
# define the filename pattern
name_pattern = f"911_Calls_for_dashboard.csv"
if len(list(directory.glob(name_pattern))) == 0:
	# Select years of data to include
	years = [2021]
	preprocessing.process_data_dashboard(years)
df = pd.read_csv('../data/processed/911_Calls_for_dashboard.csv', 
	thousands=",", dtype={'priority':'str', 'block_id': 'str'},
	parse_dates=['call_timestamp_EST'])


# print(df.info())
# print(df['call_timestamp_EST'].head(5))
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


def power_of_10_w_zero(num, min_val=-2, max_val=3):
	'''
	Return power of 10 to the number num, i.e. 10**num.
	Note: if num == -2, return 0 instead of 0.01.
	'''
	if num <= min_val:
		return 0
	elif num >= max_val:
		return 10**6 # return some large number.
	else:
		return 10**num

##################################################################
############### DASH APP ###############################
##########################################################
# Background color
colors = {
    'background': '#006e99',
    'text': '#7FDBFF'
}
# DEfine filter schema dictionary:
# key: str. columns in dataframe to be used as filters.
# value: dict of format {filter label, format (e.g dropdown or slider), and type of match (i.e. exact or range)}
# For example, exact match filter will keep values that as exact match the criteria.
# Range filter will keep values between the min and max of the criterai.
filters_info = {'officerinitiated': {'label': 'Call made by officer?',
											'format': 'dropdown',
											'type': 'exact'},
				'priority': {'label': 'Call priority',
							'format': 'dropdown',
							'type': 'exact'},
				'zip_code': {'label': 'Zip code',
							'format': 'dropdown',
							'type': 'exact'},
				'neighborhood': {'label': 'Neighborhood',
								'format': 'dropdown',
								'type': 'exact'},
				'block_id': {'label': 'Block ID',
							'format': 'dropdown',
							'type': 'exact'},
				'calldescription': {'label': 'Call description',
									'format': 'dropdown',
									'type': 'exact'},
				'agency': {'label': 'Responding agency',
							'format': 'dropdown',
							'type': 'exact'},
				'intaketime': {'label': 'Intake time (mins)',
								'format': 'slider_range',
								'type': 'range'},
				'dispatchtime': {'label': 'Dispatch time (mins)',
								'format': 'slider_range',
								'type': 'range'},
				'traveltime': {'label': 'Travel time (mins)',
								'format': 'slider_range',
								'type': 'range'},
				'totalresponsetime': {'label': 'Total response time (mins)',
										'format': 'slider_range',
										'type': 'range'},
				'time_on_scene': {'label': 'Time on scence (mins)',
								'format': 'slider_range',
								'type': 'range'},
				'totaltime': {'label': 'Total time (mins)',
							'format': 'slider_range',
							'type': 'range'}}
dropdown_filters = [dcc.Dropdown(id = col,
						options=[{'label': x, 'value': x} for x in df[col].unique()],
						multi=True,
						placeholder=value['label'])
					for col, value in filters_info.items()
					if value['format']=='dropdown']

max_slider_val = 3 # Very very few call with total reponse time more than 10**3 = 1000 minutes
slider_filters_id = [col for col, value in filters_info.items() if value['format']=='slider_range']
slider_filters = [dcc.RangeSlider(id=col,
								marks={(i): str(power_of_10_w_zero(i, max_val=max_slider_val+1)) for i in range(-2,max_slider_val+1)},
								min=-2,
        						#max=df[col].max(),
        						max=max_slider_val,
        						step=0.01,
        						#value=[0, df[col].max()],
        						value=[-2, max_slider_val],
        						pushable=0.01,
        						allowCross=False,
        						#tooltip={'always_visible': True, 'placement': 'bottom'}
        						) for col in slider_filters_id]
# add title to sliders
slider_filter_units = [html.Div([filters_info[col]['label'],
						html.Br(),
						html.Div(id=col+'vals'), # Display the selected range values here.
						slider_filters[i],
						html.Br(), html.Br()]) 
						for i, col in enumerate(slider_filters_id)]


grid = dui.Grid(_id="grid", num_rows=12, num_cols=12, grid_padding=5)
# Title of the web-app
# grid.add_element(col=3, row=1, width=10, height=1,
# 				element=html.Div([html.H2("Detroit City 911 Call Records", style={"text-align": "center"}), 
# 								# dataframe place holder (HIDDEN)
# 								html.Div(id='current_df', hidden=True),
# 								# Test area
# 								html.Div(id='test_area', children=[])
# 								])
# 				)
# Filters (left side)
grid.add_element(col=1, row=1, width=4, height=11,
				element=html.Div(
							children=[
								html.H3('Filters'),
								# DROP DOWN FILTERS
								*dropdown_filters,
								# DATE RANGE PICKER
								dcc.DatePickerRange(
		        					id='date_range',
							        min_date_allowed=dt(2016, 9, 20),
							        initial_visible_month=dt(2021, 4, 1),
							        #end_date=dt.now(),
							        start_date_placeholder_text="Start date",
		    						end_date_placeholder_text="End date",
		    						minimum_nights=0,
		    						clearable=True),
								# TIME SLIDER FILTERS
								*slider_filter_units
								])
				)
# grid.add_graph(col=3, row=2, width=7, height=5, graph_id='calls_map')
# grid.add_graph(col=10, row=2, width=3, height=5, graph_id='top_right')
# grid.add_graph(col=3, row=7, width=3, height=6, graph_id='bottom_left')
# grid.add_graph(col=6, row=7, width=3, height=6, graph_id='bottom_mid')
# grid.add_graph(col=9, row=7, width=4, height=6, graph_id='bottom_right')

# App layout (Dash components all go inside layout)
app.layout = html.Div(
    dui.Layout(
        grid=grid
        #controlpanel=controlpanel
    ),
    style={
        'height': '100vh',
        'width': '100vw'
    }
)

###########################################################
########## Call back #####################################
###########################################################
# @app.callback(
# 	Output('test_area', 'children'),
# 	Input('date_range', 'start_date')
# 	)
def update_test_area(test_input):
	result = test_input
	print(result)
	print(type(result))
	return [result, str(type(result))]

# Filter dataframe
# Exact filters
# filters_exact = [col for col, value in filters_info.items() if value['type']=='exact']
filters_exact = [Input(key, 'value') for key, value in filters_info.items() if value['type']=='exact']
# Range filters
#filters_range = [col for col, value in filters_info.items() if value['type']=='range']
filters_range = [Input(key, 'value') for key, value in filters_info.items() if value['type']=='range']
# List of all filters ID. IN ORDER !!!
# Exact filter first then range filters.
filter_inputs = filters_exact + filters_range
@app.callback(
	#Output('current_df', 'children'),
	Output('test_area', 'children'),
	filter_inputs
	)
def filter_df(*args):
	'''
	Filter dataframe based on the supplied values.
	----------------------------------------------
	*args: list of list. Values to filter df by.
		Note that the order of item in list mater. Exact filters first then range filter after.
	'''
	print('-'*40)
	dff = df.copy()
	exact_args = args[:len(filters_exact)]
	range_args = args[len(filters_exact):]
	# keep track of filter index
	idx = 0
	# filter exact value
	for exact_arg in exact_args:
		if exact_arg==None:
			idx += 1
			continue
		elif type(exact_arg)==list: # filter a feature with multiple values (i.e. UNION filter, that is 'or')
			print(filter_inputs[idx].component_id, ' ', exact_arg, ' ', type(exact_arg))
			if len(exact_arg)==0: # if empty list. This happens when dropdown is cleared.
				idx += 1
				continue
			dff = dff[dff[filter_inputs[idx].component_id].isin(exact_arg)]
		else:
			print(filter_inputs[idx].component_id, ' ', exact_arg, ' ', type(exact_arg))
			dff = dff[dff[filter_inputs[idx].component_id]==exact_arg]
		idx += 1
	# Filter by range
	for range_arg in range_args:
		min_val = power_of_10_w_zero(range_arg[0], max_val=max_slider_val)
		max_val = power_of_10_w_zero(range_arg[1], max_val=max_slider_val)
		print(filter_inputs[idx].component_id, ' ', (min_val, max_val), ' ', type(range_arg))
		dff = dff[(dff[filter_inputs[idx].component_id]>=min_val) & (dff[filter_inputs[idx].component_id]<max_val)]
		idx += 1
	#result = [str(dff.shape), str(power_of_10_w_zero(range_args[-1][0])), str(power_of_10_w_zero(range_args[-1][1]))]
	result = str(dff.shape)
	return result

@app.callback(
	# 'id'+'vals' match with the id of html.Div
	[Output(obj.component_id+'vals', 'children') for obj in filters_range],
	filters_range
	)
def update_select_range(*args):
	# remember to power_10 the input
	# if power_of_10_w_zero(arg[1]) >= power_of_10_w_zero(max_slider_val):
	# 	results = [dcc.Markdown(f'**{power_of_10_w_zero(arg[0]):.1f}** --> **> {power_of_10_w_zero(arg[1]):.1f}**') for arg in args]
	# else:
	# 	results = [dcc.Markdown(f'**{power_of_10_w_zero(arg[0]):.1f}** --> **{power_of_10_w_zero(arg[1]):.1f}**') for arg in args]

	results = []
	for arg in args:
		if power_of_10_w_zero(arg[1]) >= power_of_10_w_zero(max_slider_val, max_val=max_slider_val+1):
			results.append(dcc.Markdown(f'**{power_of_10_w_zero(arg[0]):.1f}** --> **> {power_of_10_w_zero(max_slider_val, max_val=max_slider_val+1):.1f}**'))
		else:
			results.append(dcc.Markdown(f'**{power_of_10_w_zero(arg[0]):.1f}** --> **{power_of_10_w_zero(arg[1]):.1f}**'))
	return results


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