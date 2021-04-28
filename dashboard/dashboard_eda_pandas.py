import pandas as pd
import numpy as np
import pathlib
import json
import plotly.graph_objs as go
import plotly.express as px

import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire, kbc, bmw, gray

import dash
import dash_core_components as dcc 
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
# define current working df.
# The cur_df is the filtered df using currently selected filters
cur_df = df.copy()

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
# value: dict with the following keys {label, format (e.g dropdown or slider), and type of match (i.e. exact or range)}
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
				'category': {'label': 'Call category',
							'format': 'dropdown',
							'type': 'exact'},
				# 'calldescription': {'label': 'Call description',
				# 					'format': 'dropdown',
				# 					'type': 'exact'},
				# 'agency': {'label': 'Responding agency',
				# 			'format': 'dropdown',
				# 			'type': 'exact'},
				# 'intaketime': {'label': 'Intake time (mins)',
				# 				'format': 'slider_range',
				# 				'type': 'range'},
				# 'call_timestamp_EST': {'label': 'Call time (EST)',
				# 						'format': 'date_range',
				# 						'type': 'date_range'},
				'dispatchtime': {'label': 'Dispatch time (mins)',
								'format': 'slider_range',
								'type': 'val_range'},
				'traveltime': {'label': 'Travel time (mins)',
								'format': 'slider_range',
								'type': 'val_range'},
				'totalresponsetime': {'label': 'Total response time (mins)',
										'format': 'slider_range',
										'type': 'val_range'},
				'time_on_scene': {'label': 'Time on scence (mins)',
								'format': 'slider_range',
								'type': 'val_range'},
				'totaltime': {'label': 'Total time (mins)',
							'format': 'slider_range',
							'type': 'val_range'}}
dropdown_filters = [dcc.Dropdown(id = col,
						options=[{'label': x, 'value': x} for x in df[col].unique()],
						multi=True,
						placeholder=value['label'])
					for col, value in filters_info.items()
					if value['format']=='dropdown']

#max_slider_val = 3 # Very very few call with total reponse time more than 10**3 = 1000 minutes
max_slider_val = 120 # Very very few call with total reponse time more than 10**3 = 1000 minutes
slider_filters_id = [col for col, value in filters_info.items() if value['format']=='slider_range']
# slider_filters = [dcc.RangeSlider(id=col,
# 								marks={(i): str(power_of_10_w_zero(i, max_val=max_slider_val+1)) for i in range(-2,max_slider_val+1)},
# 								min=-2,
#         						#max=df[col].max(),
#         						max=max_slider_val,
#         						step=0.01,
#         						#value=[0, df[col].max()],
#         						value=[-2, max_slider_val],
#         						pushable=0.01,
#         						allowCross=False,
#         						#tooltip={'always_visible': True, 'placement': 'bottom'}
#         						) for col in slider_filters_id]
slider_filters = [dcc.RangeSlider(id=col,
								marks={i: str(i) for i in range(0, 121, 20)},
								min=0,
        						#max=df[col].max(),
        						max=max_slider_val,
        						step=0.5,
        						#value=[0, df[col].max()],
        						value=[0, max_slider_val],
        						pushable=0.5,
        						allowCross=False,
        						#tooltip={'always_visible': True, 'placement': 'bottom'}
        						) for col in slider_filters_id]
# add title to sliders
slider_filter_units = [html.Div([filters_info[col]['label'],
						html.Br(),
						# Display the selected range values here.
						html.Div(id=col+'vals', style={'text-align': 'center', 'font-weight': 'bold'}),
						slider_filters[i],
						], style={'border-style': 'solid', 'border-width': '2px', 'border-color': 'gray'}) 
						for i, col in enumerate(slider_filters_id)]

# App layout (Dash components all go inside layout)
app.layout = html.Div(
	#style={"backgroundColor": colors["background"], "pad": 0},
	children = [
		# Title of the web-app
		#html.H1("Detroit City", style={"text-align": "center"}),
		# html.H2("Detroit City 911 Call Records", style={"text-align": "center"}),

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
				# Filters (left side)
				html.Div(className='two columns',
					children=[
						html.H3('Filters'),
						# DROP DOWN FILTERS
						html.Div(dropdown_filters),
						# DATE RANGE PICKER
						dcc.DatePickerRange(id='call_timestamp_EST',
								        min_date_allowed=dt(2016, 9, 20),
								        initial_visible_month=dt(2021, 4, 1),
								        #end_date=dt.now(),
								        start_date_placeholder_text="Start date",
			    						end_date_placeholder_text="End date",
			    						minimum_nights=0,
			    						clearable=True),
						# TIME SLIDER FILTERS
						html.Div(slider_filter_units)
						]),

				# Graphs (right side)
				html.Div(className='ten columns',
					children=[
						# tripgger to indicate if cur_df has been updated.(HIDDEN)
						# value of the trigger can be anything
						html.Div(id='cur_df_trig', children='', hidden=True),
						html.H2("Detroit City 911 Call Records", style={"text-align": "center"}),

						# Choropleth map
						html.Div([
							dcc.RadioItems(id='geounit_select',
											options=[
												{'label': ' Neighborhood  ', 'value': 'nhood'},
												{'label': ' City block', 'value': 'block'},
												{'label': ' Individual', 'value': 'lng_lat'}
											],
											value='lng_lat',
											labelStyle={'display': 'inline-block', 'margin-right': '10px'},
											style={'text-align': 'center', 'font-weight': 'bold'}),
							dcc.Graph(id="calls_map", figure={})
						], className='eight columns'),

						# Histogram
						dcc.Graph(className='four columns', id="top_right", figure={}),
						# Time series graph
						html.Div([
							dcc.RadioItems(id='freq_select',
											options=[
												{'label': ' Hourly  ', 'value': '1h'},
												{'label': ' Daily  ', 'value': '1D'},
												{'label': ' Weekly', 'value': '1W'}
											],
											value='1D',
											labelStyle={'display': 'inline-block', 'margin-right': '10px'},
											style={'text-align': 'center', 'font-weight': 'bold'}),
							dcc.Graph(id="bottom_left", figure={})
						], className='eight columns'),
						#dcc.Graph(className='four columns', id="bottom_mid", figure={}),

						# Bar graph
						dcc.Graph(className='four columns', id="bottom_right", figure={}),
						# Test area
						# html.Div(id='test_area', children=[])
					])
			])
	])

###########################################################
########## Call back #####################################
###########################################################
# @app.callback(
# 	Output('test_area', 'children'),
# 	Input('date_range', 'start_date')
# 	)
# def update_test_area(test_input):
# 	result = test_input
# 	print(result)
# 	print(type(result))
# 	return [result, str(type(result))]

# Filter dataframe
# Exact filters
filters_exact = [Input(key, 'value') for key, value in filters_info.items() if value['type']=='exact']
# Value Range filters
filters_val_range = [Input(key, 'value') for key, value in filters_info.items() if value['type']=='val_range']
# Date range filters
# filters_date_range = [Input(key, 'value') for key, value in filters_info.items() if value['type']=='date_range']
# List of all filters ID. IN ORDER !!!
# Exact filter first then range filters.
filter_inputs = filters_exact + filters_val_range
@app.callback(
	#Output('test_area', 'children'),
	Output('cur_df_trig', 'children'),
	*filter_inputs,
	Input('call_timestamp_EST', 'start_date'),
	Input('call_timestamp_EST', 'end_date')
	)
def filter_df(*args):
	'''
	Filter dataframe based on the supplied values.
	----------------------------------------------
	*args: list of list. Values to filter df by.
		Note that the order of item in list mater. Exact filters first then range filter after.
	'''
	global cur_df
	print('-'*40)
	cur_df = df.copy()
	exact_args = args[:len(filters_exact)]
	val_range_args = args[len(filters_exact):len(filters_exact)+len(filters_val_range)]
	start_date = args[len(filters_exact)+len(filters_val_range)]
	end_date = args[len(filters_exact)+len(filters_val_range)+1]
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
			cur_df = cur_df[cur_df[filter_inputs[idx].component_id].isin(exact_arg)]
		else:
			print(filter_inputs[idx].component_id, ' ', exact_arg, ' ', type(exact_arg))
			cur_df = cur_df[cur_df[filter_inputs[idx].component_id]==exact_arg]
		idx += 1
	# Filter by value range
	# Using log scale
	# for range_arg in val_range_args:
	# 	min_val, max_val = [power_of_10_w_zero(x, max_val=max_slider_val) for x in range_arg]
	# 	print(filter_inputs[idx].component_id, ' ', (min_val, max_val), ' ', type(range_arg))
	# 	cur_df = cur_df[(cur_df[filter_inputs[idx].component_id]>=min_val) & (cur_df[filter_inputs[idx].component_id]<max_val)]
	# 	idx += 1

	# Using linear scale
	for range_arg in val_range_args:
		min_val, max_val = range_arg
		print(filter_inputs[idx].component_id, ' ', (min_val, max_val), ' ', type(range_arg))
		if max_val>=max_slider_val: # if at maxium allowed by slider
			cur_df = cur_df[cur_df[filter_inputs[idx].component_id]>=min_val]
		else:
			cur_df = cur_df[(cur_df[filter_inputs[idx].component_id]>=min_val) & (cur_df[filter_inputs[idx].component_id]<max_val)]
		idx += 1

	# Filter by date range
	# this convert str type to timestamp for comparison
	start_date, end_date = pd.to_datetime([start_date, end_date])
	print(start_date, ' ', end_date)
	if pd.isna(end_date):
		if not pd.isna(start_date):
			cur_df = cur_df[cur_df['call_timestamp_EST'] >= start_date]
	else:
		# At 23-hours and 59 minutes and 59 seconds to max-date to have INCLUSIVE range.
		end_date = end_date + pd.to_timedelta('23h 59m 59S')
		if pd.isna(start_date):
			cur_df = cur_df[cur_df['call_timestamp_EST'] <= end_date]
		else:
			cur_df = cur_df[(cur_df['call_timestamp_EST']>=start_date) & (cur_df['call_timestamp_EST']<=end_date)]
	result = str(cur_df.shape)
	print(result)
	return result


# Display selected range ontop of the sliders.
@app.callback(
	# 'id'+'vals' match with the id of html.Div
	[Output(obj.component_id+'vals', 'children') for obj in filters_val_range],
	filters_val_range
	)
def update_select_range(*args):
	# remember to power_10 the input
	results = []
	# Using log scale
	# for arg in args:
	# 	if power_of_10_w_zero(arg[1]) >= power_of_10_w_zero(max_slider_val, max_val=max_slider_val+1):
	# 		results.append(f'{power_of_10_w_zero(arg[0]):.1f} --- over {power_of_10_w_zero(max_slider_val, max_val=max_slider_val+1):.1f}')
	# 	else:
	# 		results.append(f'{power_of_10_w_zero(arg[0]):.1f} --- {power_of_10_w_zero(arg[1]):.1f}')

	# Using linear scale
	for arg in args:
		min_val, max_val = arg
		if max_val >= max_slider_val:
			results.append(f'{min_val:.1f} --- over {max_val:.1f}')
		else:
			results.append(f'{min_val:.1f} --- {max_val:.1f}')
	return results


##############################################################################
########## GRAPHS callbacks ########################
#######################################################
# Totalresponse time HISTOGRAM
# @app.callback(
# 	Output('top_right', 'figure'),
# 	Input('cur_df_trig', 'children')
# 	)
# def update_hist(_):
# 	if cur_df.empty: # if empty dataframe, skip
# 		return go.Figure()
# 	else:
# 		fig = px.histogram(cur_df, x='totalresponsetime', marginal='box', color='priority', nbins=50)
# 		fig.update_layout(#autosize=True,
# 							width=550,
# 							height=420,
# 							margin={"r":0, "t":0, "l":0, "b":0},
# 		                  font={"size": 14})
# 	return fig

# TIME SERIES graph
@app.callback(
	Output('bottom_left', 'figure'),
	Input('cur_df_trig', 'children'),
	Input('freq_select', 'value')
	)
def update_time_graph(_, freq):
	# Prepare data
	agg = 'count'
	series = preprocessing.time_groupby(cur_df, dt_col='call_timestamp_EST', agg_col='incident_id', agg='count', freq=freq)
	#print(series.describe())
	fig = go.Figure()
	fig.add_trace(go.Scattergl(x=series.index, y=series.values, mode='lines+markers'))
	fig.update_layout(xaxis_title='Call time (EST)', yaxis_title=agg,
					autosize=True, margin={"r":0, "t":0, "l":0, "b":0},
					height=380,
					font={'size': 14})
	return fig

# Category graph
@app.callback(
	Output('bottom_right', 'figure'),
	Input('cur_df_trig', 'children')
	)
def update_bar_graph(_):
	if cur_df.empty: # if empty dataframe, skip
		return go.Figure()
	else:
		# prepare data
		series = preprocessing.cat_groupby(cur_df, 'category', 'incident_id', agg='count')
		series.sort_values(ascending=True, inplace=True)
		# only keep the largest 20 for clear view
		series = series[-20:]
		# hoover info
		hoover_info = series.index + [str(x) for x in series.values]
		fig = px.bar(x=series.values, y=series.index, orientation='h', labels={'x': 'count', 'y': 'Category'})
		fig.update_layout(#autosize=True,
							width=550,
							height=400,
							margin={"r":0, "t":10, "l":0, "b":0},
		                  font={"size": 14},
		                  xaxis_title='count',
		                  yaxis_title='Call category')
	return fig

# MAP
@app.callback(
	Output('calls_map', 'figure'),
	Input('cur_df_trig', 'children'),
	Input('geounit_select', 'value')
	)
def update_map(_,geounit):

	# center of map (Detroit)
	lat = 42.355753
	lng = -83.076760

	# Dict to hold selection info
	select = {'nhood': {'geo_col': 'neighborhood',
						'unit_geo': nhoods_geo,
						'featureidkey': 'properties.nhood_name'},
			'block': {'geo_col': 'block_id',
						'unit_geo': blocks_geo,
						'featureidkey': 'properties.GEOID10'}}

	# Datadasher map
	if geounit=='lng_lat':
		# Data must not have NaN in long and lat for datashader to work.
		df = cur_df.dropna(subset=['latitude', 'longitude'])
		return datashader_map(df, lng_c=lng, lat_c=lat, height=400)

	# Prepare data for choropleth map
	df = preprocessing.cat_groupby(cur_df, select[geounit]['geo_col'], 'incident_id', agg='count')
	if df.empty:
		return fig.add_trace(go.Choroplethmapbox(geojson=select[geounit]['unit_geo'],
												featureidkey="properties.nhood_name"))
	# select type of boundaries to plot
	return choropleth_map(df, geojson=select[geounit]['unit_geo'],
								featureidkey=select[geounit]['featureidkey'],
								lng_c=lng, lat_c=lat, height=400)

# Create choropleth map
def choropleth_map(df, geojson, featureidkey, lng_c=-83.076760, lat_c=42.355753, height=400):
	# colorscale min and max
	thres_min = df.min()
	thres_max = df.max()

	fig = go.Figure()
	# add choropleth traces to figure   
	fig.add_trace(go.Choroplethmapbox(geojson=geojson,
                                       locations=df.index,
                                       z=df.values,
                                       featureidkey=featureidkey,
                                       zmin=thres_min,
                                       zmax=thres_max,
                                       colorscale=fire,
                                       #text=df["Neighborhood"],
                                       #hovertemplate="<b>%{text}</b><br>Attention level: %{z:.2f}",
                                       #name=times[i],
                                       marker_opacity=0.7))

	fig.update_layout(mapbox_style="carto-darkmatter",
	                  mapbox_zoom=10,
	                  mapbox_center={"lat": lat_c, "lon": lng_c},
	                  #sliders=sliders
	                  )
	fig.update_layout(autosize=True,
						height=height,
						margin={"r":0, "t":0, "l":0, "b":0},
	                  font={"size": 15})
	return fig

# Create map using Datashader and Plotly
def datashader_map(df, lng_c=-83.076760, lat_c=42.355753, height=400):
	'''
	df: Dataframe with 'longitude' and 'latitude' columns.
	lat_c: float. Latitude to center the map.
	lng_c: float. Longitude to center the map.
	height: int. Map height
	-------------------------------------------------
	return: Plotly figure object.
	'''
	# Build an abstract canvas representing the space in which to plot data
	cvs = ds.Canvas(plot_width=1500, plot_height=750)

	# project the longitude and latitude onto the canvas and
	# map the data to pixels as points
	aggs = cvs.points(df, x='longitude', y='latitude')

	# aggs is an xarray object, see http://xarray.pydata.org/en/stable/ for more details
	coords_lat, coords_lon = aggs.coords['latitude'].values, aggs.coords['longitude'].values

	# Set the corners of the image that need to be passed to the mapbox
	coordinates = [[coords_lon[0], coords_lat[0]],
	               [coords_lon[-1], coords_lat[0]],
	               [coords_lon[-1], coords_lat[-1]],
	               [coords_lon[0], coords_lat[-1]]]


	# Set the image color, and the legend (how) types
	# linear (how=linear), logarithmic (how=log), percentile (how=eq_hist)
	img = tf.shade(aggs, cmap=fire, how='eq_hist', alpha=255, min_alpha=40)[::-1].to_pil()

	# Create a quick mapbox figure with plotly
	fig = px.scatter_mapbox(df[:1], lat='latitude', lon='longitude', zoom=10)

	# Add the datashader image as a mapbox layer image
	fig.update_layout(mapbox_style="carto-darkmatter",
	                  mapbox_layers=[{
	                    "sourcetype": "image",
	                    "source": img,
	                    "coordinates": coordinates}],
	                    mapbox_center={"lat": lat_c, "lon": lng_c}
	)
	fig.update_layout(autosize=True,
						height=height,
						margin={"r":0, "t":0, "l":0, "b":0},
	                  font={"size": 15})
	return fig

# ------------------------------------------------
if __name__ == "__main__":
	app.run_server(host='0.0.0.0', debug=True)