# IMPORTANT NOTES:
# About sharing data between callbacks, I first attempted to filter the dataframe and 
# then share the dataframe's state (which is a dict) between callbacks. However I ran into
# some technical issues with this approach. The dataframe state dict COULD NOT be converted
# to JSON (which is the required dataformat of Dash to be passed on) due to non-string memory
# variable in the dict. Therefore, I decided to go with this method.
# The dataframe state dict will be stored as fask-cache. A signal will be passed to other
# callbacks through a hidden Div. 


import vaex
import numpy as np
import pandas as pd
import pyarrow as pa
import geopandas as gpd # to filter GEOJSON file easier
import pathlib
import json
import ast
import plotly.graph_objs as go
import plotly.express as px

import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire, kbc, bmw, gray

import dash
import dash_core_components as dcc 
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
from flask_caching import Cache
import os

# to be used with date_picker
import datetime

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from utils import preprocessing

import pprint

# Using style css sheet by chriddyp
# https://codepen.io/chriddyp/pen/bWLwgP
# This is done by having the css file inside the 'assets' directory.
app = dash.Dash(__name__)

# Set up the caching mechanism
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
# set negative to disable (useful for testing/benchmarking)
CACHE_TIMEOUT = int(os.environ.get('DASH_CACHE_TIMEOUT', '60'))
#################################################################
################ Read in csv data file #########################
##################################################################
# 911 call records
# Check if processed data exists
directory_path = '../data/processed/'
directory = pathlib.Path(directory_path)
# define the filename pattern
name_pattern = f"911_Calls_for_dashboard.csv.hdf5"
if len(list(directory.glob(name_pattern))) == 0:
	# check if csv file exists
	name_pattern_csv = f"911_Calls_for_dashboard.csv"
	if len(list(directory.glob(name_pattern))) == 0:
		# Select years of data to include
		years = range(2016, 2022, 1)
		preprocessing.process_data_dashboard(years)
	else:
		# if csv exists, then create hdf5 file
		vaex.from_csv(directory_path + name_pattern_csv, parse_dates=['call_timestamp_EST'], convert=True)
df_original = vaex.open(directory_path + name_pattern)
#df_original['call_timestamp_EST'] = df_original['call_timestamp_EST'].astype('datetime64[ns]')

# print(df.info())
# print(df['call_timestamp_EST'].head(5))
# print(df['zip_code'].unique())
# print(df['priority'].unique())

# Read in geo-json file and convert to geopandas
# read in Detroit neighborhood geojson.
with open("../data/raw/neighborhoods_info_2021-04-09.geojson") as geojson_file:
    #nhoods_geo = gpd.GeoDataFrame.from_features(json.load(geojson_file)['features'])
    nhoods_geo = json.load(geojson_file)
# read in Detroit city blocks geojson.
with open("../data/raw/city_blocks_info_2021-04-09.geojson") as geojson_file:
    blocks_geo = gpd.GeoDataFrame.from_features(json.load(geojson_file)['features'])



##################################################################
# Components
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
						options=[{'label': x, 'value': x} for x in df_original[col].unique()],
						multi=True,
						placeholder=value['label'])
					for col, value in filters_info.items()
					if value['format']=='dropdown']

#max_slider_val = 3 # Very very few call with total reponse time more than 10**3 = 1000 minutes
max_slider_val = 120 # Very very few call with total reponse time more than 10**3 = 1000 minutes
slider_filters_id = [col for col, value in filters_info.items() if value['format']=='slider_range']
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


# Call category map
category_map = {'TRAFF2': 'Traffic incident with injuries',
				'TRF STOP': 'Traffic stops',
				'SPCL ATT': 'Special attention',
				'STRTSHFT': "Officers' shift start updates",
				'REMARKS': "Officers' remarks",
				'FA IP': 'Felonious assault in pursuit',
				'ACCUNK': 'Automobile accident, unknown injuries',
				'LARCREPT': 'Larceny report',
				'FRAUDRPT': 'Fraud report',
				'INVAUTO': 'Investigating auto',
				'BUS BRD': 'Bus boarding',
				'TOW': 'Towing',
				'HNGUP': 'Calls were hung-up',
				'DISTURB': 'Disturbance',
				'INVPERS': 'Investigate person',
				'UNKPROB': 'Unknown problem',
				'AB IP/JH': 'Assaut and battery in pursuit',
				'VERALRM': 'Verified alarm or person w/o code',
				'SHOTS IP': 'Shots fired, in pursuit',
				'WEAPON': 'Person with weapon',
				'DV A/B': 'Domestic violence',
				'UDAAREPT': 'Unlawful driving away of automobile',
				'OD': 'Drug overdose',
				'HI1 I/P': 'Burglary occupied residence',
				'BLDGCHK': 'Building check',
				'MISCTRAF': 'Miscellaneous traffic',
				'MDPIP': 'Malicious destruction in pursuit',
				'ACCINJ': 'Automobile accident with injuries',
				'THREATRP': 'Threat report',
				'PANIC': 'Panic or duress alarm',
				'INFORPT': 'Non-criminal information report',
				'RECAUTO': 'Recover auto',
				'MNTLNARM': 'Mental violent not armed',
				'WBC': 'Well-being check',
				'HRUNK': 'Automobile hit and run, unknown injuries',
				'VICANML': 'Vicious animal',
				'PARK': 'Parking complaint',
				'HOLDUP': 'Hold-up alarm',
				'LARCENY': 'Larceny',
				'MDPRPT': 'Malicious destruction report',
				'AO': 'Assist other',
				'ANMLCOMP': 'Animal complaint'}
###########################################################
# CREATE DATA
###########################################################
# Filter dataframe
# Exact filters
filters_exact = [Input(key, 'value') for key, value in filters_info.items() if value['type']=='exact']
# Value Range filters
filters_val_range = [Input(key, 'value') for key, value in filters_info.items() if value['type']=='val_range']

# List of all filters ID. IN ORDER !!!
# Exact filter first then range filters.
filter_inputs = filters_exact + filters_val_range

# Display selected range ontop of the sliders.
@app.callback(
	# 'id'+'vals' match with the id of html.Div
	[Output(obj.component_id+'vals', 'children') for obj in filters_val_range],
	filters_val_range
	)
def update_select_range(*args):
	# remember to power_10 the input
	results = []

	# Using linear scale
	for arg in args:
		min_val, max_val = arg
		if max_val >= max_slider_val:
			results.append(f'{min_val:.1f} --- over {max_val:.1f}')
		else:
			results.append(f'{min_val:.1f} --- {max_val:.1f}')
	return results


# FILTER DATA
# Generate the data based on selection
@app.callback(
	#Output('test_area', 'children'),
	Output('cur_df_trig', 'children'),
	Output('data_selected', 'children'),
	Output('selected_progress', 'value'),
	*filter_inputs,
	Input('call_timestamp_EST', 'start_date'),
	Input('call_timestamp_EST', 'end_date')
	)
def function(*args):
	# Convert inputs list to str to be used as vaex selection label
	inputs = str(args)
	# filter data
	df, label = filter_df(inputs)
	df_cur_count = df.count(selection=True)
	df_org_count = df.count(selection=False)
	# return the inputs to hidden Div.
	# this is needed to get cached of filter_df by other functions.
	return inputs, f'{df_cur_count:,} / {df_org_count:,}', str(df_cur_count)


@cache.memoize(timeout=CACHE_TIMEOUT)
def filter_df(inputs: str):
	'''
	inputs: str representation of an input list
	------------------------------------------------
	return: vaex df and selection (str of the inputs)
	'''
	# Convert input str to list
	args = ast.literal_eval(inputs)

	# Start out with the original dataset.
	df = df_original.copy()

	print('-'*40)
	exact_args = {key.component_id: args[i] for i, key in enumerate(filters_exact)}
	val_range_args = {key.component_id: args[len(filters_exact) + i] for i, key in enumerate(filters_val_range)}
	start_date = args[len(filters_exact)+len(filters_val_range)]
	end_date = args[len(filters_exact)+len(filters_val_range)+1]

	# exact value filters, as dict
	print(exact_args)
	# Range value filters, as dict
	print(val_range_args)
	print('Date start: ', start_date)
	print('Date end: ', end_date)

	# define filter label. Useful for caching purpose later on
	label = inputs # inputs is a string here

	# Filter
	df = filter_df_exact(df, exact_args)
	#print('Filter exact: ', df.count(selection=selection))
	df = filter_df_range(df, val_range_args)
	#print('Filter range: ', df.count(selection=selection))
	df = filter_df_date_range(df, start_date, end_date)
	#print('Filter date: ', df.count(selection=selection))
	return df, label


# NEED CACHE HERE !!!
def filter_df_exact(df, schema):
	'''
	Filter dataframe to match exact the provided values.
	----------------------------------------------
	schema: dict. key: dataframe column names.
				value: values or list of values to filter the column by.
	-------------------------------------------------------
	return: vaex df.
	'''
	# filter exact value
	for key,values in schema.items():
		if values==None:
			continue
		elif type(values)==list: # filter a feature with multiple values (i.e. UNION filter, that is 'or')
			if len(values)==0: # if empty list. This happens when dropdown is cleared.
				continue
			df.select(df[key].isin(values), mode='and')
		else:
			df.select(df[key]==values, mode='and')
	return df


def filter_df_range(df, schema):
	'''
	Filter dataframe to within range of values.
	----------------------------------------------
	schema: dict. key: dataframe column names.
				value: values or list of values to filter the column by.
	------------------------------------------------
	return: Vaex df
	'''
	# filter in range of values
	for key,values in schema.items():
		min_val, max_val = values
		if max_val>=max_slider_val: # if at maxium allowed by slider
			df.select(df[key] >= min_val, mode='and')
		else:
			df.select((df[key] >= min_val) & (df[key] < max_val), mode='and')
	return df


def filter_df_date_range(df, start_date, end_date):
	'''
	Filter dataframe to within range of values.
	----------------------------------------------
	start_date: str. Of format: YYYY/MM/DD
	end_date: str. Of format: YYYY/MM/DD
	-----------------------------------------------
	return. Vaex df.
	'''
	# Filter by date range
	if end_date==None:
		if start_date != None:
			# this convert str type to timestamp for comparison
			#start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
			start_date = np.datetime64(start_date)
			df.select(df['call_timestamp_EST'] >= start_date, mode='and')
	else:
		#end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
		end_date = np.datetime64(end_date)
		# Add 23-hours and 59 minutes and 59 seconds to max-date to have INCLUSIVE range.
		#end_date = end_date + datetime.datetime.timedelta(hours=23, minutes=59, seconds=59)
		end_date = end_date + np.timedelta64(23, 'h') + np.timedelta64(59, 'm') + np.timedelta64(59, 's')
		if start_date==None:
			df.select(df['call_timestamp_EST'] <= end_date, mode='and')
		else:
			start_date = np.datetime64(start_date)
			df.select((df['call_timestamp_EST'] >= start_date) & (df['call_timestamp_EST'] <= end_date), mode='and')
	return df



#######################################################
# CREATE DATA for GRAPHS 
#######################################################
@cache.memoize(timeout=CACHE_TIMEOUT)
def create_hist_data(df: vaex.dataframe.DataFrameLocal,
					label: str, limits=[0, 120]):
	'''
	Calculate aggregated data to be used for histogram.
	--------------------------------------------------------
	df: Vaex dataframe
	limits: LIst of 2 int. It specifies the min and max of binning range.
		By default, consider only 0 - 120 minutes.
	label: str. NOT USED for calculation. Only here for caching purpose
	-------------------------------------------------------------
	return: dict.
	'''
	bins = 50

	results = {'totalresponsetime': {'x': df.bin_edges(df['totalresponsetime'], limits=limits, shape=bins),
									'count': df.count(binby=[df['totalresponsetime']], limits=limits, shape=bins, selection=True)},
				'time_on_scene': {'x': df.bin_edges(df['time_on_scene'], limits=limits, shape=bins),
									'count': df.count(binby=[df['time_on_scene']], limits=limits, shape=bins, selection=True)}}
	return results


@cache.memoize(timeout=CACHE_TIMEOUT)
def create_time_data(df: vaex.dataframe.DataFrameLocal, label: str,
					start_date, end_date, dt_col, agg_col, agg='count', freq='D'):
	'''
	Aggregrate on a time column using the selected agg method.
	----------------------------------------------------------------
	label: str. NOT USED for calculation. Only here for caching purpose
	----------------------------------------------------------------
	return: 2 arrays for x and y values. Contains column name dt_col and agg.
	'''
	# group by time interval
	data = preprocessing.time_groupby_vaex(df, selection=True, dt_col=dt_col, agg_col=agg_col, agg=agg, freq=freq)

	# Since vaex time-groupby also return data outside of selection (with 0 counts),
	# I will remove these datapoints for better presentation
	# NOte that trying to .filter() data first then time-groupby resulted in error, somehow.
	selection = None
	if start_date != None:
		start_date = np.datetime64(start_date) # start date str at -2 location
		if end_date != None:
			end_date = np.datetime64(end_date) + np.timedelta64(1, 'D') # end date str at -1 location
			data.select((data[dt_col] >= start_date) & (data[dt_col] <= end_date)) # filter date range
			selection=True
		else: # no end date selected
			data.select(data[dt_col] >= start_date)
			selection=True
	else:  # no start date selected
		if end_date != None:
			end_date = np.datetime64(end_date) + np.timedelta64(1, 'D') # end date str at -1 location
			data.select(data[dt_col] <= end_date) # filter date range
			selection=True

	z = None
	if freq=='D':
		# Add day of week label
		data['day_of_week'] = data[dt_col].dt.day_name
		z = data.evaluate(data['day_of_week'], selection=selection)

	# get x and y values
	x = data.evaluate(data[dt_col], selection=selection)
	y = data.evaluate(data[agg], selection=selection)
	return x, y, z


@cache.memoize(timeout=CACHE_TIMEOUT)
def create_bar_data(df, label, cat_col, agg_col, agg='count') -> pd.DataFrame:
	'''
	label: str. NOT USED for calculation. Only here for caching purpose
	'''
	# Group by call category
	data = preprocessing.cat_groupby_vaex(df, selection=True, cat_col=cat_col, agg_col=agg_col, agg=agg)
	# sort ascending on agg values
	data = data.sort(by=data[agg], ascending=True).to_pandas_df()

	data['hover'] = data[cat_col].apply(lambda x: category_map.get(x, x))
	return data


@cache.memoize(timeout=CACHE_TIMEOUT)
def create_choropleth_nhoods_data(df, label, cat_col) -> pd.DataFrame:

	# Group by
	agg = 'count'
	# data is of type vaex.dataframe.DataFrameLocal
	data = preprocessing.cat_groupby_vaex(df, selection=True, cat_col=cat_col, agg_col='incident_id', agg=agg)
	# data is vaex.dataframe
	# Convert data to pandas df. To join with geopandas later.
	return data.to_pandas_df()


# NOTE: I create a different (but the same structure) choropleth data creation function 
# for block level geo because this function is called with map zoom and pan (unlike the function above)
# therefore this one will not be cached.
#@cache.memoize(timeout=CACHE_TIMEOUT)
def create_choropleth_blocks_data(df, label, cat_col) -> pd.DataFrame:
	# Group by
	agg = 'count'
	# data is of type vaex.dataframe.DataFrameLocal
	data = preprocessing.cat_groupby_vaex(df, selection=True, cat_col=cat_col, agg_col='incident_id', agg=agg)
	return data.to_pandas_df()


@cache.memoize(timeout=CACHE_TIMEOUT)
def create_heatmap_data(df, label, heatmap_limits):
	'''
	df: Vaex dataframe
	selection: Vaex selection, bool or str.
	heatmap_limits: dict.
	'''
	w_h = 1.9
	height_bins = 250
	width_bins = int(250 * w_h)

	heatmap_data_array = df.count(binby=[df['longitude'], df['latitude']],
									selection=True,
									limits=heatmap_limits, # coordinates of view
									shape=[width_bins, height_bins], # Number of pixel (i.e. aggregate groups) in view.
									array_type='xarray') # Return xarray for easier plotting.
	# Rotate the axis to have longitude as horizontal and latitude as vertical
	# NOTE: IT IS CRUCIAL THAT THE DATATYPE IS SET (e.g. 'unit32').
	# Without explicitly setting the datatype, later datashader.transform_function.shade() WILL NOT
	# return TRANSPARENT background image.
	return heatmap_data_array.astype('uint32').T

#######################################################
# CREATE GRAPHS
#######################################################
def make_empty_plot():
	layout = go.Layout(plot_bgcolor='white', width=10, height=10,
						xaxis=go.layout.XAxis(visible=False),
						yaxis=go.layout.YAxis(visible=False))
	return go.FigureWidget(layout=layout)

def create_hist_fig(x, counts, title=None, xlabel=None, ylabel=None):
	'''
	Create a mimic of plotly histogram using plotly scatter plot.
	This mimicing is to boost performance as creating histogram with plotly is slow.
	----------------------------------------------------------------
	x: list or np.array. X values
	count: list or np.array. Counts for each x values.
	'''
	color = 'royalblue'
	# list of traces
	traces = []

	# Create the figure
	line = go.scatter.Line(color=color, width=2)
	hist = go.Scatter(x=x, y=counts,
					mode='lines', line_shape='hv', line=line,
					name=title, fill='tozeroy')
	traces.append(hist)

	# Layout
	title = go.layout.Title(text=title, x=0.5, y=1, font={'color': 'black'})
	margin = go.layout.Margin(l=0, r=0, b=5, t=5, pad=0)
	legend = go.layout.Legend(orientation='h',
								bgcolor='rgba(0,0,0,0)',
								x=0.5,
								y=1,
								itemclick=False,
								itemdoubleclick=False)
	layout = go.Layout(height=215,
						width=400,
						margin=margin,
						legend=legend,
						title=title,
						xaxis=go.layout.XAxis(title=xlabel),
						yaxis=go.layout.YAxis(title=ylabel),
						font={'size': 14},
						#**fig_layout_defaults
						)

	# Now calculate the most likely value (peak of the histogram)
	peak = np.round(x[np.argmax(counts)], 2)

	return go.FigureWidget(data=traces, layout=layout)


def create_time_fig(x, y, z, title=None, xlabel=None, ylabel=None, height=350):
	'''
	z: None or array. Day of week labels when datapoints are in day.
	'''
	traces = []
	if z!=None:
		line = go.Scattergl(x=x, y=y, text=z, mode='lines+markers',
							hovertemplate='%{x}<br>%{text}<br>%{y:,}<extra></extra>')
	else:
		line = go.Scattergl(x=x, y=y, mode='lines+markers',
							hovertemplate='%{x}<br>%{y:,}<extra></extra>')
	traces.append(line)
	# layout
	title = go.layout.Title(text=title, x=0.5, y=1, font={'color': 'black'})
	margin = go.layout.Margin(l=0, r=0, b=0, t=0)
	legend = go.layout.Legend(orientation='h',
							bgcolor='rgba(0,0,0,0)',
							x=0.5,
							y=1,
							itemclick=False,
							itemdoubleclick=False)
	layout = go.Layout(height=height,
						font={'size': 14},
						margin=margin,
						legend=legend,
						title=title,
						xaxis=go.layout.XAxis(title=xlabel),
						yaxis=go.layout.YAxis(title=ylabel))
	return go.FigureWidget(data=traces, layout=layout)

def create_bar_fig(x, y, hover=None, title=None, xlabel=None, ylabel=None):
	traces = []
	bar = go.Bar(x=x, y=y, orientation='h', text=hover,
				hovertemplate='%{text}<br>%{x:,}<extra></extra>')
	traces.append(bar)
	# layout
	title = go.layout.Title(text=title, x=0.5, y=1, font={'color': 'black'})
	margin = go.layout.Margin(l=0, r=0, b=0, t=20)
	legend = go.layout.Legend(orientation='h',
							bgcolor='rgba(0,0,0,0)',
							x=0.5,
							y=1,
							itemclick=False,
							itemdoubleclick=False)
	layout = go.Layout(height=380,
						width=400,
						font={'size': 14},
						margin=margin,
						legend=legend,
						title=title,
						xaxis=go.layout.XAxis(title=xlabel),
						yaxis=go.layout.YAxis(title=ylabel))
	return go.FigureWidget(data=traces, layout=layout)


# Create choropleth map
def create_choropleth_map(geo_array: pd.Series, z: pd.Series, geojson, featureidkey,
						lng_c=-83.076760, lat_c=42.355753, zoom=10,
						height=400, color_min=None, color_max=None,
						hover_label='Neighborhood'):
	'''
	geojson: GEOJSON or geopandas['geometry']. If use geopandas, must use plotly express instead of Graphic Object.
	'''
	# colorscale min and max
	if color_min==None or color_max==None:
		color_min = z.min()
		color_max = z.max()
	fig = go.Figure()
	# # add choropleth traces to figure   
	fig.add_trace(go.Choroplethmapbox(geojson=geojson,
                                       locations=geo_array,
                                       z=z,
                                       featureidkey=featureidkey,
                                       zmin=color_min,
                                       zmax=color_max,
                                       colorscale=fire,
                                       showscale=False, # Turn off color scale
                                       #coloraxis='coloraxis',
                                       text=geo_array,
                                       hovertemplate=f"<b>{hover_label}: %{{text}}</b><br>Count: %{{z:,}}<extra></extra>",
                                       #name=times[i],
                                       marker_opacity=0.7))
	#fig.update_traces(showscale=False)
	fig.update_layout(mapbox_style="carto-darkmatter",
	                  mapbox_zoom=zoom,
	                  mapbox_center={"lat": lat_c, "lon": lng_c},
	                  #sliders=sliders
	                  )
	fig.update_layout(autosize=True,
						height=height,
						margin={"r":0, "t":0, "l":0, "b":0},
	                  font={"size": 15})
	return fig


def create_choropleth_empty_map(geojson, featureidkey, lng_c=-83.076760, lat_c=42.355753, zoom=10, height=400):
	fig = go.Figure()
	# add choropleth traces to figure   
	fig.add_trace(go.Choroplethmapbox(geojson=geojson,
                                       featureidkey=featureidkey,
                                       colorscale=fire,
                                       marker_opacity=0.7))

	fig.update_layout(mapbox_style="carto-darkmatter",
	                  mapbox_zoom=zoom,
	                  mapbox_center={"lat": lat_c, "lon": lng_c},
	                  #sliders=sliders
	                  )
	fig.update_layout(autosize=True,
						height=height,
						margin={"r":0, "t":0, "l":0, "b":0},
	                  font={"size": 15})
	return fig


def create_heatmap_fig(data, lat_c, lng_c, zoom, height=400):
	'''
	data: Xarray data. Must have coordinates names: 'longitude' and 'latitude'
	'''
	# Create heatmap layer
	# Set the image color, and the legend (how) types
	# linear (how=linear), logarithmic (how=log), percentile (how=eq_hist)
	img = tf.shade(data, cmap=fire, how='eq_hist', alpha=255, min_alpha=40)[::-1].to_pil()

	# Create mapbox layer
	left = data.coords['longitude'].min().values
	right = data.coords['longitude'].max().values
	bottom = data.coords['latitude'].min().values
	top = data.coords['latitude'].max().values
	coordinates = [[left, bottom],
               [right, bottom],
                [right, top],
                [left, top]]
	# coordinates = [[left, top],
 #               [right, top],
 #                [right, bottom],
 #                [left, bottom]]

	traces = []
	# mapbox
	mapbox = go.Scattermapbox()
	traces.append(mapbox)

	# layout
	margin = go.layout.Margin(l=0, r=0, b=0, t=0)
	layout = go.Layout(height=height,
	                   font={'size': 14},
	                   margin=margin,
	                   mapbox_style="carto-darkmatter",
	                   mapbox_layers=[{
	                       "sourcetype": "image",
	                       "source": img,
	                       "coordinates": coordinates
	                   }],
	                   mapbox_center={"lat": lat_c, "lon": lng_c},
	                   mapbox_zoom=zoom
	                  )
	fig = go.Figure(data=traces, layout=layout)
	return fig


def create_empty_mapbox(lat_c, lng_c, zoom, height=400):
	mapbox = go.Scattermapbox(mode="markers+lines", hoverinfo='all')
	# layout
	margin = go.layout.Margin(l=0, r=0, b=0, t=0)
	layout = go.Layout(#height=height,
						autosize=True,
	                   margin=margin,
	                   mapbox_style="carto-darkmatter",
	                   mapbox_center={"lat": lat_c, "lon": lng_c},
	                   mapbox_zoom=zoom
	                  )
	fig = go.FigureWidget(data=[mapbox], layout=layout)
	return fig

#######################################################
# UPDAte GRAPHS callbacks
#######################################################
# Totalresponse time HISTOGRAM
@app.callback(
	Output('top_right_bottom', 'figure'),
	Output('top_right_top', 'figure'),
	Input('cur_df_trig', 'children')
	)
def update_hist(inputs: str) -> go.Figure:

	# Get the data
	df, label = filter_df(inputs)

	# get the histogram data
	data = create_hist_data(df, label=label, limits=[0, 120])

	# Graph labels define
	labels = {'totalresponsetime': {'xlabel': 'Total Response Time (minutes)',
									'ylabel': 'Count'},
				'time_on_scene': {'xlabel': 'Time on Scence (minutes)',
									'ylabel': 'Count'}}
	figs = []
	for key, values in data.items():
		# Create figure
		if len(values['count'])==0: # if empty dataframe, skip
			figs.append(make_empty_plot())
		else:
			figs.append(create_hist_fig(values['x'], values['count'],
									xlabel=labels[key]['xlabel'],
									ylabel=labels[key]['ylabel']))
	return figs

# TIME SERIES graph
@app.callback(
	Output('bottom_left', 'figure'),
	Input('cur_df_trig', 'children'),
	Input('freq_select', 'value')
	)
def update_time_graph(inputs: str, freq: str) -> go.Figure:

	# Get selected data
	df, label = filter_df(inputs)

	# need start_date and end_date to chop agg data.
	inputs = ast.literal_eval(inputs)
	start_date = inputs[-2]
	end_date = inputs[-1]
	# Prepare data
	agg = 'count'
	dt_col = 'call_timestamp_EST'
	agg_col = 'totalresponsetime'
	x, y, z = create_time_data(df, label=label, start_date=start_date, end_date=end_date, 
		dt_col=dt_col, agg_col=agg_col, agg=agg, freq=freq)	
	return create_time_fig(x=x, y=y, z=z, xlabel='Time', ylabel=agg, height=350)

# Category graph
@app.callback(
	Output('bottom_right', 'figure'),
	Input('cur_df_trig', 'children')
	)
def update_bar_graph(inputs: str) -> go.Figure:

	# Get selected data
	df, label = filter_df(inputs)

	# prepare data
	agg = 'count'
	cat_col = 'category'
	agg_col = 'incident_id'
	data = create_bar_data(df, label=label, cat_col=cat_col, agg_col=agg_col, agg=agg)
	# Only keep the top 20 categories
	# x = x[-10:]
	# y = y[-10:]
	show_only = 10
	return create_bar_fig(x=data[agg][-show_only:],
						y=data[cat_col][-show_only:],
						hover=data['hover'][-show_only:],
						title='Top 10 Categories', xlabel=agg)


# MAP
@app.callback(
	Output('calls_map', 'figure'),
	Input('cur_df_trig', 'children'),
	Input('geounit_select', 'value'),
	Input('heatmap_settings', 'data'),
	#prevent_initial_call=True
	)
def update_map(inputs: str, geounit, heatmap_settings):

	# Get data
	df, label = filter_df(inputs)

	# map height (px)
	height = 400

	# Dict to hold selection info
	# select = {'nhood': {'geo_col': 'neighborhood',
	# 					'unit_geo': nhoods_geo,
	# 					'featureidkey': 'properties.nhood_name',
	# 					'json_col': 'nhood_name'},
	# 		'block': {'geo_col': 'block_id',
	# 					'unit_geo': blocks_geo,
	# 					'featureidkey': 'properties.GEOID10',
	# 					'json_col': 'GEOID10'}}

	# Datadasher heatmap
	if geounit=='lng_lat':
		# Data must not have NaN in long and lat for datashader to work.
		data = create_heatmap_data(df, label=label, heatmap_limits=heatmap_settings['limits'])
		return create_heatmap_fig(data, lng_c=heatmap_settings['lng_center'], 
									lat_c=heatmap_settings['lat_center'], 
									height=height,
									zoom=heatmap_settings['zoom'])

	# Check the current zoom level: If over threshold
	# - Get only datapoint WITHIN map viewport GPS.
	# - Then find the unique block_ids so that only render geojson of these blocks
	if heatmap_settings['zoom'] > 13.3:

		geounit = 'block'

		# get datapoints inside longitude window
		temp_df = df.copy()
		temp_df.select((temp_df['longitude'] >= heatmap_settings['limits'][0][0]) & (temp_df['longitude'] <= heatmap_settings['limits'][0][1]),
			mode='and')
		# get datapoints inside latitude window
		temp_df.select((temp_df['latitude'] >= heatmap_settings['limits'][1][0]) & (temp_df['latitude'] <= heatmap_settings['limits'][1][1]),
			mode='and')

		# blocks in mapviewport
		# To improve performance, only render selected boundary inside map viewport
		blocks_in_view = temp_df['block_id'].unique(selection=True)
		agg_df = temp_df.groupby(temp_df['block_id'], agg={'count': vaex.agg.count(selection=True)})
		# min max for color scale in view
		color_min, color_max = agg_df.minmax('count')

		data = create_choropleth_blocks_data(df, label=label, cat_col='block_id')

		geo_pd_small = blocks_geo[blocks_geo['GEOID10'].isin(blocks_in_view)].to_json() # this give json STRING actually.
		geo_pd_small = ast.literal_eval(geo_pd_small)

		return create_choropleth_map(geo_array=data['block_id'],
								z=data['count'],
								geojson=geo_pd_small,
								featureidkey='properties.GEOID10',
								lng_c=heatmap_settings['lng_center'],
								lat_c=heatmap_settings['lat_center'],
								zoom=heatmap_settings['zoom'],
								height=height,
								color_min=color_min,
								color_max=color_max,
								hover_label='Block')
	else:
		# geomap
		data = create_choropleth_nhoods_data(df, label=label, cat_col='neighborhood')
		# REMOVE neighoborhood name 'unknown'
		try:
			data = data[data['neighborhood']!='unknown']
		except:
			pass
		# select type of boundaries to plot
		return create_choropleth_map(geo_array=data['neighborhood'],
									z=data['count'],
									geojson=nhoods_geo, # need GEOJSON here
									featureidkey='properties.nhood_name',
									lng_c=heatmap_settings['lng_center'],
									lat_c=heatmap_settings['lat_center'],
									zoom=heatmap_settings['zoom'],
									height=height,
									hover_label='Neighborhood')


# When user interact with heatmap (zoom or pan), trigger settings update.
@app.callback(
	Output('heatmap_settings', 'data'),
	Input('calls_map', 'relayoutData'), 
	State('heatmap_settings', 'data'),
	prevent_initial_call=True
	)
def update_agg_limits(relayoutData: dict, heatmap_settings: dict) -> dict:
	# print('update_agg_limist: ')
	# print(relayoutData)

	# New derived viewport coordinates
	if 'mapbox._derived' in relayoutData:
		heatmap_settings['limits'] = viewport_to_heatmap_limits(relayoutData['mapbox._derived']['coordinates'])
		heatmap_settings['lat_center'] = relayoutData['mapbox.center']['lat']
		heatmap_settings['lng_center'] = relayoutData['mapbox.center']['lon']
		heatmap_settings['zoom'] = relayoutData['mapbox.zoom']
	# If map was reset to home
	elif 'mapbox.center' in relayoutData:
		heatmap_settings['lat_center'] = relayoutData['mapbox.center']['lat']
		heatmap_settings['lng_center'] = relayoutData['mapbox.center']['lon']
		heatmap_settings['zoom'] = relayoutData['mapbox.zoom']
		heatmap_settings['limits'] = heatmap_limits_initial
	# New zoom value. When only zoom was performed
	elif 'mapbox.zoom' in relayoutData:
		zoom_new = relayoutData['mapbox.zoom']
		#ratio = zoom_new / heatmap_settings['zoom']
		heatmap_settings['zoom'] = zoom_new
		# width (longitude delta) as a function of zoom (for this particular viewport pixel)
		width = 1084 * np.exp(-0.693 * heatmap_settings['zoom'])
		# width (longitude delta) as a function of zoom (for this particular viewport pixel)
		height = 430 * np.exp(-0.696 * heatmap_settings['zoom'])

		left = heatmap_settings['lng_center'] - width/2
		right = heatmap_settings['lng_center'] + width/2
		bottom = heatmap_settings['lat_center'] - height/2
		top = heatmap_settings['lat_center'] + height/2
		heatmap_settings['limits'] = [[left, right], [bottom, top]]
	else:
		pass

	# print(heatmap_settings)
	return heatmap_settings



@app.callback(
	Output('test', 'children'),
	Input('calls_map', 'relayoutData'),
	State('calls_map', 'figure'),
	prevent_initial_call=True # this is needed since there is no figure initially.
	)
def testing(relayoutData, figure):
	#print(relayoutData)
	#print(figure['layout']['mapbox']['layers'][0]['coordinates'])
	return 



# ##############################################################
# HELPER FUNCTIONS
# ##############################################################
def viewport_to_heatmap_limits(viewport):
	'''
	viewport: List of list of pair. [ [_, _], [_, _], [_, _], [_, _]]
	'''
	return [[viewport[0][0], viewport[1][0]],
			[viewport[2][1], viewport[1][1]]]



# ################################################################
# INITIAL SETTINGS 
# ################################################################

# center of map (Detroit)
lat = 42.355753
lng = -83.076760
zoom = 9.5
mapbox_empty = create_empty_mapbox(lat_c=lat, lng_c=lng, zoom=zoom)

# Heatmap starting point
#heatmap_limits_initial = [[-83.305680, -82.896303], [42.239087, 42.471065]]
_ = [[-83.45498919028684, 42.49910757691302], [-82.70629931683517, 42.49910757691302], [-82.70629931683517, 42.21207066837576], [-83.45498919028684, 42.21207066837576]]
# Starting map viewport GPS
viewport_init = [[-83.45110493672583, 42.49910757691339], [-81.95372518982248, 42.49910757691339], [-81.95372518982248, 41.9237233747794], [-83.45110493672583, 41.9237233747794]]
#viewport_init = [[-83.46, 42.51], [-82.69, 42.51], [-82.69, 42.20], [-83.46, 42.20]]
# for vaex count limits
heatmap_limits_initial = viewport_to_heatmap_limits(viewport_init)
#print(heatmap_limits_initial)
heatmap_settings = {'limits': heatmap_limits_initial,
					'zoom': zoom,
					'lat_center': lat,
					'lng_center': lng}


# last date in dataset
# type string
t_str = str(df_original.evaluate(df_original['call_timestamp_EST'], i1=len(df_original)-1)[0])[0:10]
last_date = datetime.datetime.strptime(t_str, '%Y-%m-%d')


# Frist date in dataset
# type string
t_str = str(df_original.evaluate(df_original['call_timestamp_EST'], i2=0)[0])[0:10]
first_date = datetime.datetime.strptime(t_str, '%Y-%m-%d')


###############################################################
# LAYOUT
###############################################################
# App layout (Dash components all go inside layout)
app.layout = html.Div(className = 'row container',
	#style={"backgroundColor": colors["background"], "pad": 0},
	children = [
		# GRID
		# Based on external css. Check css doc link above for more info.
		# tripgger to indicate if cur_df has been updated.(HIDDEN)
		# value of the trigger can be anything
		html.Div(id='cur_df_trig', children='', hidden=True),
		html.Div(id='test', hidden=True),
		# STore some settings
		dcc.Store(id='heatmap_settings', data=heatmap_settings),
		dcc.Store(id='mapbox_zoom', data=zoom),
		# Filters (left side)
		html.Div(className='two columns container pretty_container',
			children=[
				html.H4('Filters'),
				# Display amount of data selected
				html.Div('Data selected:'),
				html.Div(id='data_selected'),
				html.Progress(id='selected_progress', max=str(len(df_original)), value=str(len(df_original))),
				# DROP DOWN FILTERS
				html.Div(dropdown_filters),
				# DATE RANGE PICKER
				dcc.DatePickerRange(id='call_timestamp_EST',
						        min_date_allowed=datetime.datetime(2016, 9, 20),
						        initial_visible_month=datetime.datetime(2021, 4, 1),
						        max_date_allowed=last_date+datetime.timedelta(days=1),
						        start_date_placeholder_text="Start date",
	    						end_date_placeholder_text="End date",
	    						minimum_nights=0,
	    						clearable=True),
				# TIME SLIDER FILTERS
				html.Div(slider_filter_units)
				]),

		# Graphs (right side)
		html.Div(className='ten columns container pretty_container',
			children=[
				#title
				html.H3(f"Detroit City 911 Call Records: {first_date.strftime('%b')} {first_date.day}, {first_date.year} - {last_date.strftime('%b')} {last_date.day}, {last_date.year}",
					style={"text-align": "center"}),
				# TOP ROW
				html.Div(className='row twelve columns', style={'height': '450px'}, children=[
					# Choropleth map
					html.Div(className='eight columns', children=[
						dcc.RadioItems(id='geounit_select',
										options=[
											{'label': ' Neighborhood & City block ', 'value': 'nhood'},
											#{'label': ' City block', 'value': 'block'},
											{'label': ' Individual', 'value': 'lng_lat'}
										],
										value='lng_lat',
										labelStyle={'display': 'inline-block', 'margin-right': '10px'},
										style={'text-align': 'center', 'font-weight': 'bold'}),
						dcc.Graph(id="calls_map")
						]),

					# Histogram
					html.Div(className='four columns', children=[
						dcc.Graph(id="top_right_top", figure={}),
						dcc.Graph(id="top_right_bottom", figure={}),
						]),
					]),
				
				# BOTTOM ROW
				html.Div(className='row twelve columns', children=[
					# Time series graph
					html.Div(className='eight columns', children=[
						dcc.RadioItems(id='freq_select',
										options=[
											{'label': ' Hourly  ', 'value': 'h'},
											{'label': ' Daily  ', 'value': 'D'},
											{'label': ' Weekly', 'value': 'W'}
										],
										value='D',
										labelStyle={'display': 'inline-block', 'margin-right': '10px'},
										style={'text-align': 'center', 'font-weight': 'bold'}),
						dcc.Graph(id="bottom_left", figure={})
						]),

					# Bar graph
					dcc.Graph(className='four columns', id="bottom_right", figure={}),
					])
				
			])
	])

# ------------------------------------------------
if __name__ == "__main__":
	app.run_server(host='0.0.0.0', debug=True)