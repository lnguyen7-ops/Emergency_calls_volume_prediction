import pandas as pd
import numpy as np
import json
# used to merge json files
from jsonmerge import Merger
import requests
import datetime

def get_911_records_Detroit(date_start, date_end, last_30_days=False):
    '''
    Get 911 call records data from City of Detroit database.
    -----------------------------------------------------------
    date_start: str. Format example "'2020-11-30 15:00:00'" for Nov 30, 2021 15h:00m:00s LOCAL TIME. 
        NOTE THE USE OF QUOTATION MARKS FOR DATE.
        NOTE: The input time is understood as local time (of the sender) by default. However, the returned
        time formate is in UTC. Therefore, when doing time zone conversion, don't forget to account for
        DAT LIGHT SAVING in the US.
    date_end: str.
    last_30_days: bool. True: use the last-30-days API endpoints.
        False: use the main API endpoints,e.i. contains data since Sep 20, 2016.
    ----------------------------------------------------------------
    return: dataframe. No file will be writen to system.
    '''
    # initialize empty dataframe to hold results
    df = pd.DataFrame()
    num_page = 0 # count number of result pages.
    
    # API endpoint for City of Detroit
    if last_30_days:
        # last 30-day databse endpoint
        api_endpoint = 'https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services/911_Calls_for_Service_(Last_30_Days)/FeatureServer/0/query'
    else:
        # main database endpoint
        api_endpoint = 'https://opengis.detroitmi.gov/opengis/rest/services/PublicSafety/911CallsForService/FeatureServer/0/query'
    
    result_offset = 0
    next_page = True
    while next_page:
        # filter
        # GET request parameters
        # For a complete list of parameters, refer to API documentation
        payload = {'where': f"call_timestamp >= DATE {date_start} AND call_timestamp < DATE {date_end}", # WHERE clause in SQL
                   'outFields': '*', # output field. wildcard * means all.
                   'orderByFields': 'incident_id', # order query results since we will use pagination.
                   'resultOffset': str(result_offset),
                   #'resultRecordCount': '1000', # the maximum returned record can be 2000 as highest.
                   'exceededTransferLimit': 'true',
                   'resultType': 'standard',
                   'f': 'json'}
        print('-'*50)
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Sending request ...')
        r = requests.get(api_endpoint, params=payload)
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Received.')
        data = r.json()
        # if website error
        if r.status_code != 200:
            print('Errors from API')
            return df
        
        # if query error
        if 'error' in data:
            print('Error messages from ArcGIS API:')
            print('\n'.join(data['error']['details']))
            return df
        
        # if no data returned
        if len(data['features'])==0:
            return df
        
        # convert json to dataframe
        col_names = list(data['features'][0]['attributes'].keys())
        temp = pd.json_normalize(data, record_path=['features'])
        # drop geometry columns as dupplicated information
        temp.drop(columns=['geometry.x', 'geometry.y'], inplace=True)
        temp.columns = col_names # rename columns for cleaness
        df = pd.concat([df, temp], ignore_index=True)
        # convert call_timestamp into datetime type

        # check if there are more results left on next page
        # NOTE: "exceededTransferLimit": true --> indicates that there is still results left in the query. Use offset to get more results.
        # When there NO MORE results left, there is NO 'exceededTransferLimit' key in the returned JSON.
        if 'exceededTransferLimit' in data and data['exceededTransferLimit']==True:
            result_offset += temp.shape[0]
        else:
            next_page = False
        num_page += 1
    print('Number of records: ', df.shape[0])
    print('Number of pages: ', num_page)
    
    # convert call_timestamp to datetime
    df['call_timestamp'] = df['call_timestamp'].astype('datetime64[ms]')
    # convert naive to UTC
    df['call_timestamp'] = df['call_timestamp'].dt.tz_localize('utc')
    # convert UTC to US/Eastern since local time of Detroit.
    df['call_timestamp'] = df['call_timestamp'].astype('datetime64[ns, US/Eastern]')
    
    df.rename(columns={'call_timestamp': 'call_timestamp_EST'}, inplace=True)
    return df

def get_unit_boundary_Detroit(unit):
    '''
    Get unit (neighborhood or city block) boundaries data from City of Detroit databse.
    ------------------------------------------------------------------------
    unit: str. 'nhood' or 'block'
    -------------------------------------------------------------------
    return: geoJSON. No file/files will be written to system.
    '''
    if unit=='nhood':
        # Detroit neighborhoods GEOJSON API
        api_endpoint = 'https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services/Current_City_of_Detroit_Neighborhoods/FeatureServer/0/query'
    elif unit=='block':
        # Detroit city block GEOJSON API
        api_endpoint = 'https://services2.arcgis.com/HsXtOCMp1Nis1Ogr/arcgis/rest/services/DetroitBlocks2010/FeatureServer/0/query'
    else:
        print('Invalid unit. Use "nhood" or "block"')

    # initialize
    geojson = None
    # json merging schema, if there are more than one json file.
    # In this case, append the 'features' list when merging json file. Other keys will use overwritten method.
    schema = {"properties": {"features": {"mergeStrategy": "append"}}}
    merger = Merger(schema)

    num_page = 0 # count number of result pages.
    result_offset = 0
    next_page = True
    while next_page:
        payload = {'where': '1=1', # WHERE clause in SQL. Required by database
                   'outFields': '*', # output field. wildcard * means all.
                   'geometryType': 'esriGeometryPolygon',
                   'orderByFields': 'OBJECTID',
                   'resultOffset': str(result_offset),
                   'resultType': 'standard',
                   'f': 'geoJSON'}
        # Detroit neighborhoods polygons
        print('-'*50)
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Sending request ...')
        r = requests.get(api_endpoint, params=payload)
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Received.')
        data = r.json()
        # Check for errors
        if r.status_code != 200:
            print('Errors from API: ', r.status_code)
            return geojson

        # if query error
        if 'error' in data:
            print('Error messages from ArcGIS API:')
            print('\n'.join(data['error']['details']))
            return geojson

        # merge geoJSON file
        if geojson != None:
            geojson = merger.merge(geojson, data)
        else:
            geojson = data

        # check for next page
        if 'properties' in data:
            if 'exceededTransferLimit' in data['properties'] and data['properties']['exceededTransferLimit']==True:
                result_offset += len(data['features'])
            else:
                next_page = False
        else:
            next_page = False
        num_page += 1
    print('Number of records: ', len(geojson['features']))
    print('Number of pages: ', num_page)
    return geojson

def boundary_json_to_df(geojson):
    '''
    Convert geojson neighborhood boundary to dataframe, with some cleaning done.
    ---------------------------------------------------------------------------
    geojson: json or str.
        If str: file path to json file.
    ----------------------------------------------------------------------------
    return: dataframe
    '''
    if type(geojson)==str:
        with open(geojson) as f:
            geojson = json.load(f)
    df = pd.json_normalize(geojson['features'])
    # Drop unnecessary columns
    df.drop(columns=['type', 'id'], inplace=True)
    # clean up column names
    df.columns = [col.replace('properties.', '') for col in df.columns]
    return df