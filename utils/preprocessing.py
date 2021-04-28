import pandas as pd
import pathlib
import ast
import vaex

class Preprocess:        
    def __init__(self, city):
        '''
        city: neighborhoods.py object
                Object about a city neighborhoods. Including names, and boundaries.
        '''
        self.city = city
    
    #####################################################################
    def process_tweets(self, paths, file=None):
        '''
        paths: list
                List of tweets csv files to process.
        file: str
                Name of file to write cleaned data to.
                None = "data/tweets_cleaned.csv"
        ------------------------------------------------------
        return: clean dataframe, and also
                Write clean dataframe to csv file.
        '''
        def process1(path):
            '''
            Read in one tweets csv file and clean it.
            ------------------------------------------------
            path: str
                    Path of tweets csv file.
            -----------------------------------------------
            return: dataframe
            '''
            print(path)
            # read in raw tweets
            tweets = pd.read_csv(path)
            print("Raw shape: ", tweets.shape)

            # Keep non-duplicate (REMOVE DUPLICATES)
            tweets = tweets.loc[~tweets.duplicated(subset="id_str"),:]
            print("Unique tweets: ", tweets.shape)

            # drop tweets without GPS tag
            tweets.dropna(subset=["coordinates"], axis=0, inplace=True)
            tweets.reset_index(drop=True, inplace=True)

            # split coordinates into seperate longitude and latitude columns
            tweets[["lng", "lat"]] = pd.DataFrame(tweets["coordinates"].apply(lambda x: ast.literal_eval(x)).tolist())
            tweets.drop(labels="coordinates", axis=1, inplace=True)
            print("GPS tagged: ", tweets.shape)

            # add FID (neighborhood ID) to  tweets
            tweets["FID_nhood"] = tweets.apply(self.city.which_nhood,
                                               axis=1,
                                               df=True,
                                               lng="lng",
                                               lat="lat")

            # Drop tweets not in Detroit neighborhoods.
            # That is tweets which "FID_nhood" is NaN
            tweets.dropna(subset=["FID_nhood"], axis=0, inplace=True)
            print("Tweets in Detroit: ", tweets.shape)
            print("-"*50)

            # Keep only columns relevant to this study
            tweets = tweets[["created_at", "id_str", "text", "user", 
                             "source", "lng", "lat", "FID_nhood"]]
            
            # FILTER OUT BOTS, ADVERTISING AND TRAFFIC TWEETS
            # Keep tweets from these sources:
            keep_sources = ["Instagram", "Twitter for Android", "Twitter for iPhone", "Foursquare", "Tweetlogix", "Untappd"]
            # mask for index
            mask = tweets["source"].apply(lambda x: any(source in x for source in keep_sources))
            tweets = tweets.loc[mask,:]
            
            return tweets

        # loop through all tweets csv files:
        df = []
        for path in paths:
            df.append(process1(path))
        df = pd.concat(df, axis=0, ignore_index=True)

        # REMOVE DUPLICATES 2nd time
        df = df.loc[~df.duplicated(subset="id_str"),:]
        # Set created_at as datetime64
        df["created_at"] = df["created_at"].astype("datetime64")
        # Sort in ascending time order
        df.sort_values(by="created_at", axis=0, ascending=True, ignore_index=True, inplace=True)
        # set created_at as index
        df.set_index("created_at", inplace=True)

        # WRITE CLEAN DATAFRAME TO CSV FILE
        if file==None:
            file = "data/tweets_cleaned.csv"
        df.to_csv(file, mode="w", header=True, index=True)
        print("Keep: ", df.shape)
        print("Write to: ", file)
        return df
           
    ###################################################################    
    def process_calls(self, paths, out_file=None):
        '''
        paths: list
                List of 911 calls csv files to process.
        file: str
                Name of file to write cleaned data to.
                None = "data/Detroit_911_calls/911_Calls_2020_cleaned.csv"
        ------------------------------------------------------
        return: clean dataframe and also
                Write clean dataframe to csv file.
        '''
        def process2(path):
            '''
            Read in one 911 calls csv file and clean it.
            ------------------------------------------------
            path: str
                    Path of 911 calls csv file.
            -----------------------------------------------
            return: dataframe
            '''
            print(path)
            # Read in calls
            calls = pd.read_csv(path, thousands=",")
            print("Raw shape: ", calls.shape)

            # Drop calls initiated by police
            #mask = calls[calls.officerinitiated=="Yes"].index
            #calls.drop(index=mask, inplace=True)

            # Drop calls outside of Detroit neighborhood
            calls.dropna(axis=0, subset=["neighborhood"], inplace=True)

            # Strip leading & trailing whitespaces of object columns
            for col in calls.select_dtypes(include="object").columns:
                calls[col] = calls[col].str.strip()

            # Drop calls in these categories as not related to this public safety
            cags_to_drop = ["HNGUP", "STRTSHFT", 
                            "REMARKS", "BUS BRD", "TOW"]
            mask = calls["category"].apply(lambda x: x not in cags_to_drop)
            calls = calls.loc[mask,:]

            # drop priority="P" or "".
            # These calls are very few and the meaning of "P" is not understood
            mask = calls[(calls.priority=="P") | (calls.priority=="")].index
            calls.drop(index=mask, inplace=True)
            
            # KEEP ONLY COLUMNS RELEVANT TO THIS STUDY
            calls = calls[["call_timestamp", "incident_id", "officerinitiated", 
                            "priority", "calldescription", "category", 
                            "neighborhood", "longitude", "latitude"]]
            
            # convert call_timestamp to datetime64
            # convert priority to int8
            calls = calls.astype({"call_timestamp": "datetime64",
                                 "priority": "int8"})

            print("Keep: ", calls.shape)
            print("-"*50)
            
            return calls
        
        # loops through all calls csv files:
        df = []
        for path in paths:
            df.append(process2(path))
        df = pd.concat(df, axis=0, ignore_index=True)
        # REMOVE DUPLICATES
        df = df.loc[~df.duplicated(subset="incident_id"),:]
        # Re-label the calls' neighborhood Oak Grove since 2 neighborhoods has the same name
        df["FID_nhood"] = df.apply(self.city.which_nhood,
                                   axis=1,
                                   df=True,
                                   lng="longitude",
                                   lat="latitude",
                                   nhood=["Oak Grove"])
        # Drop calls without neighborhood (after neighborhood relabeling)
        df.dropna(axis=0, how="any", subset=["FID_nhood"], inplace=True)
        # Sort call_timestamp in ascending order
        df.sort_values(by="call_timestamp", axis=0, 
                      ascending=True, inplace=True,
                      ignore_index=True)
        # Set call_timestamp as index
        df.set_index("call_timestamp", inplace=True)
        
        # WRITE CLEAN DATAFRAME TO CSV FILE
        if out_file==None:
            out_file = "data/Detroit_911_calls/911_Calls_2020_cleaned.csv"
        df.to_csv(out_file, mode="w", header=True, index=True)
        print("Write to: ", out_file)
        #return df
        
    ###############################################################################
    # Group by time interval by neighborhood.
    # Dictionary of neighborhoods. Each neighborhood is a timeseries dataframe
    def time_groupby(self, df, col, agg="count", nhoods="all", freq="180Min"):
        '''
        Return a dataframe of rows **UNIQUE COUNT** per each time interval (index)
        for each NEIGHBORHOOD (column)
        -------------------------------------------------------------------
        df: dataframe
                Time series dataframe (e.g. calls, tweets) for all neighborhoods. 
                Index must be of DateTime type.
        col: str
                Column name to unique count by.
        agg: str
                Aggregation method: "count" or "avg"
        nhoods: "all" or list
                "all": time grouping for all neighborhood.
                List of neighborhoods: time grouping for selected neighborhoods.
        freq: str (e.g. "180Min")
                Time interval (in minutes) to groupby.
        -------------------------------------------------------------------
        return: dataframe
        '''
        grouped = {}
        if nhoods=="all":
            nhoods = df["FID_nhood"].unique()
        for nhood in nhoods:
            # filter by neighborhood
            temp = df[df["FID_nhood"]==nhood]
            if agg=="count":
                # groupby time interval (count of UNIQUE only).
                grouped[nhood] = temp.groupby(pd.Grouper(freq=freq,
                                                         base=0,
                                                         origin="start_day")).nunique()[col]
            elif agg=="avg":
                # aggregate by average
                grouped[nhood] = temp.groupby(pd.Grouper(freq=freq,
                                                         base=0,
                                                         origin="start_day")).mean()[col]
            else:
                print("Aggregation method must be: 'count' or 'avg'.")
                return
        # Convert the dict of dataframes to one dataframe
        grouped = pd.DataFrame.from_dict(grouped)
        # fill nan with zero
        grouped.fillna(0, inplace=True)
        return grouped

    # Prepare working dataframe (for ONE neighborhood):
    # index: datetime
    # columns: calls, calls_adj, tweets, tweets_adj, users, users_adj
    def make_working_df(self, dfs, labels, nhood, freq="180Min"):
        '''
        dfs: list
                List of time series dataframe
        labels: list
                List of dataframe label string. i.e. "calls", "tweets", or "users"
        nhood: str
                Neighborhood (FID,name) to make dataframe for.
        freq: str (e.g. "180Min")
                Time interval to groupby.
        --------------------------------------------------------
        return: dataframe
        '''        
        configs = {"calls": "incident_id",
                   "tweets": "id_str",
                   "users": "user"}
        temp = []
        cols = []
        for i in range(len(dfs)):
            # get time_groupby data for selected nhood
            temp.append(self.time_groupby(df=dfs[i],
                                          col=configs[labels[i]],
                                          nhoods=[nhood],
                                          freq=freq))
            # get time_groupby data for adjacent nhoods
            temp.append(self.time_groupby(df=dfs[i],
                                          col=configs[labels[i]],
                                          nhoods=self.city.get_adjacent(nhood),
                                          freq=freq).sum(axis=1))
            cols += [labels[i], labels[i]+"_adj"]
        df = pd.concat(temp, axis=1, join="outer")
        # FILL NAN
        df.fillna(0, inplace=True)
        df.columns = cols
        return df.astype("uint32")

# Processing function of dashboard data
def process_data_dashboard(years):
    '''
    Pre-process 911 calls data (from seperate raw csv files) for dashboard_eda app.
    -----------------------------------------------------------------
    years: list of int. Years of 911 call data to process.
    ------------------------------------------------------------------------
    return: None. Write clean data to csv file.
    '''
    # Select years of data to include
    files = [] # list of data files
    for year in years:
        # define the path
        current_directory = pathlib.Path('../data/raw/Detroit_911_calls/')
        # define the filename pattern
        name_pattern = f"911_Calls_{year}*.csv"
        # Add match files to the files list
        files += [str(x) for x in current_directory.glob(name_pattern)]
    # Print out list of files
    print('Reading 911 calls files ...')
    [print(file) for file in files]
    df = pd.DataFrame()
    for file in files:
        df = pd.concat([df, pd.read_csv(file, thousands=",", dtype={'priority':'str'})], axis=0)


    # FORMAT and CLEAN UP a bit
    # Drop duplicates (using incident_id) if exists
    df.drop_duplicates(subset=["incident_id"], inplace=True, ignore_index=True)

    # the current call_timestamp is timezone aware, e.g. '2021-04-06 13:00:00-05:00'
    # I will keep only the naive portion, as in EST time as '2021-04-06 13:00:00'
    df['call_timestamp_EST'] = df['call_timestamp_EST'].apply(lambda x: x[:19]).astype('datetime64')

    # Drop unnecessary columns
    df.drop(columns=['oid', 'ObjectId', 'precinct_sca'], inplace=True)

    # clean up zip_code
    df['zip_code'] = df['zip_code'].apply(lambda x: 0 if x==' '*5 else x).astype('int')

    # strip leading and trailing white spaces of data in object type columns.
    for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.strip()

    # replace '' in priority to 'unknown'
    df['priority'] = df['priority'].apply(lambda x: 'unknown' if x=='' else x)

    # clean block_id
    df['block_id'] = df['block_id'].fillna('unknown').astype('str')

    df['block_id'] = df['block_id'].apply(lambda x: x[:15]) # Don't keep the decimal portion of the number string.

    df['neighborhood'].fillna('unknown', inplace=True) # fill unknown neighborhood.


    # Add a category detail, which contain more descriptive info of the category abbreviation
    # Mapping: key: category abbreviation (in dataframe)
    # value: descriptive information

    # write to csv file
    df.to_csv('../data/processed/911_Calls_for_dashboard.csv', mode="w", header=True, index=False)

# Group data by time.
# NOTE this is different than the time_groupby method of proprocessing class, which break down to neighborhood level.
def time_groupby(df, dt_col, agg_col, agg="count", freq="180Min"):
    '''
    Return a series of rows aggregated by : **UNIQUE COUNT** or average, per each time interval (index)
    -------------------------------------------------------------------
    df: dataframe. Must contain one datetime column as indicated by dt_col arg.
    dt_col: str. Column name of datetime to aggregate by.
    agg_col: str. Name of column to agg by.
    agg: str. Aggregation method: "count" or "avg"
    freq: str (e.g. "180Min"). Time interval (in minutes) to groupby.
    -------------------------------------------------------------------
    return: Series.
    '''
    # Prepare input dataframe to have datime index
    # Sort call_timestamp in ascending order
    # Note that this sorting step actually REDUCE the execution time overall, IMPROVE performance.
    df = df.sort_values(by=dt_col, axis=0, 
                  ascending=True, ignore_index=True)
    # Set call_timestamp as index
    df = df.set_index(dt_col)
    if agg=="count":
        # groupby time interval (count only).
        result = df.groupby(pd.Grouper(freq=freq,
                                     offset=0,
                                     origin="start_day")).count()[agg_col]
    elif agg=="avg":
        # aggregate by average
        result = df.groupby(pd.Grouper(freq=freq,
                                     offset=0,
                                     origin="start_day")).mean()[agg_col]
    else:
        print("Aggregation method must be: 'count' or 'avg'.")
        return
    # fill nan with zero
    result.fillna(0, inplace=True)
    return result

def time_groupby_vaex(df, dt_col, agg_col, selection=False, agg='count', freq='D'):
    '''
    Return a vaex dataframe of aggregration by selected time column and frequency.
    -------------------------------------------------------------------------------
    df: Vaex dataframe. Must contain one datetime column
    dt_col: str. Column name of datetime to aggregate by.
    agg_col: str. Name of column to agg by.
    agg: str. Aggregation method: "count" or "avg"
    freq: str. Time interval to groupby (e.g. "h", groupby hourly).
        Possible values: 'm', 'h', 'D', 'W', 'M' 
    -------------------------------------------------------------------
    return: Vaex dataframe with 2 columns: dt_col, agg

    '''
    agg_dict = {'count': vaex.agg.count,
                'avg': vaex.agg.mean,
                'first': vaex.agg.first,
                'max': vaex.agg.max,
                'min': vaex.agg.min,
                'nunique': vaex.agg.nunique}
    try:
        # Use .filter() first to remove non-selected time bins , somehow cause error.
        # Therefore, non-selected time bins will be kept with agg = 0
        return df.groupby(vaex.BinnerTime(df[dt_col], resolution=freq), agg={f'{agg}': agg_dict[agg](agg_col, selection=selection)})
    except KeyError:
        print(f'Aggregation method must be one of :{list(agg_dict.keys())}')
        return

# Group data by a category column
def cat_groupby(df, cat_col, agg_col, agg='count'):
    '''
    Group data by location column.
    -----------------------------------------
    df: dataframe
    cat_col: str. Column name to create category grouping.
    agg_col: str. Column label of aggregate value column.
    agg: str. Aggregrate function (e.g. 'count' or 'avg')
    --------------------------------------------
    return: Series.
    '''
    if df.empty:
        return df
    if agg=='count':
        result = df.groupby([cat_col]).count()[agg_col]
    elif agg=='avg':
        result = df.groupby([cat_col]).mean()[agg_col]
    else:
        print("Aggregation method must be: 'count' or 'avg'.")
        return
    # fill nan with zero
    result.fillna(0, inplace=True)
    return result

def cat_groupby_vaex(df, cat_col, agg_col, selection=False, agg='count') -> vaex.dataframe.DataFrameLocal:
    agg_dict = {'count': vaex.agg.count,
                'avg': vaex.agg.mean,
                'first': vaex.agg.first,
                'max': vaex.agg.max,
                'min': vaex.agg.min,
                'nunique': vaex.agg.nunique}
    try:
        # Use .filter() first to remove non-selected category
        return df.filter('default').groupby(cat_col, agg={f'{agg}': agg_dict[agg](agg_col)})
    except KeyError:
        print(f'Aggregation method must be one of :{list(agg_dict.keys())}')
        return