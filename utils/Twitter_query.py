import time
from apscheduler.schedulers.blocking import BlockingScheduler
import numpy as np
import pandas as pd
import os # to make directory
# Information about tweepy
# http://docs.tweepy.org/en/latest/api.html#api-reference
import tweepy # Twitter API made easy

# package to clear Jupyter Notebook screen.
# Only relevant when using Notebook, not as scripy.
from IPython.display import clear_output

class Twq:
    def __init__(self, secrets, geos, r, box, num):
        '''
        secrets: dict
            Twitter authentication information. 
            Dictionary with keys="key", "secret_key", "access_token", "access_token_secret"
        geos: list
            List of gps coordinates to filter tweets.
            Coordinates MUST be in string format of "lat,long"
        r: str
                Search radius. e.g. ("1mi")
        box: list of floats
                Bounding box. 2 gps set coords (long,lat,long,lat) of bounding box.
                Each bounding box should be specified as a pair of longitude and latitude pairs, 
                with the southwest corner of the bounding box coming first.
        num: int
        Trailing count for output csv file name. Use a new number to tell
        the program to write to a new file instead of appending to current one.
        '''
        # Increment of csv file to write to
        self.num = num
        # Twitter authentication
        auth = tweepy.OAuthHandler(secrets["key"], secrets["secret_key"])
        auth.set_access_token(secrets["access_token"], secrets["access_token_secret"])
        # GPS coords for used with SEARCH API
        self.geos = geos
        self.r = r
        # Boundary box for used with STREAM API
        self.box = box
        # SEARCH API
        self.search_api = tweepy.API(auth)
        # Raw tweets processor
        self.raw_data_proc = Raw_data_proc()
        # STREAM API
        self.stream_listener = My_stream_listener(self.raw_data_proc, self.box, self.num)
        self.stream = tweepy.Stream(auth=auth, listener=self.stream_listener)
        
        #######################
        # FOR DEBUGGING PURPOSE
        self.cur_status_lst = None
    
    def search(self, until=None):
        '''
        Search Twitter API for tweets from selected gps coordinates. This is REST API.
        NOT a live stream push API.
        Write processed data as dataframe in csv file and return the dataframe.
        --------------------------
        r: str
            Radius of search area. In mile ("mi") or km ("km")
            Example: "1mi" or "1km"
        until: str in format "YYYY-MM-DD"
            Search tweets PRIOR to this date
        --------------------------------
        return: dataframe
        '''
        # Print out local time for record tracking
        cur_time = time.localtime(time.time())
        print(f"Local time: {cur_time.tm_year}-{cur_time.tm_mon}-{cur_time.tm_mday}\t{cur_time.tm_hour}:{cur_time.tm_min}")
        
        results = []
        result_type = "recent"
        count = "100" # number of tweets to return
        tw_count = user_count = place_count = 0
        for geo in self.geos: # query each gps coordinates
            geocode = geo + "," + self.r
            tweets = self.search_api.search(geocode=geocode, result_type=result_type,
                                count=count, until=until)
            # Store current statuses list
            self.cur_status_lst = tweets
            if tweets: # if there are tweets returned.
                results.append(self.raw_data_proc.process_tweets(tweets, geocode))
                tw_count += results[-1]["tweet"].shape[0]
                user_count += results[-1]["user"].shape[0]
                place_count += results[-1]["place"].shape[0]
            time.sleep(0.5) # wait a bit before next API call.
        # print out summary to screen
        print(f"Tweets: {tw_count}\tUsers: {user_count}\tPlaces: {place_count}")
        print("-"*50)
        # Write to csv file and also return df
        df = self.raw_data_proc.write_df(results, num=self.num)

        return df
        
    def repeated_search(self, until=None, interval=1):
        '''
        Schedule run_process to automatically repeat every time interval (hour)
        '''
        scheduler = BlockingScheduler()
        scheduler.add_job(self.search, "interval", hours=interval, kwargs={"until": until})
        scheduler.start()
        
    def start_stream(self):
        '''
        Stream tweets within a boundary box.
        --------------------------------------------------
        return: none
        '''
        self.stream.filter(locations=self.box, is_async=True)
        
    def stop_stream(self):
        '''
        Stop the current live stream connection.
        '''
        self.stream.disconnect()
        

# FOR STREAM LIVE TWEETS
# override tweepy.StreamLister
class My_stream_listener(tweepy.StreamListener):
    def __init__(self, raw_data_proc, box, num):
        super().__init__()
        '''
        box: list of floats
                Bounding box. 2 gps set coords (long,lat,long,lat) of bounding box.
                Each bounding box should be specified as a pair of longitude and latitude pairs, 
                with the southwest corner of the bounding box coming first.
        num: int
                Trailing count for output csv file name. Use a new number to tell
                the program to write to a new file instead of appending to current one.
        '''
        # Raw tweets processor.
        self.raw_data_proc = raw_data_proc
        self.box = box
        self.box_str = ",".join([str(i) for i in self.box])
        self.num = num
        # FOR DEBUGGING PURPOSE
        self.cur_status = None
        self.status_count = 0
        
    def on_status(self, status):
        self.status_count += 1
        self.cur_status = status
        # Clear current Jupyter Notebook output
        clear_output(wait=True)
        print("Current stream session count:\t", self.status_count)
        # Process status.
        # Since process_tweets expects LIST, --> convert status to list
        dict_df_lst = self.raw_data_proc.process_tweets([status], self.box_str)
        # Since write_df expect list --> convert dict_df to list.
        self.raw_data_proc.write_df([dict_df_lst], self.num)
        return True # True to continue to connect. False breaks stream connection.        
    
    def on_error(self, status_code):
        if status_code==420: # if Twitter return error code
            print("-"*50)
            print("ERROR 420")
            print("-"*50)
            return False # to disconnect the stream.
        
        
# FOR PROCESSING RAW TWITTER DATA
class Raw_data_proc:
    def __init__(self):
        '''
        meta: dict
            Objects of a Twitter status. Object name as key, object's attributes as a list.
        '''
        # Tweet object attributes to gather.
        tw_atts = ["created_at", "id_str", "text", "user", "source", "truncated", "in_reply_to_status_id_str",
                  "in_reply_to_user_id_str", "is_quote_status", "retweet_count", "favorite_count",
                  "favorited", "retweeted", "lang", "coordinates", "place"]
        # Coordinates object (coordinates of tweet): "coordinates": list(lat, long)
        coord_atts = ["coordinates", "type"]
        # User object attributes: "user"
        user_atts = ["id_str", "name", "screen_name", "location", "url", "description", "protected", "verified",
                    "followers_count", "friends_count", "listed_count", "favourites_count", "statuses_count",
                    "created_at", "default_profile", "default_profile_image", "geo_enabled"]
        # place object: "place"
        place_atts = ["id", "url", "place_type", "name", "full_name", "country_code", "country", "bounding_box"]
        self.meta = {"tweet": tw_atts,
                    "coordinates": coord_atts,
                    "user": user_atts,
                    "place": place_atts}
        
    def make_df(self, obj, df, q_geo):
        '''
        obj: str
            Type of dataframe to create. e.g. tweet, coords or user.
            Object must valid and exist before calling this function. This function
            ASSUMES that the object exist in Twitter status and NOT None.
        df: dataframe
            Twitter status (one status) dataframe.
        q_geo: string
            Geolocation of query (lat, long, radius)
        -----------------------------------------------------
        return: DataFrame of ONE status (can be tweet or user or place)
        '''
        # query parameters
        q_time = pd.to_datetime(time.time(), unit="s")
        query_params = pd.DataFrame({"query_time": [q_time], "query_geo": [q_geo]})
        temp = None
        if obj=="tweet": # if tweet dataframe
            temp = df[self.meta[obj]].copy() # only take out selected features
            # check if tweet is geo-tagged.
            if ~temp["coordinates"].isna().any(): # if coordinates are included
                # store the gps coordinates
                temp.loc[0, "coordinates"] = str(temp.loc[0, "coordinates"]["coordinates"])
            # Keep user_id only
            temp.loc[0, "user"] = temp.loc[0, "user"]["id_str"]
        else: # other type of dataframe
            temp = pd.DataFrame.from_dict(df.loc[0, obj], orient="index").transpose()
            temp = temp[self.meta[obj]]
        return pd.concat([query_params, temp], axis=1)
    
    def process_tweets(self, tweets, q_geo):
        '''
        Process one Twitter search (one location) result (list of tweets)
        ----------------------------------
        tweets: list
                List of tweets (raw_format)
        q_geo: str
                Query geo. Only for record keeping purpose.
        return: dict
            Dictionary of dataframes. Object type as keys (i.e. tweet, user, place)
        '''
        dict_df = {} 
        for obj in ["tweet", "user", "place"]:
            temp = [] # temporary list to hold dataframes
            for tweet in tweets:
                df = pd.DataFrame.from_dict(tweet._json, orient="index").transpose()
                # check if status object is valid
                if obj=="tweet": 
                    temp.append(self.make_df(obj, df, q_geo))
                else: # if obj is not tweet
                    # if obj is NOT NONE
                    if ~df[obj].isna().any():
                        temp.append(self.make_df(obj, df, q_geo))
            if temp:
                dict_df[obj] = pd.concat(temp, axis=0, ignore_index=True)
            else: # if list is empty
                dict_df[obj] = pd.DataFrame() # empty dataframe
        return dict_df
    
    def write_df(self, lst, num):
        '''
        lst: list
                List of tweets DICTIONARY dataframes, one for each gps coordinate search.
        num: int
                Trailing count for output csv file name. Use a new number to tell
                the program to write to a new file instead of appending to current one.
        '''
        # check if "data" directory exist
        if not os.path.exists("data"):
            os.makedirs("data")
        # loop through the result dataframes
        for dict_df in lst:
            for key, df in dict_df.items():
                # if dataframe is not empty
                if not df.empty:
                    try:
                        # if file already exist
                        pd.read_csv(f"data/{key}{num}.csv", sep=",", nrows=1)
                        # don't rewrite column headers
                        df.to_csv(f"data/{key}{num}.csv", mode="a", sep=",", header=False, index=False)
                    except:
                        # if file NOT exist. Then write column header
                        df.to_csv(f"data/{key}{num}.csv", mode="a", sep=",", header=True, index=False)