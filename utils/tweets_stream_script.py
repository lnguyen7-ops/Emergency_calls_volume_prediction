import pandas as pd
import numpy as np 
# authentication
import keys.my_api_secrets
# List of gps coordinates (radius=1 mile) covering Detroit
import utils.Detroit_gps
# Import tweets search class
import utils.Twitter_query




if __name__=="__main__":
	coords = Detroit_gps.coords_1mile
	r = "1mi"
	# Detroit bounding box
	box = Detroit_gps.box
	# **CSV FILE INCREMENT**
	num = 5
	# stream object
	detroit_tweets = Twitter_query.Twq(my_api_secrets.twitter_secrets, coords, r, box, num)
	# start stream
	detroit_tweets.start_stream()

