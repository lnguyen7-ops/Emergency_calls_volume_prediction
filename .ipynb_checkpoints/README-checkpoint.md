- WatchMe video:
    - Don't want to read, watch this 3-minutes summary video instead.
    
<video width="320" controls="true" allowfullscreen="true">
<source src="https://df-project-videos.s3.us-east-2.amazonaws.com/Luan_Nguyen_forecast_police_attention_level.mp4" type="video/mp4">
    Video not supported
</video>

- Notebooks:
	- 01_Data_gathering.ipynb	Notebook to gather 911 calls, neighborhood bounding box, and tweets.
	- 02_Main.ipynb				Main notebook. This include, data processing and all analysis and modeling
	- 03_Extended_work.ipynb	Notebook contain EDA analysis of tweets (extended study of this)
- Python scripts
	- Detroit_gps.py 			Contain gps coordinates of several points spread out in Detroit city. This is used for tweets gathering purpose.
	- neighborhoods.py 			Detroit neighborhood class. To be imported. This is used to create neighborhoods object which contain information of neighborhood in Detroit.
	- preprocessing.py 			Data preprocessing class. To be imported. This is used to create data processing object which contain data processing function.
	- Twitter_query.py 			Tweets gathering class. To be imported. Gather tweets using rest API or live stream method.
	- tweets_stream_script.py 	Script to setup live stream tweets which contain gps from Detroit city.	
- Data
	- Google Drive: https://drive.google.com/drive/folders/1w9fi_HiFsuB-IEVFP4hwfN6q9jEIL2sx?usp=sharing
	- Detroit_911_calls
		- 911_Calls_For_Service.csv: 911 calls from Sep 2016 to Sep 2020
		- 911_Calls_2020_file\*.csv: 2020 calls
- Models:
    - no version: use MSE and MAPE for optimization
    - 3h_FID...v1: use MAE for optimization

- Requirements:
	- Use GPU tensor-flow
