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

- Models:
    - no version: use MSE and MAPE for optimization
    - 3h_FID...v1: use MAE for optimization

- Requirements:
	- Use GPU tensor-flow