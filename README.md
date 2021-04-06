<h1 style="text-align: center;">
	<b>Predictive Police Patrolling</b>
</h1>

- ## **Watch-me video:**
    - Don't want to read? Watch this 3-minute summary video instead.<br>

<div style="padding-left:30%; padding-right:30%">
    <a href="https://df-project-videos.s3.us-east-2.amazonaws.com/Luan_Nguyen_forecast_police_attention_level.mp4" target="_blank">
    	<img src="https://df-project-videos.s3.us-east-2.amazonaws.com/Forecast_police_attention_level.png" class="center">
    </a>
</div>

- ## **Introduction**
	- ### **Problem Statement**
		The goal of this work is to forecast daily police attention level needed for each neighborhoods in Detroit city, Michigan. The forecast is aim at 24 hours with 3-hour increments.

	- ### **Background**
		Given the increased in number of violent activities recent and how public safety has always been a concern in the community, police patrol is crucial in ensuring public security. In the past, police patrol with random routes. However, this is highly inefficient because the number of available police officers is limited. More recent approach is to use crime hot spots to plan patrol route. However, crime hot spots are based on static information, such as previous year crime record or demographics. This does not address the suggested dynamic nature of emergency need. The work in this report aim at providing a more dynamic forecast (daily in 3-hour intervals)to help police office plan their routes effectively.

- ## **Setup:**
	- ### `setup` directory contains the environment setup codes as follows:
		- `conda.yaml`: setup conda environment.
		- `nvidia_cuda_supports.sh`: OPTIONAL. This setup the necessary nvidia drivers for tensorflow GPU.
	- ### **Required packages:**
		- `shapely`: For working with neighborhood boundary polygons.
		- `ast`: Abstract syntax trees. Used to process trees of the Python abstract syntax grammar.
		- `tweepy`: Python library for accessing the Twitter API. Used for gathering tweets in 04_Extended_work.ipynb.
	- ### **Optional packages:**
		- `tensorflow-gpu 2.2.0`: To use GPU for neural net training.

- ## **Notebooks:**
	- `01_Data_gathering.ipynb`: Notebook to gather 911 calls, and neighborhood bounding box.
	- `02_EDA_Cleaning_Engineering.ipynb`: This includes exploratory data analysis, data cleaning, and feature engineering.
	- `03_Extended_work.ipynb`: Notebook contains different models used for forecasting police attention level needed.
	- `04_Extended_work.ipynb`: Gathering of gps-tagged tweets. Attempted to use gps-tagged tweets counts as additional variable for forecasting of needed police attention level.

- ## **Python scripts**
	- ### **utils**
		- `dash_app.py`: Build interactive Plotly-dash app to visualize the  forecast as choropleth map.
		- `Detroit_gps.py`: Contain gps coordinates of several points spread out in Detroit city. This is used for tweets gathering purpose.
		- `neighborhoods.py`: Detroit neighborhood class. To be imported. This is used to create neighborhoods object which contain information of neighborhood in Detroit.
		- `preprocessing.py`: Data preprocessing class. To be imported. This is used to create data processing object which contain data processing function.
		- `Twitter_query.py`: Tweets gathering class. To be imported. Gather tweets using rest API or live stream method.
		- `tweets_stream_script.py`: Script to setup live stream tweets which contain gps from Detroit city.
	- ### **analysis**
		- `Analysis.py`: Contains all custom functions for model building and analysis in this study.
		- `TSData.py`: Used to build data object for this study. Data object contains the time-series data for all neighborhoods and all train-validation-test splits.

- ## **Data:**
	- Google Drive: https://drive.google.com/drive/folders/1w9fi_HiFsuB-IEVFP4hwfN6q9jEIL2sx?usp=sharing
	- Detroit_911_calls
		- 911_Calls_For_Service.csv: 911 calls from Sep 2016 to Sep 2020
		- 911_Calls_2020_file\*.csv: 2020 calls

- ## **Models:**
	Contains pre-trained models. Note that these pre-trained models were produced using `tensorflow-gpu` and hence only work with `tensorflow-gpu`.
    - no version: use MSE and MAPE for optimization
    - 3h_FID...v1: use MAE for optimization

