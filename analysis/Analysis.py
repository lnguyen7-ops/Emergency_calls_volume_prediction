# ANALYSIS:
# partail autocorrelation
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.api import tsa # time series analysis
# ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
# Plot
import matplotlib.pyplot as plt
# tensor math
import tensorflow as tf
# Keras network
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Flatten, BatchNormalization, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# For saving model
import joblib

# Calls priority data:
# low value ==> high priority
def get_att_level(data, nhoods, freq, data_proc):
    '''
    Generate attention level data for each neighborhood and selected time interval.
    ----------------------------------------------------
    data : DataFrame. 911 calls data
    data_proc : data processing object.
    nhoods : list of neighborhood or "all"
    freq : str. Time interval to groupby. e.g. "3H", "1D"
    '''
    count = data_proc.time_groupby(data, "incident_id", 
                                  agg="count", 
                                  nhoods=nhoods, 
                                  freq=freq)
    avg_priority = data_proc.time_groupby(data, "priority", 
                                          agg="avg", 
                                          nhoods=nhoods, 
                                          freq=freq)
    att_level = (count / (avg_priority+1)) + 1
    # plus 1 at denominator to avoid division by zero.
    # plus 1 again at the end to have min=1. This prevent MAPE error to be infinity
    # when the actual value = 0.
    return att_level

# PARTIAL autocorrelation
def get_pacf(data, max_lag, ret=False):
    abs_pacfs = list(abs(pacf(data, nlags=max_lag)))
    # find best lag (skip the lag=0)
    best_lag = abs_pacfs.index(max(abs_pacfs[1:]))
    print("Best lag:\t", best_lag)
    fig, (ax) = plt.subplots(1,1, figsize=(15,3))
    plot_pacf(data, lags=max_lag, ax=ax)
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation')
    plt.show()
    return abs_pacfs if ret else None

# DECOMPOSITION
def decomposition(data, lag, show="all"):
    '''
    show: str or float
            Portion of data shown on plot."all"=>show all.
    '''
    if show=="all":
        length = len(data)
    else:
        length = int(len(data)*show)
    decom = tsa.seasonal_decompose(data, model='additive', freq=lag)
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(15,12), sharex=True)
    ax0.set_ylabel("Observed")
    decom.observed[-length:].plot(marker=".", ax=ax0)
    ax1.set_ylabel("Trend")
    decom.trend[-length:].plot(ax=ax1)
    ax2.set_ylabel("Seasonal")
    decom.seasonal[-length:].plot(ax=ax2)
    ax3.set_ylabel("Residual")
    #plt.hlines(y=0, ax=ax3)
    decom.resid[-length:].plot(marker=".", markersize=10, linewidth=0, ax=ax3)
    plt.axhline(y=0, linestyle="--", color="black")
    plt.tight_layout()
    plt.show()

def arima_best(fh, train, val, p_range, d_range, q_range):
    '''
    fh : int. Forecast horizon. While validation set can be longer than
            the forecast horizon, only the fh portion of the validation set
            will be used to calculate score/loss, instead of forecasting the
            entire length of the validation set. This is to keep consistent with
            the actual use purpose of the model which will be to predict only
            the selected forecast horizon.
    p_range: tuple of 2
    d_range: tuple of 2
    q_range: tuple of 2
    '''
    # Hyperparameters tunning
    print("Tuning p, d, q:")
    print("-"*50)
    # true values to be scored again
    true = val[:fh]
    min_loss = float("inf")
    best_model = None
    best_p = best_d = best_q = None
    for p in range(*p_range):
        for d in range(*d_range):
            for q in range(*q_range):
                model = SARIMAX(train, order=(p,d,q), 
                        enforce_stationarity=False, 
                        enforce_invertibility=False,
                        trend=None).fit(maxiter=1000)
                # make prediction
                predictions = model.forecast(fh)
                loss = mean_squared_error(true, predictions)
                #loss = sMAPE(val, predictions)
                if loss < min_loss:
                    min_loss = loss
                    best_model = model
                    best_p = p
                    best_d = d
                    best_q = q
                    print(f"{p}, {d}, {q}: Validation MSE ", round(min_loss, 4), end="\r")
    print("-"*50)
    #return (best_p, best_d, best_q)
    return best_model, (best_p, best_d, best_q)
    

def rand_forest_reg_best(X_train, y_train, X_test, y_test, max_n_estimator, max_depth, fh=None):
    '''
    Return model with best tuned hyperparameter
    '''
    n_estimators = [50*i for i in range(1, int(max_n_estimator/50))]
    depths = [i for i in range(1,max_depth,2)]
    min_loss = float("inf")
    rfr_best = None # best model
    for n in n_estimators:
        for depth in depths:
            rfr = RandomForestRegressor(n_estimators=n, 
                                        max_depth=depth, 
                                        n_jobs=-1, 
                                        verbose=0,
                                        random_state=1).fit(X_train, y_train)
            # use only small portion of the test set to score due to the
            # function model_evaluate() is quite slow.
            loss = model_evaluate(model=rfr,
                                  fh=fh,
                                  X_test=X_test[:fh],
                                  y_test=y_test[:fh],
                                  metric=MAPE,
                                  input_3d=False)
            
            # Note that evaluate loss using the model_evaluate function appears to
            # make more sense as the function uses recursive_forecasting. However,
            # recursive_forcasting take some time to run and hence, I will use
            # the native predict method instead.
            #loss = MAPE(y_test, rfr.predict(X_test))
            if loss < min_loss:
                min_loss = loss
                print(f"{n}, {depth}, MAPE: ", round(loss, 4), end="\r")
                rfr_best = rfr
    return rfr_best

def build_dnn_straight(num_hidden_layers, nodes_per_hidden, 
                       num_output_nodes, input_shape, flatten=False):
    '''
    input_shape : [num_of_features]
    '''
    model = Sequential()
    if flatten:
        model.add(Flatten())
    else:
        model.add(InputLayer(input_shape=input_shape))
    for i in range(num_hidden_layers):
        model.add(Dense(nodes_per_hidden, activation="relu"))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
    # Output layer
    # Use relu because don't want to have negative(-) result.
    model.add(Dense(num_output_nodes, activation="relu"))
    return model

def build_dnn_pyramid(num_hidden_layers, nodes_per_hidden, 
                       num_output_nodes, input_shape, flatten=False):
    '''
    Dense network with pyramid scheme (wide-input => narrow-output).
    The shrink factor between each layer is 0.4 (i.e. 0.6 remained)
    -----------------------------------------------
    input_shape: [None, int] where int=number of features
    '''
    # calculate number of nodes for each layer
    keep = 0.6 # keep 60% number of nodes from previous layer
    num_nodes = [int(nodes_per_hidden*keep**i) for i in range(1,num_hidden_layers)]
    num_nodes = [nodes_per_hidden] + num_nodes
    # calculate dropout ratio for each layer
    drop_outs = [round(0.2*0.9**i, 2) for i in range(1,num_hidden_layers)]
    drop_outs = [0.2] + drop_outs
    
    model = Sequential()
    if flatten:
        model.add(Flatten())
    else:
        model.add(InputLayer(input_shape=input_shape))
    for i in range(num_hidden_layers):
        model.add(Dense(num_nodes[i], activation="relu"))
        model.add(Dropout(drop_outs[i]))
        model.add(BatchNormalization())
    # Output layer
    # Use relu because don't want to have negative(-) result.
    model.add(Dense(num_output_nodes, activation="relu"))
    return model

def build_rnn_straight(num_recurrent_layers, nodes_per_hidden, 
                       num_output_nodes, input_shape, flatten=None):
    '''
    input_shape: [None, int] where int=number of features
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(num_recurrent_layers):
        if i != num_recurrent_layers-1: # if NOT the last recurrent layer
            # return sequence
            model.add(LSTM(nodes_per_hidden, return_sequences=True, activation="relu"))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
        else: # If this is the last recurrent layer
            model.add(LSTM(nodes_per_hidden, return_sequences=False, activation="relu"))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
    # output
    model.add(Dense(num_output_nodes, activation="relu"))
    return model

def build_rnn_pyramid(num_recurrent_layers, nodes_layer1, 
                       num_output_nodes, input_shape, flatten=None):
    '''
    Recurrent network with pyramid scheme (wide-input => narrow-output).
    The shrink factor between each layer is 0.4 (i.e. 0.6 remained)
    -----------------------------------------------
    input_shape: [None, int] where int=number of features
    '''
    # calculate number of nodes for each layer
    keep = 0.6 # keep 60% number of nodes from previous layer
    num_nodes = [int(nodes_layer1*keep**i) for i in range(1,num_recurrent_layers)]
    num_nodes = [nodes_layer1] + num_nodes
    # calculate dropout ratio for each layer
    drop_outs = [round(0.2*0.9**i, 2) for i in range(1,num_recurrent_layers)]
    drop_outs = [0.2] + drop_outs
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(num_recurrent_layers):
        if i != num_recurrent_layers-1: # if NOT the last recurrent layer
            # return sequence
            model.add(LSTM(num_nodes[i], return_sequences=True, activation="relu"))
            model.add(Dropout(drop_outs[i]))
            model.add(BatchNormalization())
        else: # If this is the last recurrent layer
            model.add(LSTM(num_nodes[i], return_sequences=False, activation="relu"))
            model.add(Dropout(drop_outs[i]))
            model.add(BatchNormalization())
    # output
    model.add(Dense(num_output_nodes, activation="relu"))
    return model

def compile_and_fit(model, X_train, y_train, X_val, y_val, max_epochs=20, patience=2):
    '''
    Compile and fit tensor.keras models with early stoping condition.
    -----------------------------------------------
    return : history
    '''
    # use early stopping to halt training when desire condition is met.
    early_stopping = EarlyStopping(monitor="val_loss", # check validation loss
                                   mode="min", # keep minimum of validation loss
                                   patience=patience,
                                   verbose=1,
                                   min_delta=0.001, # minimum change to classify and improvement
                                   restore_best_weights=True)
    model.compile(loss="mse", optimizer="adam")
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=max_epochs, 
                        verbose=0,
                        callbacks=[early_stopping])
    return history

def neural_net_best(X_train, y_train, X_val, y_val, layer_type = "dense_straight", 
                    max_hidden_layers=4, nodes_per_hidden=500, output_nodes=1, 
                    flatten=False, input_3d=False):
    '''
    layer_type : str. "dense_straight", "dense_pyramid", "recurrent"
    '''
    # neural net build dictionary
    builts = {"dense_straight": build_dnn_straight,
              "dense_pyramid": build_dnn_pyramid,
              "recurrent_straight": build_rnn_straight,
              "recurrent_pyramid": build_rnn_pyramid}
    # appropriate input_shape for different model type
    input_shapes = {"dense_straight": X_train.shape[1:],
                  "dense_pyramid": X_train.shape[1:],
                  "recurrent_straight": [None, output_nodes],
                  "recurrent_pyramid": [None, output_nodes]}
    # record the iteration results
    log = {"hidden_layers": [],
           "Val_MSE": []}
    model_best = model_history = None
    min_loss = float("inf")
    # Tune number of hidden layer
    for i in range(2, max_hidden_layers+1):
        model = builts[layer_type](i, nodes_per_hidden, output_nodes,
                                   input_shapes[layer_type], flatten)
        # Compile and fit the model
        history = compile_and_fit(model=model,
                                  X_train=X_train,
                                  y_train=y_train,
                                  X_val=X_val,
                                  y_val=y_val,
                                  max_epochs=100,
                                  patience=10)
        # evaluate the model
        loss = model_evaluate(model=model,
                              fh=fh,
                              X_test=X_val,
                              y_test=y_val,
                              metric=mean_squared_error,
                              input_3d=input_3d)
        print(f"Hidden layers: {i}, Validation MSE: ", round(loss, 4))
        print("-"*50)
        # add info the log
        log["hidden_layers"].append(i)
        log["Val_MSE"].append(loss)
        
        if loss < min_loss:
            min_loss = loss
            #print(f"Hidden layers: {i}, Validation MSE: ", round(loss, 4), end="\r")
            model_best = model
            model_history = history
    return model_best, model_history, log


def model_evaluate(model, fh, X_test, y_test, metric, input_3d=False):
    '''
    Return evaluation metrics of a model 
    **(CAN NOT APPLY TO ARIMA MODEL)** using recursive forecast 
    of several forecast horizon in the test set. The function use sliding
    window technique (window width = fh) to make multiple loss evaluations 
    in the test set range. Then return the average of the losses at the end.
    ------------------------------------------------------------
    fh : int. Forecast horizon. Need to be EQUAL or LESS than the length of 
            the test set.
    X_train : pd.Series or DataFrame
    --------------------------------------------------------
    '''
    results = []
    for i in range((X_test.shape[0]-fh)+1): # minimum of 1 iteration
        true = y_test[i:(i+fh)] # setup true values
        # predict values
        pred = recursive_forecast(model=model, 
                                  X=X_test, 
                                  start=i, 
                                  fh=fh, 
                                  input_3d=input_3d)
        # score/loss of the model
        results.append(metric(true, pred))
    return np.mean(results)

# Loss metric
def sMAPE_tensor(actual, pred):
    '''
    Symmetric Mean Absolute Percentage Error.
    Range of return value (0 - inf).
    Note the same operations can be written using numpy. However, Tensor
    is being used here to make the function usable as a loss or metric function when
    training models.
    '''
    loss = tf.reduce_mean(tf.abs(actual - pred)/(tf.abs(actual) + tf.abs(pred)))*100
    return loss

def sMAPE(actual, pred):
    loss = np.mean(np.abs(actual - pred)/(np.abs(actual) + np.abs(pred)))*100
    return loss

def MAPE(actual, pred):
    '''
    Mean Absolute Percent Error.
    '''
    return np.mean(np.abs(actual - pred) / np.abs(actual)) * 100

##################################################
# predict method for tabular form
def recursive_forecast(model, X, start, fh, input_3d=False):
    '''
    model: a fitted model
    X: array
    start: index (NOT time stamp)
            Starting point of forecast. e.g. 0 = start at the beginning.
    fh: int
            foreast horizon
    input_3d : bool. Whether or not the model's predict() method require 3d input data.
    '''
    pred = None
    # point to start forecasting.
    # Also add one more dimension in the front, due to requirement of input
    # for model prediction.
    point = X[start][np.newaxis,...]
    # recursive forecasting
    for i in range(fh):
        temp = model.predict(point) 
        # note, some model predict return 1D array while some return 2D array.
        if i < 1: # first prediction point
            pred = temp
        else:
            pred = np.append(pred, temp, axis=0)
        # update point to test with new prediction
        if not input_3d: # if model uses 2-d input
            point = np.append(point[:,1:], temp.reshape(1,-1), axis=1)
        else: # if model requires 3-d input
            point = np.append(point[:,1:,:], temp.reshape(1,1,-1), axis=1)
    return pred

# Plot
def plot(data, labels):
    '''
    Make overlay plot of given data and labels.
    ----------------------------------------
    data: list of data sets to plot
    labels: list of labels
    '''
    plt.figure(figsize=(10,3))
    for i, dat in enumerate(data):
        plt.plot(dat, label=labels[i], marker="o")
    plt.legend()
    plt.show()