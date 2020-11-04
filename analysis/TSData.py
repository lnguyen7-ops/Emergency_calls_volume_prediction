from sklearn.model_selection import train_test_split

# DATA object
class TSData:
    def __init__(self, data):
        self.data = data
        self.neighborhoods = list(self.data.columns)
        # list of the neighborhoods FIDs
        self.FIDs = [int(x.split(",")[0]) for x in self.neighborhoods]
        self.remain = None
        self.train = None
        self.validation = None
        self.test = None
        # tabular data format
        self.X = self.y = None #features and target
        self.X_train = self.X_remain = self.X_validation = None
        self.y_remain = self.y_train = self.y_validation = self.y_test = None
        
    # Train-validation-test split
    def ts_split(self, test_size, val_size=0.2, plot=True, train_show="all"):
        '''
        test_size: int or float
                Validation & test size.
        plot: bool
                True ==> show plot of train, validation, test.
        train_show: str or float
                Portion of train set to show, "all" or ratio.
        '''
        self.remain, self.test = train_test_split(self.data, test_size=test_size, shuffle=False)
        self.train, self.validation = train_test_split(self.remain, test_size=val_size, shuffle=False)
        print("Train shape:\t", self.train.shape)
        print("Validation shape:\t", self.validation.shape)
        print("Test shape:\t", self.test.shape)
        # plot out the data sets
        if train_show=="all":
            length = len(self.train)
        else:
            length = int(len(self.train)*train_show)
        if plot:
            plt.figure(figsize=(15,3))
            plt.plot(self.train[-length:], marker=".", label="Train")
            plt.plot(self.validation, marker=".", label="Validation")
            plt.plot(self.test, marker=".", label="Test")
            plt.legend()
            plt.show()
            
    # Data reduction into tabular format
    def TsToTabular(self, lag):
        # convert dataframe to numpy
        source = self.data.to_numpy()
        data = None
        # sliding window
        window = lag + 1 #lag+1 includes 1 target value
        for right in range(window, len(self.data)+1): 
            # select window and also convert to 3-D array
            temp = source[(right-window):right, :][np.newaxis,...]
            if type(data) != np.ndarray: # if 1st time series
                data = temp
            else:
                data = np.append(data, temp, axis=0)
        self.X = data[:,:lag,:]
        self.y = data[:,-1,:]
        
    # Train test split tabular format
    def tabular_split(self, test_size, val_size=0.2):
        '''
        test_size: int or float
                Validation & test size
        '''
        # shuffle set to False to NOT randomize the data picking.
        self.X_remain, self.X_test, self.y_remain, self.y_test = train_test_split(self.X, self.y,
                                                                  test_size=test_size,
                                                                  shuffle=False)
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_remain, self.y_remain,
                                                                               test_size=val_size,
                                                                               shuffle=False)
    
    def get_FID_from_index(self, index):
        '''
        Get a neighborhood FID based on it's index in the dataframe.
        -----------------------------------------------------------
        return : int
        '''
        return int(self.neighborhoods[index].split(",")[0])
    
    def get_name_from_index(self, index):
        '''
        Get a neighborhood name based on it's index in the dataframe.
        -----------------------------------------------------------
        return : str
        '''
        return self.neighborhoods[index].split(",")[1]
    
    def get_index_from_FID(self, fid):
        '''
        Get index of a neighborhood in the dataframe by the neighborhood's FID.
        '''
        return self.FIDs.index(fid)
