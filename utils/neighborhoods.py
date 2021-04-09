# Packages for generating polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import ast
import pandas as pd
import numpy as np

# Neighborhood class
class neighborhoods:
    def __init__(self, file_path, id_cols=('nhood_num', 'nhood_name')):
        '''
        file_path: str
                File path where neighborhoods csv file located.
        id_cols: tuple of form (x,y). Contain 2 column names to be used as neighborhood id.
            e.g. ('nhood_num', 'nhood_name') or ('neighborho', 'new_nhood')
            NOTE that if the csv contain rows with the same id_cols values, the later row
            information will be used. For example, if somehow 2 neighborhoods has the same
            nhood_num and nhood_name, information of the 2nd neighborhoods will be used.
        '''
        self.df = pd.read_csv(file_path)
        self.num_nhood = self.df.shape[0]
        # Neighborhood dictionary holds boundary polygon objects
        self.boundaries = {}
        for i in range(self.df.shape[0]):
            key = str(self.df.loc[i, id_cols[0]]) + "," + self.df.loc[i, id_cols[1]]
            self.boundaries[key] = Polygon(ast.literal_eval(self.df["geometry.coordinates"][i])[0])
        # list of neighborhoods (nhood_num, nhood_name)
        self.nhoods = list(self.boundaries.keys())
                    
    # Function that return neighborhood identification of a location
    def which_nhood(self, row, df=False, lng="lng", lat="lat", nhood="all"):
        '''
        Return a neighborhood idenfication (nhood_num and name) which a location (gps coordinate)
        belongs to.
        ***ATTENTION***
        gps point that is ON (instead of inside) a boundary is also counted as inside.
        ------------------------------------------
        row: tuple, list OR dataframe rows
                If df=True, row=dataframe row.
                Location gps in the form of longitude, latitude
        lng: str
                Column name of the longitude
        lat: str
                Column name of the latitude
        nhood: str or list of str. 
                "all" apply to all neighborhood names.
                Or for only selected neighborhood. Original neighborhood FID and names
                will be returned for non-selected neighborhoods.
        ----------------------------------------------
        return: string
                Neighborhood idenfication where location belong to.
                Return np.nan if location is outside of neighborhood dictionary.
        '''
        def inside_check():
            # loop through neighborhood boundaries
            for name, polygon in self.boundaries.items():
                # if point INSIDE or TOUCHES boundary
                if polygon.contains(point) or polygon.touches(point):
                    return name
            return np.nan # if no neighborhood contain this point.

        if df: # if operate on dataframe rows
            # create point
            point = Point(row[lng], row[lat])
        else: # if take in a single location
            point = Point(row[0], row[1])
        if nhood=="all": # apply to all neighrborhoods
            return inside_check()
        else: # apply to only selected neighborhood
            name = row["neighborhood"]
            if name in nhood:
                return inside_check()
            else: # Not in selected list, return FID and original name
                return f"{self.get_nhood_num(name)[0]},{name}"

    def get_adjacent(self, nhood):
        '''
        Return list of adjacent neighborhoods.
        ----------------------------------------------------------
        nhood: str (e.g. "19,Warrendale")
                Neighborhood to find adjacents of.
        ----------------------------------------------------------
        return: list
        '''
        return [name for name, polygon in self.boundaries.items() if self.boundaries[nhood].touches(polygon)]

    def get_nhood_num(self, nhood):
        '''
        Return nhood_num/nhood_nums(list) of a neighborhood (by name)
        '''
        nhood_nums = [x.split(",")[0] for x in self.nhoods if x.split(",")[1]==nhood]
        return nhood_nums