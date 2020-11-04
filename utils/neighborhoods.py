# Packages for generating polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import ast
import pandas as pd
import numpy as np

# Neighborhood class
class neighborhoods:
    def __init__(self, file_path):
        '''
        file_path: str
                File path where neighborhoods csv file located.
        '''
        self.df = pd.read_csv(file_path)
        self.num_nhood = self.df.shape[0]
        # Neighborhood dictionary holds boundary polygon objects
        self.boundaries = {}
        for i in range(self.df["geometry"].shape[0]):
            key = str(self.df.loc[i, "FID"]) + "," + self.df.loc[i, "new_nhood"]
            self.boundaries[key] = Polygon(ast.literal_eval(self.df["geometry"][i])["coordinates"][0])
        # list of neighborhoods
        self.nhoods = list(self.boundaries.keys())
                    
    # Function that return neighborhood identification of a location
    def which_nhood(self, row, df=False, lng="lng", lat="lat"):
        '''
        Return a neighborhood idenfication which a location (gps coordinate)
        belongs to.
        ------------------------------------------
        row: tuple, list OR dataframe rows
                If df=True, row=dataframe row.
                Location gps in the form of longitude, latitude
        lng: str
                Column name of the longitude
        lat: str
                Column name of the latitude
        ----------------------------------------------
        return: string
                Neighborhood idenfication where location belong to.
                Return np.nan if location is outside of neighborhood dictionary.
        '''
        if df: # if operate on dataframe rows
            # create point
            point = Point(row[lng], row[lat])
        else: # if take in a single location
            point = Point(row[0], row[1])
        # loop through neighborhood boundaries
        for name, polygon in self.boundaries.items():
            # if point INSIDE or TOUCHES boundary
            if polygon.contains(point) or polygon.touches(point):
                return name
        return np.nan
    
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
            