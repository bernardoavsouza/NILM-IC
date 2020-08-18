from nilmtk import DataSet
import matplotlib.pyplot as plt
import pandas as pd

'''
    This class has been created to reunite the datasets and to get a specific 
    signal in a easy way. The __path is needed to be setted according to the
    location of the .h5 files.

'''

class Load:
    __path = r"Database"
    def __init__(self, name, building, dataset):
        
        self._dataset = 0
        self._name = 0
        self._data = pd.DataFrame()
        self._building = building
        
        self.name = name
        
        self.dataset = dataset
        
    
    # Getters and setters      
    @property
    def dataset(self):
        return self._dataset     
    @dataset.setter
    def dataset(self, value):       
        if value.lower() == "redd".lower():
            temp = DataSet(Load.__path + r'\redd.h5')
        elif value.lower() == "synd".lower():
            temp = DataSet(Load.__path + r'\SynD.h5')
        else:
            raise Exception("Invalid dataset. Please insert a valid argument.")
        self.data = next(temp.buildings[self.building].elec[self.name].power_series())
    
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, value):
        self._data = value
    
    @property
    def building(self):
        return self._building
    @building.setter
    def building(self, value):
        self._building = value
    
    # Ploting the signal
    def plot(self):
        plt.plot(self.data)
    
    # Path setter
    @staticmethod
    def set_path(value):
        # It should be a String
        if type(value) is not str:                          
            raise ValueError("Please, insert the __path as a String.")
        else:
            import os.path
            # It should be a valid existing __path
            if os.path.isdir(value):
                Load.__path = value
            else:
                raise ValueError("Please, choose a valid __path.")
    
    # Path getter
    @staticmethod
    def show_path():
        return Load.__path