from nilmtk import DataSet
import matplotlib.pyplot as plt
import pandas as pd

'''
    This class has been created to reunite the datasets and to get a specific 
    signal in a easy way. The path is needed to be setted according to the
    location of the .h5 files.

'''

class Load:
    def __init__(self, name, building, dataset):
        self._path = 0
        self._dataset = 0
        self._name = 0
        self._data = pd.DataFrame()
        self._building = building
        
        self.name = name
        self.path = r"C:\Users\Noudy\Desktop"
        self.dataset = dataset
        
    # Getters and setters    
    @property
    def path(self):
        return self._path
    @path.setter
    def path(self, value):
        self._path = value
        
    @property
    def dataset(self):
        return self._dataset     
    @dataset.setter
    def dataset(self, value):
        if value.lower() == "redd".lower():
            temp = DataSet(self.path + r'\redd.h5')
        elif value.lower() == "synd".lower():
            temp = DataSet(self.path + r'\SynD.h5')
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