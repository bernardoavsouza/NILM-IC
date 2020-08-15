from sklearn.model_selection import train_test_split

'''
    This class split the signal in train and test.
    Insert as arguments the signal and the relative amount 
    of train OR test you want (from 0 up to 1).
    
'''

class Split:
    def __init__(self, data, train=None, test=None):
        self._data = None
        self._train = None
        self._test = None
        
        if train == None or test == None or train + test == 1:
            self.train = train
            self.test = test
            self.data = data
        else:
            print("Valores inválidos de train e test.")
        
    @property
    def train(self):
        return self._train
    @train.setter
    def train(self, valor):
        if self.test == None and valor != None:
            self._train = valor
            self._test = 1 - self.train
        
    @property
    def test(self):
        return self._test
    @test.setter
    def test(self, valor):
        if self.train == None and valor != None:
            self._test = valor
            self._train = 1- self.test 
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, valor):
        self._data = valor
        
    def divide(self):
        train, test = train_test_split(self.data, test_size=self.test, train_size=self.train)
        return train, test