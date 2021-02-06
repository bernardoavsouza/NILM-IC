import pywt

'''
    This code is projected to plot the discrete wavelet decompositions

'''

class InputSignal:
    def __init__(self, original_signal, level=4, kind='db1' ):
        self._original_signal = original_signal
        self._level = 0
        self._fig = []
        self._kind = kind
        self._details = None
        self._approx = None
        
        self.level = level
        
        
    @property
    def fig(self):
        return self._fig
    @fig.setter
    def fig(self, value):
        self._fig = value

    @property
    def original_signal(self):
        return self._original_signal
    @original_signal.setter 
    def original_signal(self, value):
        self._original_signal = value
    
    @property
    def level(self):
        return self._level
    @level.setter 
    def level(self, value):
        # Calculating the max level of decomposition
        maxlevel = pywt.dwt_max_level(data_len=len(self.original_signal), filter_len=pywt.Wavelet(self.kind).dec_len)
        
        # Verifying if the level setted is lower than the maximum level
        if maxlevel < value:
            raise ValueError("High level of decomposition chosen. It should be lower than " + str(maxlevel + 1) + ".")
        
        # Assinging the chosen value
        self._level = value
 
    @property 
    def kind(self):
        return self._kind
    @kind.setter 
    def kind(self, value):
        self._kind = value
    
    @property
    def details(self):
        if self._details == None:
            # Applying the wavelet decomposition to get the datails. 
            # The first argument is the approximation coefficients.
            dec = pywt.wavedec(self.original_signal , self.kind, level=self.level)
            self._approx = dec[0]
            return dec[1:]
        else:
            return self._details
    
    @property
    def approx(self):
        try:
            if self._approx == None:
                # Applying the wavelet decomposition to get the approximation. 
                # The first argument is the approximation coefficients.
                dec = pywt.wavedec(self.original_signal , self.kind, level=self.level)
                self._details = dec[1:]
                return dec[0]
            else:
                return self._approx
        except ValueError:
            return self._approx
        
       
    def kind_show():
        # This function shows the kinds of discrete wavelets available
        
        kindlist = pywt.wavelist(kind='discrete')
      
        print(kindlist[0], end='')
        for i in range(1, len(kindlist)):
            print(", " + kindlist[i], end='')
        
    def plot(self):
        import matplotlib.pyplot as plt
        
        # Plotting original signal and the Wavelet approximation
        self.fig.append(plt.figure())
        plt.plot(self.original_signal)
        plt.title("Original Signal")
        self.fig.append(plt.figure())
        plt.plot(self.approx)
        plt.title("Approximation")
        
        # Plotting all of the decompositions
        for i in range(0, len(self.details)):
            self.fig.append(plt.figure())
            plt.plot(self.details[i])
            plt.title("Detail " + str(i + 1))
    
    def show(self):
        # This function opens the figures without process them again
        for i in range(len(self.fig)):
            self.fig[i].show()
            