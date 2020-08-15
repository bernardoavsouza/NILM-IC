import pywt

'''
    This code is projected to plot the discrete wavelet decompositions

'''

class PlotWavelet:
    def __init__(self, signal, level=4, kind='db1' ):
        self._signal = signal
        self._level = 0
        self._fig = []
        self._kind = kind
        
        self.level = level
        
        
    @property
    def fig(self):
        return self._fig
    @fig.setter
    def fig(self, value):
        self._fig = value

    @property
    def signal(self):
        return self._signal
    @signal.setter 
    def signal(self, value):
        self._signal = value
    
    @property
    def level(self):
        return self._level
    @level.setter 
    def level(self, value):
        # Calculating the max level of decomposition
        maxlevel = pywt.dwt_max_level(data_len=len(self.signal), filter_len=pywt.Wavelet(self.kind).dec_len)
        
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
        calcdim()
        self._kind = value
       
    def kind_show(self):
        # This function shows the kinds of discrete wavelets available
        
        kindlist = pywt.wavelist(kind='discrete')
      
        print(kindlist[0], end='')
        for i in range(1, len(kindlist)):
            print(", " + kindlist[i], end='')
        
    def plot(self):
        import matplotlib.pyplot as plt
        
        # Applying the wavelet decomposition
        coeffs = pywt.wavedec(self.signal , self.kind, level=self.level)
        
        # Adding new plots to the figure list
        for i in range(0, len(coeffs) + 1):
            self.fig.append(plt.figure())
            if i == 0:
                plt.plot(self.signal)
                plt.title("Original Signal")
            else:
                plt.plot(coeffs[i-1])
                plt.title("Decomposition " + str(i))
    
    def show(self):
        # This function opens the figures without process them again
        for i in range(len(self.fig)):
            self.fig[i].show()

        

