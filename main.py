# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 23:37:50 2020

@author: Bernardo Vasconcellos
"""

from Load import Appliance as ap
from Waveletlib import InputSignal as wl
import numpy as np
from windowing import processing as win
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans as km
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix



                             # Setup
                             
database = 'redd'                   # Chosen dataset
house_num = 1                       # Chosen house
win_size = 60
win_step = 60
K = 1
level_list = [2, 4, 8]                # Decomposition level
wl_list = ['sym11', 'sym12', 'sym13', 'sym14', 'sym15',
           'sym16', 'sym17', 'sym18', 'sym19', 'sym20']


def window_define(x):    
    # Mode
    from scipy import stats
    return stats.mode(x)


appliances = {'fridge':ap('fridge', [1,2,3,5,6], database), 
              'oven':ap('electric oven', [1], database),
              'mw':ap('microwave', [1,2,3,5], database),
              "wd":ap('washer dryer', [1,2,3,4,5,6], database),
              'dw':ap('dish washer', [1,2,3,4,5,6], database),
              'light':ap('light', [1,2,3,4,5,6], database),
              'bathroom_gfi':ap('unknown', [1,3,4,5,6], database),
              'heater':ap('electric space heater', [1,5,6], database),
              'stove':ap('electric stove', [1,2,4,6], database),
              'disposal': ap('waste disposal unit', [2,3,5], database),
              'electronics': ap('CE appliance', [3,5,6], database),
              'furnace': ap('electric furnace', [3,4,5], database),
              'sa': ap('smoke alarm', [3,4], database),
              'air_cond': ap('air conditioner', [4,6], database),
              'subpanel': ap('subpanel', [5], database)}
cluster_num = {'fridge':3,'oven':2, 'mw':2, "wd":3, 'dw':2,
               'light':3,'bathroom_gfi':2, 'heater':3, 'stove':2,
               'disposal':2, 'electronics':2, 'furnace':2, 'sa':2,
               'air_cond':2, 'subpanel':2}
print("\rLoading data completed.")

# This part set the labels of the states
labels_dict = {}
counter = 0
for i in appliances.keys():
    print('\rSetting label of ' + i + '.')
    labels_dict[i] = []
    for j in range(len(appliances[i].data)):
        temp = km(n_clusters=cluster_num[i]).fit(appliances[i].data[j].values.reshape(-1,1))
        labels_dict[i].append(temp.predict(appliances[i].data[j].values.reshape(-1,1)) + counter)
    counter = counter + cluster_num[i]
print("\rAll labels have been setted.")

                            # Algorhythm

# Windowing the labels 
print('\rWindowing labels.')
labels_win_dict = {}
for i in appliances.keys():
    labels_win_dict[i] = []
    for j in range(len(appliances[i].data)):
        labels_win_dict[i].append(win(labels_dict[i][j], size=win_size, stepsize=win_step))

# Windowing the data
print("\rWindowing data.")
x_win_dict = {}
for i in appliances.keys():
    x_win_dict[i] = []
    for j in range(len(appliances[i].data)):
        x_win_dict[i].append(win(appliances[i].data[j], size=win_size, stepsize=win_step))

# Applying window_define to get the array of window's labels
labels_win = []
for i in appliances.keys():
    print('\rProcessing windows of ' + i + '.')
    for L in range(len(appliances[i].data)):
        for j in range(labels_win_dict[i][L].shape[1]):
            labels_win.append(int(window_define(labels_win_dict[i][L][j])[0]))
labels_win = np.array(labels_win)

print("\rWindowing completed.")


iteration = 1
for num in level_list:
    level = num
    for wave in wl_list:
        wl_kind = wave
        try:
            print("Processing . . . " + str(iteration) + r'/' + str(len(level_list)*len(wl_list)))
            
            # Applying Wavelet in the windows 
            wl_dict = {}
            temp = None
            temp2 = None
            for i in x_win_dict.keys():
                print('\rCalculating Wavelets in ' + i + '.')
                wl_dict[i] = []
                for L in range(len(x_win_dict[i])):
                    dt = pd.DataFrame()
                    for j in range(x_win_dict[i][L].shape[1]):
                        temp = wl(x_win_dict[i][L].iloc[:,j], level=level, kind=wl_kind)
                        temp2 = np.concatenate([k for k in temp.details])
                        temp = np.concatenate([temp2, temp.approx])
                        dt[j] = temp
                    wl_dict[i].append(dt)
                    
            # Turning all windows in one array
            print("\rPreparing to classify.")
            x_win = []
            for i in appliances.keys():
                for k in range(len(wl_dict[i])):    
                    for j in range(wl_dict[i][k].shape[1]):
                        x_win.append(list(wl_dict[i][k].iloc[:, 1]))
            x_win = np.array(x_win)
            
            # Splitting data
            x_train, x_test, y_train, y_test = train_test_split(x_win, labels_win, train_size=0.7)
            
            # Training
            print("\rTraining.")
            knn=KNeighborsClassifier(n_neighbors=K)    
            knn.fit(x_train, y_train)
            
            # Final Test
            print('\rTesting.')
            pred = knn.predict(x_test)
            result = accuracy_score(y_test,pred)
            conf = pd.DataFrame(confusion_matrix(y_test, pred))
            
            counter = 0
            temp = []
            for i in appliances.keys():
                counter = counter + cluster_num[i]
                for j in range(cluster_num[i]):
                    temp.append(i) 
            conf[str(counter)] = temp
            
            conf.to_csv(str(round(result*100,2)) + '%-' + wl_kind + '-'+ str(level) + 'level.csv')
            
        except ValueError as e:
            print(e)
        iteration +=1
            
print("\rDONE!")
  
#    -----------------------------------------------------------------------
    
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

entries = ['57.43%-rbio3.5-2level.csv', '59.43%-rbio3.1-2level.csv',
 '60.66%-sym2-2level.csv', '60.97%-rbio2.2-2level.csv',
 '62.25%-bior2.2-2level.csv', '62.38%-bior3.5-2level.csv',
 '62.9%-sym2-4level.csv', '63.03%-bior5.5-2level.csv',
 '63.6%-sym3-2level.csv', '63.79%-bior4.4-2level.csv',
 '63.97%-bior3.1-2level.csv', '64.16%-bior3.7-2level.csv',
 '64.34%-bior1.3-2level.csv', '64.67%-sym5-2level.csv',
 '64.86%-rbio1.5-2level.csv', '65.0%-rbio3.1-4level.csv',
 '65.05%-rbio3.7-2level.csv', '65.12%-rbio2.4-2level.csv',
 '65.16%-rbio1.3-2level.csv', '65.97%-bior3.1-4level.csv',
 '66.28%-rbio3.3-2level.csv', '66.29%-rbio2.6-2level.csv',
 '67.04%-bior3.3-2level.csv', '67.39%-bior2.6-2level.csv',
 '67.58%-sym8-2level.csv', '67.84%-sym7-2level.csv',
 '68.34%-rbio4.4-2level.csv', '69.39%-rbio5.5-2level.csv']

axis = np.array(['fridge', 'fridge', 'fridge', 'oven', 'oven', 'm. w.', 
                 'm. w.', 'w. d.','w. d.', 'w. d.', 'd. w.', 'd. w.', 
                 'light', 'light', 'light', 'gfi', 'gfi', 'heater',
                 'heater', 'heater', 'stove', 'stove', 'disposal',
                 'disposal', 'electronics', 'electronics', 'furnace',
                 'furnace', 'smoke a.', 'smoke a.', 'air_cond', 'air_cond',
                 'subpanel', 'subpanel'])
sns.set(font_scale=.6)
for i in entries:
    df = pd.read_csv(i)
    df = df.set_index(axis)
    df = df.iloc[:,1:-1]
    df.columns = axis
    df = df/df.max().max()
    hm = sns.heatmap(df,cmap="YlOrRd")
    fig = hm.get_figure() 
    fig.savefig(i[:-4] + '.png')
    plt.close()

#------------------------------------------------------------------------------

from Load import Appliance as ap
from Waveletlib import InputSignal as wl
import numpy as np
from windowing import processing as win
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans as km
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt

database = 'redd'                   # Chosen dataset
house_num = 1                       # Chosen house
win_size = 5
win_step = 1
epsilon = 20

signal = ap(1, 1, database).data

'''
# Windowing the data
signal_win = win(signal, size=win_size, stepsize=win_step)

# Standard Deviation
std = []
for i in range(signal_win.shape[1]):
    std.append(np.std(signal_win[i]))
'''

# Delta P
deltaP = [0]     
for k in range(1,len(signal)):
       deltaP.append(signal[k]-signal[k-1])


'''
# Defining epsilon
epsilon = {}
for i in appliances.keys():
    epsilon[i] = []
    for j in range(len(appliances[i].data)):
        epsilon[i].append(appliances[i].data[j].std())

# Getting events
time = []
transient = []
ss_time = []
for k in range(len(signal)-4):
    if std[k] >= epsilon:
        transient.append(std[k])
        time.append(k)
    elif abs(deltaP[k]) >= epsilon:
        transient.append(deltaP[k])
        time.append(k)
    else:
        ss_time.append(k)
'''

# Getting events
time = []
transient = []
ss_time = []
for k in range(len(deltaP)):
    if abs(deltaP[k]) >= epsilon:
        transient.append(deltaP[k+win_size])
        time.append(k+win_size)
    else:
        ss_time.append(k)
time = np.array(time)
transient = np.array(transient)
ss_time = np.array(ss_time)      

abs_transient = np.array([abs(i) for i in transient]).reshape(-1,1)


# Elbow method
n_clusters = 18


# Clustering
kmeans = km(n_clusters).fit(abs_transient)
prediction = kmeans.predict(np.array(abs(transient)).reshape(-1,1))
clusters_tags = np.unique(prediction)


# Functions
def grouping(transient, idx):
    ridx = []
    fidx = []
    
    temp = [idx[0]]
    for i in range(len(idx)):
        try:
            if transient[idx[i]]*transient[idx[i+1]] > 0:
                temp.append(idx[i+1])
            else:
                if transient[idx[i]] > 0:
                    ridx.append(temp)
                else:
                    fidx.append(temp)
                temp = [idx[i+1]]
        except IndexError:
            if transient[idx[i]] > 0:
                ridx.append(temp)
            else:
                fidx.append(temp)
                
    return ridx, fidx

def match(ridx, fidx):
    threshold = epsilon
    temp = []
    tempidx = []
    match_tuple = []
    if ridx[0][0] < fidx[0][0]:
        for i in range(len(ridx)):
            try:
                for j in ridx[i]:
                    for k in fidx[i]:
                        temp.append(abs(transient[j] + transient[k]))
                        tempidx.append((j,k))
                if min(temp) > threshold: break                        
                match_index = temp.index(min(temp))
                match_tuple.append(tempidx[match_index])
            except IndexError:
                pass
            temp = []   
            tempidx = []
    else:
        for i in range(len(ridx)):
            try:
                for j in ridx[i]:
                    for k in fidx[i+1]:
                        temp.append(abs(transient[j] + transient[k]))
                        tempidx.append((j,k))
                if min(temp) > threshold: break
                match_index = temp.index(min(temp))
                match_tuple.append(tempidx[match_index])
            except IndexError:
                pass
            temp = []   
            tempidx = []
    
    return match_tuple



cycles_signal = {}
for i in range(n_clusters):
    try:
        idx = np.where(prediction==i)[0]
        ridx, fidx = grouping(transient,idx)
        cycle = match(ridx, fidx)
        temp = np.zeros(len(signal))
        for t in cycle:
            temp[time[t[0]]:time[t[1]+1]] = kmeans.cluster_centers_[i]
        cycles_signal[i] = temp
        
        plt.figure()
        plt.scatter(time[idx],transient[idx],color='red')
        plt.plot(temp)
        
    except IndexError:
        pass



plt.figure()
for i in range(n_clusters):
    x = time[np.where(prediction==i)]
    y = transient[np.where(prediction==i)]
    if i <= 10:
        plt.scatter(x,y)
    else:
        plt.scatter(x,y, marker="x")

#plt.legend([i for i in range(0,18)])
    


distortions = []
K = range(1,50)
for k in K:
    kmeanModel = km(n_clusters=k)
    kmeanModel.fit(abs_transient)
    distortions.append(kmeanModel.inertia_)


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
















