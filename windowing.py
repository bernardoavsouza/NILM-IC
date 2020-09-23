# -*- coding: utf-8 -*-
"""
These functions are about windowning, making the use of a sliding window possible.

"""
# This function take the signal, apply the windowning and fit it in a dictionary
def windowing_dict(size=60, stepsize=60,**infos):
    keys = list(infos.keys())
    length = len(keys)    
    values = list(infos.values())
    
    output = {}
    for i in range(length):
        output[keys[i]] = processing(values[i], size=size, stepsize=stepsize)
    return output


def processing(data, size=60, stepsize=60, debug_mode=False):
    import pandas as pd
    dt = pd.DataFrame()
    
    # The last samples is thrown away (up to size number)
    win_num = int((len(data)-size)/stepsize) + 1
    if debug_mode == True:
        print('win_num: ', win_num)
        
    temp = []
    for i in [stepsize * k for k in range(win_num)]:
        temp.append(data[i:size+i])
    
    temp = [list(i) for i in zip(*temp)]
    dt = pd.DataFrame(temp)
    return dt


