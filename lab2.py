#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:56:03 2021

@author: pimi
"""
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np

def LCG(x,n):
    # listLCG=[]
    n_array=np.arange(1,n)
    listLCG=(1664525*(n_array+x-1+1013904223))%(2^32)
    # for i in range(n):
        # result=(1664525*(i+x-1+1013904223))%(2^32)
        # listLCG.append(result)
    df_LCG=pd.DataFrame({'DATA':listLCG})  
    MAX=df_LCG['DATA'].max()    
    df_LCG=list(df_LCG['DATA']/MAX)

    return df_LCG

result=LCG(2,1000000)
plt.plot(result)

scipy.stats.kstest(result,cdf='uniform')
