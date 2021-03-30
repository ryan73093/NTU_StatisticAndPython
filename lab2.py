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
%matplotlib inline

def LCG(x,n):
    n_array=np.arange(1,n)
    listLCG=(1664525*(n_array+x-1+1013904223))%(2^32)
    df_LCG=pd.DataFrame({'DATA':listLCG})  
    MAX=df_LCG['DATA'].max()    
    df_LCG=(df_LCG['DATA']/MAX).round(5)
    

    return df_LCG

result=LCG(10,100000)
result=result.value_counts()
plt.plot(result)

scipy.stats.kstest(result,cdf='uniform')
