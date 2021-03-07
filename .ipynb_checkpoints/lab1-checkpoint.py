#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 20:34:21 2021

@author: pimi
"""
import pandas as pd
import ffn

import matplotlib.pyplot as plt


df = ffn.get('fb, aapl, amzn, nflx, goog', start = "2020-1-2")
df_rebase=df.copy()
def US1Dollar(newColumn,column):
    df[newColumn]=df[column]/df.iloc[0].loc[column]
    print(df.iloc[0].loc[column])

    return df

oriList=['fb','aapl','amzn','nflx','goog']
newList=['fb_new','aapl_new','amzn_new','nflx_new','goog_new']


plt.figure(figsize=(15,15))
for i ,j in zip(newList,oriList):
    df=US1Dollar(i, j)
    plt.plot(df.index,df[i])

plt.legend(oriList)

plt.grid(True)
plt.show()
plt.savefig("lab1_1.png")


df_rebase.rebase.plot()
