# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:34:58 2018

@author: Puneet Garg
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset2=pd.read_excel('rough data.xlsx')

x1=dataset2.iloc[0:637,10].values
y1=dataset2.iloc[0:637,6].values
y2=dataset2.iloc[0:637,7].values
y3=dataset2.iloc[0:637,8].values

plt.scatter(y1,x1,color='red')
plt.title('Fall of wicket between 11-15 overs vs avg run-rate')
plt.xlabel('Fall of wicket between 11-15 overs')
plt.ylabel('avg run-rate')
plt.show()


plt.scatter(y2,x1,color='red')
plt.title('run by wicket between 11-15 overs vs avg run-rate')
plt.xlabel('run by wicket between 11-15 overs')
plt.ylabel('avg run-rate')
plt.show()


import pandas as pd
import statsmodels.formula.api as sm
df = pd.DataFrame({"A": x1, "B": y1, "C": y2,"D": y3})
result = sm.ols(formula="A ~ B + C + D", data=df).fit()
print(result.params)

print (result.summary())