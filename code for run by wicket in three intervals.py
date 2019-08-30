
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_excel('r and w 1-6,7-10,11-15.xlsx')
dataset=dataset.fillna(0)

for i in range(len(dataset)):
    if(dataset["w1-6"][i]!=0):
        dataset["r/w1-6*"][i]=dataset["r1-6"][i]/(dataset["w1-6"][i])
    else:
        dataset["r/w1-6*"][i]=dataset["r1-6"][i]
    if(dataset["w7-10"][i]!=0):
        dataset["r/w7-10*"][i]=dataset["r7-10"][i]/(dataset["w7-10"][i])
    else:
        dataset["r/w7-10*"][i]=dataset["r7-10"][i]
    if(dataset["w11-15"][i]!=0):
        dataset["r/w11-15*"][i]=dataset["r11-15"][i]/(dataset["w11-15"][i])
    else:
        dataset["r/w11-15*"][i]=dataset["r11-15"][i]
        
    
        