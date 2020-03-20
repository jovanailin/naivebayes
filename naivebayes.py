# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:27:36 2020

@author: Jovana
"""
import csv
import math
import random
import pandas as pd
import numpy as np

#dataset[i] = [float(x) for x in dataset[i]]
def loadcsv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    return dataset
data = loadcsv("drug.csv")
new_data = loadcsv("drug_novi.csv")

def table(dataset):
    column_names = np.array(dataset[0])
    dataset = np.delete(dataset, (0), axis=0)
    table = pd.DataFrame(dataset, columns=column_names, index = [int(i) for i in range(len(dataset))])
    return table, column_names

data, atributes = table(data)
new_data, new_atributes = table(new_data)
atributes = np.delete(atributes,-1)


def apriori(data):
    a = data.iloc[:,-1]
    a = a.value_counts()
    a = a / a.sum()
    a = a.to_frame()
    a = a.T
    return a

def model(data, klasa, atributes):
    m = {}
    for atribute in atributes:
        mat_kont = pd.crosstab(data[atribute],data[klasa])
        mat_kont = mat_kont.div(mat_kont.sum())
        m[atribute]=mat_kont
    return m

a = apriori(data)
m = model(data,"Drug",atributes)

def predict(a,m,slucaj):
    pred={}
    p=1
    for klasaA in a:
#        p=p*a.iloc[0][klasaA]
        p=p*math.log1p(a.iloc[0][klasaA])
        for klasaS in slucaj:
#            p=p*m[klasaS].loc[slucaj.iloc[0][klasaS]].loc[klasaA]
            p=p+math.log1p(m[klasaS].loc[slucaj.iloc[0][klasaS]].loc[klasaA])
            
        pred[klasaA]=p
        p=1
    return pred
#                
prediction = {}
new = new_data.copy()
for i in range(len(new_data)):
    slucaj = new_data.loc[i].to_frame().T
    prediction = predict(a,m,slucaj)
    for x in prediction:
       new.loc[i,'klasa='+x] = prediction[x]
       new.loc[i,'prediction'] = max(prediction, key=lambda f: prediction[f])
       
print(new)

output = new
output.to_csv(r'output.csv', index = False)





    

