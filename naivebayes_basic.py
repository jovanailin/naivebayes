import csv
import math
import random
import pandas as pd
import numpy as np

#dataset[i] = [float(x) for x in dataset[i]]
data = pd.read_csv("prehlada.csv")
new_data = pd.read_csv("prehlada_novi.csv")

atributes_char = []
for col in data.select_dtypes(exclude = np.number).columns:
    atributes_char.append(col)
atributes_char.pop()

atributes_num=[]
for col in data.select_dtypes(include = np.number).columns:
    atributes_num.append(col)



alfa = 0

def apriori(data):
    a = data.iloc[:,-1]
    a = a.value_counts()
    a = a / a.sum()
    a = a.to_frame()
    a = a.T
    return a

def model(data, klasa, atributes_char,atributes_num):
    m = {}
    for atribute in atributes_char:
        mat_kont = pd.crosstab(data[atribute],data[klasa])
        mat_kont = mat_kont + alfa
        mat_kont = mat_kont.div(mat_kont.sum()+len(atributes_char)*alfa)
        m[atribute]=mat_kont
    for atribute in atributes_num:
        mat_kont = pd.crosstab(data[atribute],data[klasa])
        columns = mat_kont.columns
        mat = pd.DataFrame(columns=columns, index=["mean","std"])
        for col in columns:
            x=data[[atribute,klasa]].loc[data[klasa]==col]
            
            x_mean = x[atribute].mean()
            x_std = x[atribute].std()
            mat.loc["mean",col]=x_mean
            mat.loc["std",col]=x_std
            
            m[atribute]=mat
            
    return m

a = apriori(data)
m = model(data,"Prehlada",atributes_char,atributes_num)


def predict(a,m,slucaj):
    pred={}
    p=1
    for klasaA in a:
#        p=p*a.iloc[0][klasaA]
        p=p*a.iloc[0][klasaA]
        for klasaS in slucaj:
#            p=p*m[klasaS].loc[slucaj.iloc[0][klasaS]].loc[klasaA]
            p=p*m[klasaS].loc[slucaj.iloc[0][klasaS]].loc[klasaA]
            
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
