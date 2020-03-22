import csv
import math
import random
import pandas as pd
import numpy as np
from scipy.stats import norm

#dataset[i] = [float(x) for x in dataset[i]]
data = pd.read_csv("drug.csv")
new_data = pd.read_csv("drug_novi.csv")

atributes_char = []
for col in data.select_dtypes(exclude = np.number).columns:
    atributes_char.append(col)
atributes_char.pop()

atributes_num=[]
for col in data.select_dtypes(include = np.number).columns:
    atributes_num.append(col)



alfa = 1

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
m = model(data,"Drug",atributes_char,atributes_num)




def predict(a,m,slucaj,atributes_char,atributes_num):
    pred={}
    p=1
    for klasaA in a:
        p=p*math.log(a.iloc[0][klasaA])
        for klasaS in slucaj:
            if klasaS in atributes_char:
                 p=p+math.log(m[klasaS].loc[slucaj.iloc[0][klasaS]].loc[klasaA])
            elif klasaS in atributes_num:
               x_i = slucaj.iloc[0][klasaS]
               x_mean = m[klasaS].loc["mean",klasaA]
               x_std = m[klasaS].loc["std",klasaA]
               f= norm.pdf(x_i, x_mean, x_std)
               p=p+math.log(f)
           
        pred[klasaA]=p
        p=1
    return pred
#                
prediction = {}
new = new_data.copy()
for i in range(len(new_data)):
    slucaj = new_data.loc[i].to_frame().T
    prediction = predict(a,m,slucaj,atributes_char,atributes_num)
    for x in prediction:
       new.loc[i,'klasa='+x] = prediction[x]
       new.loc[i,'prediction'] = max(prediction, key=lambda f: prediction[f])
       
print(new)
