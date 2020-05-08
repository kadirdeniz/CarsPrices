# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:40:59 2020

@author: deniz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
veriler=pd.read_csv('USA_cars_datasets.csv')
veriler.drop(columns=['Unnamed: 0','condition','country','vin','lot'],inplace=True)
print(veriler.groupby('state').price.mean())
print(veriler.info())
print(veriler.describe())
veriler.columns=(['Price','Brand','Model','Year','Title_status','Mileage','Color','State'])
veriler.rename(columns={"Brand":"brand","Price":"price","Model":"model","Year":"year","Title_status":"title_status","Millage":"milage","Color":"color","State":"state"},inplace=True)
print(pd.isna(veriler).sum())
newyork=veriler[veriler['state'].isin(["new york"])]
newyork.reset_index(inplace=True)
print(veriler.groupby('state').model.nunique())
print(veriler.groupby('brand').price.max())
print(veriler.state.nunique())

print(veriler.groupby('brand').model.nunique())
florida=veriler[veriler['state'].isin(['florida'])].reset_index()
print(florida.groupby('brand').price.max())
print(florida.groupby('brand').model.unique())
print(florida.groupby('brand').year.min())
florida_mercedes=florida[florida.brand=='mercedes-benz']
florida_chevrolet=florida[florida.brand=='chevrolet']
california=veriler[veriler.state.isin(['california'])]
california_chevrolet=california[california.brand=='chevrolet'].reset_index()

num_of_brand=veriler.groupby('brand').model.count().reset_index()
num_of_brand=num_of_brand.sort_values("model",ascending=False)
plt.bar(num_of_brand['brand'],num_of_brand['model'])
plt.title('Markalara Göre Araba Sayısı')
plt.xlabel('Araba Modelleri')
plt.ylabel('Araba Sayısı')
plt.show()

num_of_states=veriler.groupby('state').price.count().reset_index()
num_of_states=num_of_states.sort_values('price',ascending=False)
plt.bar(num_of_states['state'],num_of_states['price'])
plt.title('Eyalete Göre Araba Sayısı')
plt.xlabel('Eyaletler')
plt.ylabel('Araba Sayısı')
plt.show()

num_of_colors=veriler.groupby('color').price.count().reset_index()
num_of_colors=num_of_colors.sort_values('price',ascending=False)
plt.bar(num_of_colors['color'],num_of_colors['price'])
plt.title('Renge Göre Araba Sayısı')
plt.xlabel('Renkler')
plt.ylabel('Araba Sayısı')
plt.show()

num_of_years=veriler.groupby('year').price.count().reset_index()
num_of_years=num_of_years.sort_values('price',ascending=False)
plt.bar(num_of_years['year'],num_of_years['price'])
plt.title('Yıla Göre Araba Sayısı')
plt.xlabel('Renkler')
plt.ylabel('Araba Sayısı')
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
ss=StandardScaler()
lr=LinearRegression()
x = veriler['Mileage'].values.reshape(-1,1)
y = veriler['price'].values.reshape(-1,1)
x=ss.fit_transform(x)
y=ss.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr.fit(x_train,y_train)
prediction=lr.predict(x_test)

plt.scatter(x_train,y_train)
plt.plot(x_test,prediction,color='red')














