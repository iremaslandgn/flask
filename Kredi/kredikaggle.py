# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:12:52 2019

@author: ayrem
"""

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
import pickle

df = pd.read_csv("krediVeriseti.csv",sep=";")
print(df.columns)
df = df.replace({"kiraci":0,"evsahibi":1,"verme":0,"krediver":1,"var":1,"yok":0})
df.head()

min_max_scaler = MinMaxScaler()

df = min_max_scaler.fit_transform(df)
df = pd.DataFrame(df)

egitimveri,validationveri = train_test_split(df,test_size=0.2,random_state=0)

egitimgirdi = egitimveri.drop(df.columns[5],axis=1)
egitimcikti = egitimveri[5]

valgirdi = validationveri.drop(df.columns[5],axis=1)
valcikti = validationveri[5]

a=SVC(C=100,kernel="rbf",gamma=0.1).fit(egitimgirdi,egitimcikti)
#Eğitim setiyle model kurulması
a = a.fit(egitimgirdi,egitimcikti)
#Kurulan model'in test edilmesi
y_pred = a.predict(valgirdi)
#Çıkan doğruluk skoru ve Hata Matrisi

models = []
models.append(("SVC",SVC()))

cm = confusion_matrix(valcikti, y_pred) 
print("SVC confusion_matrix:\n", cm)
print("SVC accuracy_score: ", accuracy_score(valcikti, y_pred)),
print("\nSVC f1_score:",f1_score(valcikti, y_pred)),
filename = 'traditionalml.sav'
pickle.dump(a, open("ram_model.pkl", 'wb'))
a=pickle.load(open("ram_model.pkl","rb"))
print("\n")