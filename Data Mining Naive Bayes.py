import pandas as pd
import numpy as np
from sklearn import preprocessing

# input
df = pd.read_excel('data\data_naive_bayes.xlsx','Sheet1')

# outpud data
print(df)

# konversi ke kode (numerik)
le = preprocessing.LabelEncoder()
df.Umur = le.fit_transform(df.Umur)
df.Pendapatan = le.fit_transform(df.Pendapatan)
df.Mhs = le.fit_transform(df.Mhs)
df.Rating_Kredit = le.fit_transform(df.Rating_Kredit)
df.Beli_Komputer = le.fit_transform(df.Beli_Komputer)

# menentukan fitur
x=df[["Umur","Pendapatan","Mhs","Rating_Kredit"]]
# menentukqn label
y=df["Beli_Komputer"]

#Membuat Model
from sklearn.naive_bayes import GaussianNB

#buat model klasifikasi NBGaussian
model = GaussianNB()

# latih model menggunakan dataset training
model.fit(x,y)

# data numerik
print(df)

# prediksikan class dari data baru apakah Yes atau No
prediksi= model.predict([[1, 0, 0, 0]]) # 1:<=30, 0:rendah, 0:bukan, 0:Exccelent
print ("prediksi untuk umur<=30, pendapatan=rendah, mhs=tidak,rating kredit=Excellent  : ", prediksi)
if(prediksi[0]==0):
 print("Hasil Prediksi : tdk")
else:
 print("Hasil Prediksi : ya")