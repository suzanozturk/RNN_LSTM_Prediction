#!/usr/bin/env python
# coding: utf-8

# Tensorflow: Açık kaynak kodlu bir deep learning(derin öğrenme) kütüphanesidir. En büyük özelliği işlemleri birden fazla makineye dağıtabilmesi.
# Keras: Keras, Theano veya Tensorflow’u backend olarak kullanır.

# In[415]:


#1. Adım: Gerekli Kütüphanelerin Yüklenmesi
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.metrics import r2_score

from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential
from datetime import datetime, date
import datetime
import seaborn as sns

import datetime
import warnings
import lightgbm as lgb
import xgboost as xgb
import plotly.express as px
from typing import Optional, List, Dict
from fbprophet import Prophet
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[416]:


#2. Adım: Verinin yüklenmesi
fpath = r'D:\Users\200142\Desktop\TEZ\Data\dagitilan_data2.xlsx'
df = pd.read_excel(fpath)

df.shape
df.info()
df.head()
df.isnull().sum()

"""
Veri zaman serisi verisinden oluşmakta olup, saatlik tüketilen enerji değerlerini içermektedir. Görüldüğü gibi veri 'tarih', 'dagitilan', 'sicaklik' isimlerinde olan 3 kolondan ve 53.352 satırdan oluşmaktadır ve boş değer içermemektedir.
Kolonlardaki veri formatları datetime64, float64 ve int64'tür.
""" 

#3. Adım: Aykırı/Uç Değerleri Tespit Etme
df.describe()
sns.boxplot(x=df['dagitilan'])

"""
Veride negatif değerler bulunmaktadır. Ayrıca boxplot grafiği incelenirse; Q1, Q2 ve Q3 değerlerinin sırasıyla 1000-1500, 1500-2000 değerleri arasında hesaplandığı, 
ASD’nin 500'ün altında olduğu ÜSD’nin de 2000'in üzerinde olduğu ve ASD'nin altında değerlerin olduğu görülmektedir.
"""
Q1 = df.dagitilan.quantile(0.25)
Q2 = df.dagitilan.quantile(0.5)
Q3 = df.dagitilan.quantile(0.75)
Q4 = df.dagitilan.quantile(1)
IQR = Q3 - Q1

print("Q1-->", Q1)
print("Q3-->", Q3)
print("Q2-->", Q2)
print("Q4-->", Q4)
print("IQR-->", IQR)
print("Alt sınır: Q1 - 1.5 * IQR--->", Q1 - 1.5 * IQR)
print("Üst sınır: Q3 + 1.5 * IQR--->", Q3 + 1.5 * IQR)

df[df.dagitilan < Q1 - 1.5 * IQR].sort_values('dagitilan')
df.sort_values(['dagitilan'], ascending=True).head(20)

df.loc[(df['dagitilan'] < Q1 - 1.5 * IQR ), 'dagitilan'] = Q1 - 1.5 * IQR
df.sort_values(['dagitilan'], ascending=True).head()


# Index'i değiştirilir ve tarih index olarak atanır.

# In[417]:


df = pd.read_excel(fpath, index_col='datetime', parse_dates=['datetime'])
df.head()


# In[418]:


#boş değer kontrolü
df.isna().sum()


# In[420]:


df['dagitilan'].plot(figsize=(16,4),legend=True)

plt.title('Saatlik tüketilen enerji - Normalleştirme Öncesi')

plt.show()


# Dağıtılan enerji verisi 0-1 arasında olacak şekilde normalleştirilir. Bunun için sklearn MinMaxScaler kullanılmıştır.

# In[421]:


def normalize_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    df['dagitilan'] = scaler.fit_transform(df['dagitilan'].values.reshape(-1,1)) 
    return df

df_norm = normalize_data(df)
df_norm.shape


# In[422]:


df_norm['dagitilan'].plot(figsize=(16,4),legend=True)

plt.title('Saatlik tüketilen enerji - Normalleştirme Sonrası')

plt.show()


# input_shape= (nb_samples, timesteps, input_dim)

# In[292]:


def load_data(stock, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        X_train.append(stock.iloc[i-seq_len : i, 0])
        y_train.append(stock.iloc[i, 0])
        
    #Son 3352 adet veri test için kullanılacak.
    X_test = X_train[50000:]             
    y_test = y_train[50000:]
    
    #ilk 50000 adet veri egitimde kullanılacak.
    X_train = X_train[:50000]           
    y_train = y_train[:50000]
        
    #Numpy dizisine dönüştürülür.
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #RNN modeli için veri tekrar şekillendirilir.
    X_train = np.reshape(X_train, (50000, seq_len, 1))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]


# # Son 20 saatin değerlerini kullanarak tüketimin tahmin edilmesi
# 

# In[423]:


#Egitim ve test data boyutları
seq_len = 20 #kaç birim zamanlık veri verilecek burda onu belirtiyoruz.

X_train, y_train, X_test, y_test = load_data(df, seq_len)

print('X_train.shape = ',X_train.shape)
print('y_train.shape = ', y_train.shape)
print('X_test.shape = ', X_test.shape)
print('y_test.shape = ',y_test.shape)


# In[458]:


y_testdf = pd.DataFrame(y_test, columns = ['dagitilan'])


# # Basit RNN Modeli Kurulması
# 3 simpleRNN katmanı oluşturulmuştur. 
# ilk katman giriş katmanı 40 çıkışı bulunmaktadır. Kendinden sonraki katmana bağlandığından return_sequences = True değerindedir. 
# Giriş katmanı sonrasında  0.15 lik bir Dropout katmanı koyulmuştur. Dropout en basit ifade ile sistemin ezberlemesini önlemeye çalışan bir katmandır. 
# İkinci katman ve üçüncü katman da simpleRNN katmanıdır ve 40 çıkışı vardır. Onlardan sonra da yine Dropout eklenmiştir.
# Sonda ise çıkış katmanı olarak normal bir YSA katmanı Dense kullanılmıştır. Bir saatin tüketim tahminini vereceği için çıkışı 1'dir.
# 
# 
# input_shape= (nb_samples, timesteps, input_dim),
# imput_dim = tek bir sayısal değerin değişimi var ise  giriş veri boyutu 1 dir.
# 
# LSTM ler  genel anlamda sequence zaman sıralı şekilde gelen veriler üzerinden çalışırlar.
# keras.layers.recurrent.LSTM(output_dim,
#                                               batch_input_shape=(nb_samples, timesteps, input_dim),
#                                               return_sequences=False,....)
# (eğer LSTM ilk katmansa batch_input_shape  verilmesi gerekiyor.)
#   
# nb_samples : veri sayısıdır, None verilebilir. 
# timesteps :  LSTM e verilecek veriler zaman bazında değişen aynı türdeki verilerdir ve kaç birim zamanlık veri verilecek ise burada o belirtilmelidir.
# input_dim :  giriş verisinin boyutudur.
# return_sequences :  Kendinden sonraki katman Yine Bir LSTM olacaksa True  yazılmalıdır.
# output_dim :  Katman çıkış boyutu ve LSTM birim sayısıdır.  Ağın çıkışı , eğer kendinden sonraki katman Yine Bir LSTM olacaksa  (nb_samples, timesteps, output_dim), olmayacaksa  (nb_samples, , output_dim) olur.
# 
# parametre sayısı = (m * n + n^2 + n) 
# m = giriş parametre sayısı
# n = çıkış parametre sayısı
# 
# 1. katman : ( 1 * 40 + 40^2 + 40 = 1680 )
# 2. ve 3. katman : ( 40 * 40 + 40^2 + 40 = 3240 )
# Çıkış katmanı ( 40 ağırlık + 1 bias = 41 )
# 

# In[440]:


rnn_model = Sequential()

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
rnn_model.add(Dropout(0.15))

rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
rnn_model.add(Dropout(0.15))

rnn_model.add(Dense(1))  

rnn_model.summary()


# Birden fazla girdinin parçalar halinde işlenmesi “mini-batch” olarak adlandırılmaktadır. 
# Model tasarlanırken mini-batch parametresi olarak belirlenen değer; modelin aynı anda kaç veriyi işleyeceği anlamına gelmektedir.
# 
# Mutlak minimim değerinin bulunması için: Yapay sinir ağlarının optimizasyonu için en çok kullanılan yöntemlerden biri gradyan inişidir (Gradient descent).
# Gradyan inişi yöntemini esas alan çeşitli algoritmalar : Rmsprop, Adagrad, Adam, Nadam

# In[441]:


rnn_model.compile(optimizer="adam",loss="MSE")
rnn_model.fit(X_train, y_train, epochs = 10, batch_size = 1000)   #epoch = 10, bathc_size = 1000 


# In[442]:


rnn_predictions = rnn_model.predict(X_test)
print (X_test[:5,0].T)
print (rnn_predictions[:5,0])

rnn_score = r2_score(y_test,rnn_predictions)
print("R2 Score of RNN model = ",rnn_score)


# evaluate(), önceden eğitilmiş modeli değerlendirmek içindir. Model için kayıp değerini ve ölçüm değerlerini döndürür.

# In[443]:


trainScore = rnn_model.evaluate(X_train, y_train, batch_size = 1000, verbose = 0)
print('Train Score : ', trainScore)
testScore = rnn_model.evaluate(X_test[:252], y_test[:252], batch_size = 1000, verbose = 0)
print('Test Score : ', testScore)


# In[459]:


y_predictions_rnn = pd.DataFrame(rnn_predictions, columns = ['dagitilan'])


# In[445]:


def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16,4))
    plt.plot(test, color='blue',label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='orange',label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()
    
plot_predictions(y_test, rnn_predictions, "Predictions made by simple RNN model")


# # LSTM Katmanı kurulması
# Geleneksel RNN’lerde, tekrarlayan kısımda tek bir tanjant fonksiyonu (katmanı) bulunmaktadır.
# LSTM’lerde ise tek bir sinir katmanı yerine tekrarlayan zincir şeklinde dört farklı katman bulunmaktadır. 
# Bu nedenle;
# parametre sayısı = 4 * (m * n + n^2 + n) 
# m = giriş parametre sayısı 
# n = çıkış parametre sayısı
# 
# 1. katman : 4* ( 1 * 40 + 40^2 + 40 = 6720 )
# 2. ve 3. katman : 4 * ( 40 * 40 + 40^2 + 40 = 12960 ) 
# Çıkış katmanı ( 40 ağırlık + 1 bias = 41 )

# In[446]:


lstm_model = Sequential()

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
lstm_model.add(Dropout(0.2))

lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
lstm_model.add(Dropout(0.2))

lstm_model.add(Dense(1))

lstm_model.summary()


# In[447]:


lstm_model.compile(optimizer="adam",loss="MSE")
lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)


# In[448]:


lstm_predictions = lstm_model.predict(X_test)

lstm_score = r2_score(y_test, lstm_predictions)
print("R^2 Score of LSTM model = ",lstm_score)


# In[449]:


plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")


# In[450]:


plt.figure(figsize=(15,6))

plt.plot(y_test, c="orange", linewidth=3, label="Original values")
plt.plot(lstm_predictions, c="red", linewidth=3, label="LSTM predictions")
plt.plot(rnn_predictions, alpha=0.5, c="blue", linewidth=3, label="RNN predictions")
plt.legend()
plt.title("Predictions vs actual data", fontsize=20)
plt.show()


# In[451]:


y_predictions_lstm = pd.DataFrame(lstm_predictions, columns = ['dagitilan'])
y_predictions_lstm


# # Model değerlendirme
# Aşağıdaki metrikleri kullanarak zaman serisi tahmin modellerinin performansını değerlendirilir. 
# ortalama mutlak hata (MAE), ortalama karesel hata (RMSE), ortalama mutlak yüzde hatası (MAPE) ve sınır ortalama mutlak yüzde hatası (düşük gerçekliğe sahip kayıtları ortadan kaldıran CMAPE). 

# In[455]:


def evaluate_forecast_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mape_cutoff: int = 100,
) -> Dict:
 
    # Ortalama Mutlak Hata  (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Ortalama Kare Hata  (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Ortalama Mutlak Yüzde Hata  (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
    return {"MAE":round(mae, 3),
            "RMSE":round(rmse, 3),
            "MAPE":round(mape, 3)}
        


# In[456]:


rnn_metrics = evaluate_forecast_metrics(y_testdf['dagitilan'], y_predictions_rnn["dagitilan"])
print("RNN evaluate forecast metrics = ", rnn_metrics)
lstm_metrics = evaluate_forecast_metrics(y_testdf['dagitilan'], y_predictions_lstm["dagitilan"])
print("LSTM evaluate forecast metrics = ", lstm_metrics)


# In[457]:


rnn_predictions = rnn_model.predict(X_test)
rnn_score = r2_score(y_test,rnn_predictions)
print("R2 Score of RNN model = ",rnn_score)

lstm_predictions = lstm_model.predict(X_test)
lstm_score = r2_score(y_test, lstm_predictions)
print("R2 Score of LSTM model = ",lstm_score)

