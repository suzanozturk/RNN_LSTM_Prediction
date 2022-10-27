#!/usr/bin/env python
# coding: utf-8

# # 2015-2020 Yılları Arasında Marmara Bölgesindeki 4 İlde(Bursa-Balıkesir-Çanakkale-Yalova) Dağıtılan Enerji

# In[72]:


#2015-2020 yılları arasında Marmara bölgesindeki 4 ilde dağıtılan toplam enerji verisi(MWh) kullanılmıştır.

#1. Adım: Gerekli Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import datetime
import seaborn as sns


# In[73]:


#2. Adım: Verinin yüklenmesi
data = pd.read_excel(r'D:\Users\200142\Desktop\TEZ\Data\dagitilan_data2.xlsx')

data.info()

data['datetime'] = pd.to_datetime(data['datetime'], format= '%Y-%m-%d %H:%M:%S')
data


# **Veri zaman serisi verisinden oluşmakta olup, saatlik tüketilen enerji değerlerini içermektedir. Görüldüğü gibi 'tarih', 'dagitilan', 'sicaklik' isimlerinde olan 3 kolondan ve 53.352 satırdan oluşmaktadır ve boş değer içermemektedir.
# Kolonlardaki veri formatları datetime64, float64 ve int64'tür.

# In[74]:


#3. Adım: Aykırı/Uç Değerleri Tespit Etme
data.describe().T


# In[75]:


_= sns.boxplot(x=data['dagitilan'])
data.sort_values(['dagitilan'], ascending=True).head(10)


# **Veride negatif değerler bulunmaktadır. Ayrıca boxplot grafiği incelenirse; Q1, Q2 ve Q3 değerlerinin sırasıyla 1000-1500,  1500-2000 değerleri arasında hesaplandığı, alt sınır değerinin altında değerlerin olduğu görülmektedir.
# 

# In[78]:


#**Quartile (Kartiller) ve IQR ile Aykırı Değer Tespiti

Q1 = data.dagitilan.quantile(0.25)
Q2 = data.dagitilan.quantile(0.5)
Q3 = data.dagitilan.quantile(0.75)
Q4 = data.dagitilan.quantile(1)
IQR = Q3 - Q1

print("Q1-->", Q1)
print("Q2-->", Q2)
print("Q3-->", Q3)
print("Q4-->", Q4)
print("IQR-->", IQR)
print("Alt sınır: Q1 - 1.5 * IQR--->", Q1 - 1.5 * IQR)
print("Üst sınır: Q3 + 1.5 * IQR--->", Q3 + 1.5 * IQR)

data[data.dagitilan < Q1 - 1.5 * IQR].sort_values('dagitilan')
data.sort_values(['dagitilan'], ascending=True).head(20)

data.loc[(data['dagitilan'] < Q1 - 1.5 * IQR ), 'dagitilan'] = Q1 - 1.5 * IQR
data.sort_values(['dagitilan'], ascending=True).head(10)


# **Verinin Q1, Q2, Q3 değerleri bulunduktan sonra Interquartile range (IQR — Çeyrekler açıklığı) hesaplanır. 
# IQR, birinci çeyrek Q1 ve üçüncü çeyrek Q3 arasındaki uzaklıktır. IQR kullanılarak alt ve üst sınır değerleri aşağıdaki gibi hesaplanır.                
# 
# IQR = (Q3 – Q1)
# 
# Alt Sınır Değer (ASD)= Q1–1.5(IQR)
# 
# Üst Sınır Değer (ÜSD)= Q3 + 1.5(IQR)
# 
# ASD ve ÜSD dışında kalan tüm değerler outlier (aykırı değer) olarak tespit edilir.
# 
# ASD = 487.30 ÜSD = 2275.23
# 
# **Grafikte de görüleceği üzere alt sınır değerinin altında olan değerler mevcuttur.
# Bu değerlere ait günler incelendiğinde 31 Mart 2015 tarihinde Ülke genelinde kesinti olduğu bilgisine ulaşılmıştır.
# ASD altındaki tüm değerler ASD'ne eşitlenmiştir.

# In[79]:


#Son 6 senenin enerji tüketimi
plt.figure(figsize=(15,6))
plt.plot(data.datetime,data.dagitilan, 'b--') 
plt.xlabel('tarih') 
plt.ylabel('dagitilan enerji (MWh)')
plt.show()

#Son 6 senenin hava sıcaklığı
plt.figure(figsize=(15,6))
plt.plot(data.datetime,data.sicaklik, 'r--') 
plt.xlabel('tarih') 
plt.ylabel('sıcaklık')
plt.show()


# In[80]:


#Adım 4: Yeni Öznitelikler Oluşturma
#Yükün farklı saatlerde, günlerde, haftalarda ve sıcaklıklardaki değişiminin incelenebilmesi için eldeki verilerden yeni özellikler çıkarılır. 


def create_features(df, label = None, label2= None):
    
    df.index = df.datetime  
    df = df.copy()
    df['date'] = df.index
    df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d', errors= 'ignore')
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['day_name']=df['date'].dt.day_name()
    df['sicaklik'] = df['sicaklik']
    df['is_weekend'] = np.where(df['day_name'].isin(['Sunday', 'Saturday']), 1,0)
    df["ix"] = range(0,len(df)) 
    df["movave_30"] = df.dagitilan.rolling(30).agg([np.mean])
    df["movave_90"] = df.dagitilan.rolling(90).agg([np.mean])
    df["movave_365"] = df.dagitilan.rolling(365).agg([np.mean])
    
    X = df[['hour','date', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 
            'weekofyear','day_name', 'is_weekend','ix', 'sicaklik', 
            'movave_30','movave_90','movave_365']]
    
    if label:
        y = df[label]
        return X, y
    return X
        
        
X, y = create_features(data, label='dagitilan', label2 = 'sicaklik')


new_featuresdf = pd.concat([X, y], axis=1)        
new_featuresdf.info()


# In[81]:


new_featuresdf.tail(3)


# In[82]:


sns.pairplot(new_featuresdf.dropna(), hue='hour',
             x_vars = ['hour', 'dayofweek','year', 'weekofyear', 'sicaklik'],
             y_vars = ['dagitilan'],
             height = 4,
             plot_kws = {'alpha' : 0.15, 'linewidth' : 0})

plt.suptitle('24 Saatlik, Haftanın Farklı Günleri, Yıllık, Tüm Haftalar, Sıcaklık')
plt.show()


# **Grafikte elektrik tüketiminin 24 saatlik, 1 haftalık ve yıl toplamlarının profili ve tüketim-sıcaklık ilişkisi verilmiştir. 
# 
# Grafikler incelendiğinde gün içerisinde saat 10:00-15:00 arasında yükün maksimuma ulaştığı, 
# 
# 1 haftalık profilden hafta ortalarında yükün arttığı,
# 
# 2020'ye kadar tüketimin hep arttığı, 2020 yılında düşüş olduğu,
# 
# Yıllık profilden ise mevsimselliğin etkisi görülebilmektedir.
# 
# Ayrıca sıcaklık değişimine bağlı olarak, çok soğuk/sıcak olması halinde tüketimin artışı görülmektedir.

# In[83]:


_ = new_featuresdf['dagitilan'].plot.hist(figsize=(15, 5), bins=200, title='Dagıtılan Yükün Histogramı')


# **Dağıtılan Enerji Histogramı 2 tepeli çıkmıştır. Yük daha çok 1000-1250 ve 1375-1625 aralığında yığılmıştır. Dağılım sağa çarpıktır.

# In[84]:


#Adım 4. Mevsimsellik ve Trend Tespiti

new_featuresdf.groupby(new_featuresdf.index.year).mean().T #yıllık ortalama
new_featuresdf.resample(rule="MS").mean().T  #aylık ortalama

data_rolling = new_featuresdf.dagitilan.rolling(window=720) #aylık toplam saat sayısı
new_featuresdf['q10'] = data_rolling.quantile(0.1).to_frame("q10")
new_featuresdf['q50'] = data_rolling.quantile(0.5).to_frame("q50")
new_featuresdf['q90'] = data_rolling.quantile(0.9).to_frame("q90")

new_featuresdf.q10.head(500)

new_featuresdf[["q10", "q50", "q90"]].plot(title="Volatility Analysis: 720-rolling percentiles")
plt.ylabel("(MWh)")
plt.show()


plt.figure(figsize=(15,6))
new_featuresdf[[ "movave_30", "movave_90", "movave_365"]].plot(title="Enerji Tüketimi (MWh)")
plt.ylabel("(MWh)")
plt.show()


# **Trend: zaman serilerinde uzun vadede gösterilen kararlı durumdur. Veri incelendiğinde yıl içerisinde sezonlara göre dalgalanmaların mevcut olduğunu görülür.
# 
# Ayrıca diğer yıllardan farklı olarak 2020 yılında pandemi etkisinden kaynaklanan tüketim düşüşü göze çarpmaktadır.

# In[85]:


sns.boxplot(data=new_featuresdf, x='quarter', y="dagitilan")
plt.title("Çeyreklere Göre Dağılım")
plt.ylabel("(MWh)")
plt.show()

sns.boxplot(data=new_featuresdf, x="dayofweek", y="dagitilan")
plt.title("Hafta içindeki Dağılım")
plt.ylabel("(MWh)")
plt.show()


# **Zaman serisinin belirli bir davranışı belirli periyotlarla tekrar etmesi durumuna mevsimsellik denir. İncelenen grafiklerden veride mevsimsellik bulunduğu açıkça görülmektedir.
# Zaman serilerinde durağanlık zaman içerisinde varyansın ve ortalamanın sabit olmasıdır. 
# Verinin 0.1-0.5-0.9'luk çeyreklikleri için 720 saatlik durağanlık analizi yapıldığında volatilitesinin kısa vadede (çeyrek ve ay) değiştiği, uzun vadede daha kararlı yapıda olduğu yorumu yapılabilir.
# 
# Elektrik tüketiminin 30-90-365 günlük hareketli ortalamaları, çeyreklerdeki ve hafta içindeki tüketimler incelendiğinde ise;
# 
# Yılın 3. çeyreğinde tüketimin maksimuma ulaştığı ve en düşük tüketimlerin 2. çeyrekte olduğu, 
# 
# 2020 yılına kadar yıl ortalama tüketiminin sürekli arttığı,
# 
# Yıllık ortalama tüketimin en çok 2019 yılında olduğu,
# 
# 2020 yılında pandemi etkisiyle birlikte ortalama tüketimin düşüşü,
# 
# Hafta içi çarşamba gününün diğer günlere göre ortalama tüketimin daha yüksek olduğu,
# 
# Değişkenliğin en az pazar gününde yaşandığı görülebilmektedir.

# In[86]:


sns.boxplot(data=new_featuresdf["2015":"2020"], x="year", y="dagitilan")
plt.title("Trend Analizi: Yıllık Box-Plot Dağılımı")
plt.ylabel("(MWh)")
plt.show()


# Yıllık Box-Plot grafiğine bakıldığında; 
# Tüketimin her yıl artan bir trende sahip olduğu,
# Ortalama tüketimin en fazla 2016-2017 yıl geçişinde arttığı,
# 2020 yılında değişkenliğin en fazla olduğu görülmektedir.

# In[87]:


#Çeyreklerdeki Tüketim Profili
fig, ax = plt.subplots(figsize=(15,5))                                    
sns.boxplot(new_featuresdf.loc[new_featuresdf['quarter']==1].hour, new_featuresdf.loc[new_featuresdf['quarter']==1].dagitilan)
ax.set_title('Saatlik Box-Plot-Q1')
ax.set_ylim(0,3000)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(new_featuresdf.loc[new_featuresdf['quarter']==2].hour, new_featuresdf.loc[new_featuresdf['quarter']==2].dagitilan)
ax.set_title('Saatlik Box-Plot-Q2')
ax.set_ylim(0,3000)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(new_featuresdf.loc[new_featuresdf['quarter']==3].hour, new_featuresdf.loc[new_featuresdf['quarter']==3].dagitilan)
ax.set_title('Saatlik Box-Plot-Q3')
ax.set_ylim(0,3000)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(new_featuresdf.loc[new_featuresdf['quarter']==4].hour, new_featuresdf.loc[new_featuresdf['quarter']==4].dagitilan)
ax.set_title('Saatlik Box-Plot-Q4')
_ = ax.set_ylim(0,3000)                                      


Q1 = new_featuresdf[new_featuresdf["quarter"]==1]
Q2 = new_featuresdf[new_featuresdf["quarter"]==2]
Q3 = new_featuresdf[new_featuresdf["quarter"]==3]
Q4 = new_featuresdf[new_featuresdf["quarter"]==4]

fig,axes = plt.subplots(2,2,figsize=(17,7),sharex=True,sharey=True)

sns.distplot(Q1["dagitilan"],color="skyblue", ax=axes[0,0]).set_title("Q1 - Tüketim")
sns.distplot(Q2["dagitilan"],color="red", ax=axes[0,1]).set_title("Q2 - Tüketim")
sns.distplot(Q3["dagitilan"],color="green", ax=axes[1,0]).set_title("Q3 - Tüketim")
sns.distplot(Q4["dagitilan"],color="gray", ax=axes[1,1]).set_title("Q4 - Tüketim")
del Q1, Q2, Q3, Q4


# **Yılın çeyreklerindeki(Q1-Q2-Q3-Q4) profil,trend ve dağılım birbirinden farklıdır. Çeyreklerden Q3 diğer üçünden oldukça farklıdır. 
# Bunun en çok bilinen sebebi mevsimsellik olmakla birlikte başka nedenler de kaynaklanıyor olabilir.
# 
# 

# In[88]:


new_featuresdf.groupby("month")["dagitilan"].std().divide(new_featuresdf.groupby("month")["dagitilan"].mean()).plot(kind="bar")
plt.title("Ayların Varyasyon Katsayısı")
plt.show()


# Varyasyon katsayısı =  std sapma / ortama 
# 
# VK Ana kütlelerin ortalamaları büyüklük olarak birbirinden çok farklı ise karşılaştırma için kullanılmaktadır.
# VK değeri küçük olan serilerin diğerlerine göre daha az değişken olduğu söylenir ve sırasıyla Ekim-Nisan-Mayıs aylarında değişkenliğin en az olduğu grafikten görülmektedir.
# 

# In[90]:


_ = new_featuresdf[['dagitilan', 'hour']].plot(x='hour',
                                 y='dagitilan',
                                 kind='scatter',
                                 figsize=(14,4),
                                 title=('Saatlik Tüketim Profili'))

daily_trends = new_featuresdf.pivot_table(index=['hour'],
                               columns=['day_name'], 
                               values=['dagitilan'],
                               aggfunc = 'mean' ).plot(figsize=(15,4), title='Günlük Trend')


# Günlük trende bakıldığında;
# 
# Salı-Çarşamba-Perşembe günlerinin profillerinin çok benzer olduğu, 
# 
# Pazartesi gününün ilk saatlerinde tüketimin bu 3 günden daha düşük başladığı fakat saat 10'a doğru onları yakaladığı,
# 
# Cuma günlerinde hafta içi diğer günlere göre 11-14 aralığında düşüşün yaşandığı,
# 
# Cumartesi ve pazar günlerinin hafta içinden daha düşük tüketimde ve birbirlerinden farklı profillerde olduğu sonucuna varılmaktadır.

# In[91]:



#Tüketim-Sıcaklık İlişkisi
fig, ax1 = plt.subplots(figsize = (20,5))
ax1.plot(new_featuresdf.date, new_featuresdf.sicaklik, color = 'tab:orange')
ax1.set_ylabel('Sıcaklık')
ax2 = ax1.twinx()
ax2.plot(new_featuresdf.date,new_featuresdf.dagitilan,color = 'tab:blue')
ax2.set_ylabel('Enerji',color = 'tab:blue')
plt.title('Enerji Tüketimi ve Sıcaklık İlişkisi')
fig.tight_layout()
plt.show()


# Enerji tüketimi ile sıcaklık arasındaki ilişki incelendiğinde, birinin zirvelerini gördüğü yerlerde diğerinde çukurlaşmalar görülebilmektedir. 
# Bu durum düşük sıcaklıkta ısıtıcı kullanımıyla oluşan enerji tüketiminin artmasından kaynaklanmaktadır.
# Yüksek sıcaklıklarda ise sıcaklık artışıyla birlikte sogutucu kullanımından kaynaklanan tüketim artışı görülmektedir.
