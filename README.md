# RNN ve LSTM sinir ağları ile bölgesel dağıtılan enerji tahmini

* 2015-2020 yılları arasında Marmara bölgesindeki 4 ilde (Bursa-Balıkesir-Çanakkale-Yalova) dağıtılan toplam elektrik enerjisinin verisi (MWh) kullanılmıştır.
* Enerji tüketimi ile sıcaklık arasındaki ilişki incelendiğinde, birinin zirvelerini gördüğü yerlerde diğerinde çukurlaşmalar görülebilmektedir. Bu durum düşük sıcaklıkta ısıtıcı kullanımıyla oluşan enerji tüketiminin artmasından kaynaklanmaktadır. Yüksek sıcaklıklarda ise sıcaklık artışıyla birlikte sogutucu kullanımından kaynaklanan tüketim artışı görülmektedir.
## Modelde kullanılan değişkenler;
* ***sicaklik	Belirtilen tarih-saatin sıcaklık değeri***
* ***G1	Bir gün öncesi aynı saatin tüketim değeri***
* ***TG1	Bir gün öncesi aynı saatin sıcaklık değeri***
* ***G2	İki gün öncesi aynı saatin tüketim değeri***
* ***TG2	İki gün öncesi aynı saatin sıcaklık değeri***
* ***H1	Bir hafta öncesi aynı saatin tüketim değeri***
* ***TH1	Bir hafta öncesi aynı saatin sıcaklık değeri***
* ***Y1	Bir yıl öncesi aynı saatin tüketim değeri***
* ***TY1	Bir yıl öncesi aynı saatin sıcaklık değeri***
* ***dagitilan	Belirtilen tarih-saatin tüketim değeri***

