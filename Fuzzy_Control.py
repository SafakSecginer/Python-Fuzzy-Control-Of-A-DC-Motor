import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.membership as mf


x = np.arange(-4, 7, 1)

#ÜYELİK FONKSİYONLARININ TANIMLARI
hata_negatif = mf.trimf(x, [-4, -2, 0])
hata_sifir = mf.trimf(x, [-2, 0, 2])

d_hata_negatif = mf.trimf(x, [-4, -2, 0])
d_hata_sifir = mf.trimf(x, [-2, 0, 2])
d_hata_pozitif = mf.trimf(x, [0, 2, 4])

c_sinyal_negatif = mf.trimf(x, [-4, -2, 0])
c_sinyal_sifir = mf.trimf(x, [-2, 0, 2])
c_sinyal_pozitif = mf.trimf(x, [0, 2, 4])
c_sinyal_cokpozitif = mf.trimf(x, [2, 4, 6])

#ÜYELİK FONKSİYONLARININ GRAFİKLERİNİN ÇİZDİRİLMESİ
fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (8, 10))

ax0.plot(x, hata_negatif, 'r', linewidth = 2, label = 'Neg')
ax0.plot(x, hata_sifir, 'b', linewidth = 2, label = 'Zer')
ax0.set_title("HATA")
ax0.legend()

ax1.plot(x, d_hata_negatif, 'r', linewidth = 2, label = 'Neg')
ax1.plot(x, d_hata_sifir, 'b', linewidth = 2, label = 'Zer')
ax1.plot(x, d_hata_pozitif, 'y', linewidth = 2, label = 'Poz')
ax1.set_title("HATA DEĞİŞİMİ")
ax1.legend()

ax2.plot(x, c_sinyal_negatif, 'r', linewidth = 2, label = 'Neg')
ax2.plot(x, c_sinyal_sifir, 'b', linewidth = 2, label = 'Zer')
ax2.plot(x, c_sinyal_pozitif, 'y', linewidth = 2, label = 'Poz')
ax2.plot(x, c_sinyal_cokpozitif, 'g', linewidth = 2, label = 'LPoz')
ax2.set_title("KONTROL SİNYALİ")
ax2.legend()

#GİRİŞ DEĞERLERİ
input_hata = -1
input_d_hata = 1.75

#GİRİLEN DEĞERLERE GÖRE ÜYELİK DERECELERİNİN BULUNMASI
hata_fit_negatif = fuzz.interp_membership(x, hata_negatif, input_hata)
hata_fit_sifir = fuzz.interp_membership(x, hata_sifir, input_hata)

d_hata_fit_negatif = fuzz.interp_membership(x, d_hata_negatif, input_d_hata)
d_hata_fit_sifir = fuzz.interp_membership(x, d_hata_sifir, input_d_hata)
d_hata_fit_pozitif = fuzz.interp_membership(x, d_hata_pozitif, input_d_hata)

#KESİŞİM NOKTALARININ ÇİZDİRİLMESİ
ax0.vlines(input_hata, 0, hata_fit_negatif, linestyles = '--', linewidth = 1, color = 'black')
ax0.vlines(input_hata, 0, hata_fit_sifir, linestyles = '--', linewidth = 1, color = 'black')

ax1.vlines(input_d_hata, 0, d_hata_fit_negatif, linestyles = '--', linewidth = 1, color = 'black')
ax1.vlines(input_d_hata, 0, d_hata_fit_sifir, linestyles = '--', linewidth = 1, color = 'black')
ax1.vlines(input_d_hata, 0, d_hata_fit_pozitif, linestyles = '--', linewidth = 1, color = 'black')

#KURALLAR
rule1 = np.fmin(np.fmin(hata_fit_sifir, d_hata_fit_pozitif),c_sinyal_negatif)  #KURAL-1: EĞER HATA "SIFIR" VE HATA DEĞİŞİMİ "POZİTİF" İSE KONTROL GİRİŞİ "NEGATİF"
rule2 = np.fmin(np.fmin(hata_fit_sifir, d_hata_fit_sifir), c_sinyal_sifir)  #KURAL-2: EĞER HATA "SIFIR" VE HATA DEĞİŞİMİ "SIFIR" İSE KONTROL GİRİŞİ "SIFIR"
rule3 = np.fmin(np.fmin(hata_fit_negatif, d_hata_fit_negatif),c_sinyal_pozitif)  #KURAL-3: EĞER HATA "NEGATİF" VE HATA DEĞİŞİMİ "NEGATİF" İSE KONTROL GİRİŞİ "POZİTİF"
rule4 = np.fmin(np.fmin(hata_fit_negatif, d_hata_fit_sifir), c_sinyal_cokpozitif)  #KURAL-4: EĞER HATA "NEGATİF" VE HATA DEĞİŞİMİ "SIFIR" İSE KONTROL GİRİŞİ "ÇOK POZİTİF"

#BULANIKLAŞTIRILMIŞ KONTROL SİNYALİNİN ÇİZDİRİLMESİ
fig, ax0 = plt.subplots(figsize = (7, 4))
ax0.plot(x, rule1, 'r', label = "Kural1")
ax0.plot(x, rule2, 'b', label = "Kural2")
ax0.plot(x, rule3, 'y', label = "Kural3")
ax0.plot(x, rule4, 'g', label = "Kural4")
ax0.legend()
ax0.set_title("Bulanık Çıkarım")

#BERRAKLAŞTIRMA İŞLEMİ
sonuc = ((np.max(rule1)*fuzz.centroid(x,rule1)) + (np.max(rule2)*fuzz.centroid(x, rule2)) + (np.max(rule3)*fuzz.centroid(x, rule3)) + (np.max(rule4)*fuzz.centroid(x, rule4))) / ((np.max(rule1)+np.max(rule2)+np.max(rule3)+np.max(rule4)))
print("Berraklaştırma İşlemi Sonucu:",sonuc)

plt.show()
