
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tkinter import Tk, filedialog


print("HESAPLAMALAR RandomForestRegressor a göre yapılmıştır.\n")
# Dosya seçimi için Tkinter dosya açma penceresi
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

# Excel dosyasını yükleme
data = pd.read_excel(file_path)


print("-----------1.soru---------------")
le = LabelEncoder()

data['Cinsiyet'] = le.fit_transform(data['Cinsiyet'])
data['Ebeveyn Egitim Seviyesi'] = le.fit_transform(data['Ebeveyn Egitim Seviyesi'])
data['Okul Yemekhanesi'] = le.fit_transform(data['Okul Yemekhanesi'])
data['Ozel Ders'] = le.fit_transform(data['Ozel Ders'])

X = data.drop(['Okuma', 'Yazma', 'Matematik'], axis=1) 
y = data[['Okuma', 'Yazma', 'Matematik']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Modeli eğitme
model.fit(X_train, y_train)

# Tahmin yap
predictions = model.predict(X_test)

# Feature importance hesapla
importances = model.feature_importances_
print(f'Özel dersin genel başarıya etkisi:\n {importances[X.columns.get_loc("Ozel Ders")]}')
print("Yani özel dersin genel başarıya etkisi pozitiftir.")

print("\n-----------2.soru---------------")
# One-hot encoding yapma
data_encoded = pd.get_dummies(data, columns=['Ebeveyn Egitim Seviyesi', 'Cinsiyet','Okul Yemekhanesi','Ozel Ders'])

A_encoded = data_encoded.drop(['Okuma', 'Yazma', 'Matematik'], axis=1)
b = data[['Okuma', 'Yazma', 'Matematik']]

# Veriyi eğitim ve test setlerine ayırma
A_train_encoded, A_test_encoded, b_train, b_test = train_test_split(A_encoded, b, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(A_train_encoded, b_train)

# Feature importance hesapla.
importances_encoded = model.feature_importances_

# Her bir özelliğin önemini yazdır.
for feature, importance in zip(A_encoded.columns, importances_encoded):
    print(f'{feature}: {importance}')

#eğitim seviyesinde 0:lise,1:önlisans,2:lisans,3:yüksek lisans demek,diğerleri için 0:yok,1: var demek

print("'Ebeveyn Egitim Seviyesi_2' ve 'Ebeveyn Egitim Seviyesi_3' değerlerinin etkisi çok düşük görünüyor. Bu durumda, bu iki ebeveyn eğitim seviyesinin ( Lisans ve Yuksek Lisans seviyeleri) başarı üzerinde çok fazla etkisi olmadığını söyleyebiliriz. Aynı şekilde, 'Ebeveyn Egitim Seviyesi_0' ( Lise seviyesi) da düşük bir etkiye sahip.")
print("\n")
print("-----------3.soru---------------")
# Veri setini X ve y olarak bölme
X = data.drop(columns=['Matematik', 'Okuma', 'Yazma'])
y = data[['Matematik', 'Okuma', 'Yazma']].mean(axis=1)

# Random Forest modeli oluşturma
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Modeli eğitme
model.fit(X, y)

# Özellik önemlerini alma
importances = model.feature_importances_

# Özelliklerin isimlerini alma
features = X.columns

# Özellikler ve önemlerini bir DataFrame'e dönüştürme
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Öneme göre sıralama
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

#çıktıya göre:
print("Okul Yemekhanesi özelliği, diğer tüm özelliklere kıyasla en yüksek öneme sahip olan özelliği temsil eder. Bu, Random Forest modeline göre, Okul Yemekhanesinin öğrencinin başarısını belirlemede en önemli faktör olduğunu gösterir. \"Ebeveyn Egitim Seviyesi\" ve \"Ozel Ders\" özellikleri sırasıyla ikinci ve üçüncü en önemli özelliklerdir. 'Cinsiyet' özelliği ise bu dört faktör arasında en düşük öneme sahip olanıdır.")
print("\n-----------4.soru---------------")
# Spearman korelasyon analizi
spearman_correlation = data[['Okuma', 'Yazma', 'Matematik']].corr(method='spearman')
print("Spearman Korelasyon Analizi:\n", spearman_correlation)
print("\n")
spearman_correlation = data[['Okuma', 'Yazma']].corr(method='spearman')
print("Okumanın yazma üzerindeki etkisi: ", spearman_correlation.loc['Okuma', 'Yazma'])
spearman_correlation = data[['Okuma', 'Matematik']].corr(method='spearman')
print("Okumanın matematik üzerindeki etkisi: ", spearman_correlation.loc['Okuma','Matematik'])
print("\n Okuma puanının yazma ve matematik puanları üzerindeki etkisini gösterir. Okumanın yazma üzerindeki etkisi 0.9489 olarak hesaplanmıştır, bu da okuma puanının yazma puanıyla yüksek bir pozitif korelasyona sahip olduğunu gösterir. Okumanın matematik üzerindeki etkisi ise 0.8041 olarak hesaplanmıştır, bu da okuma puanının matematik puanıyla da pozitif bir ilişkisi olduğunu, ancak yazma ile olan ilişkisine göre biraz daha zayıf olduğunu gösterir. ") 