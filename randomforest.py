#import library yang dibutuhkan
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
#load dataset dan hapus kolom Id
df = pd.read_csv('diabetes.csv')
df.drop('Id',axis=1,inplace=True)
#pisahkan antara predictor dengan target
y_train = df['Species']
X_train = df.drop('Species',axis=1)

#tentukan model yang digunakan
#pada artikel ini menggunakan Random Forest
#bisa juga menggunakan model yang lain
model = RandomForestClassifier(n_estimators=100, random_state=2017)
#lihat nilai akurasi model menggunakan cross validasi
score = cross_val_score(model,X_train,y_train,cv=10)
score.mean()

model.fit(X_train,y_train)

with open('../model.pkl','wb') as f:
    pickle.dump(model,f)