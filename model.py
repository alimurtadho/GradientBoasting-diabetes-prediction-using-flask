## Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, recall_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import pickle
warnings.filterwarnings('ignore')
data = pd.read_csv('diabetes.csv')
#Feature Preprocessing
for col in ['BloodPressure', 'Glucose','SkinThickness','Insulin','BMI','Age']:
    for target in data.Outcome.unique():
        mask = (data[col] != 0) & (data['Outcome'] == target)
        data[col][(data[col] == 0) & (data['Outcome'] == target)] = data[col][(data[col] == 0) & (data['Outcome'] == target)].replace(0,data[mask][col].mean())
        
#Extract data
X = data.iloc[:,0:8]
y = data.iloc[:,-1]
print('x',X)
print('x',y)
#X2 = X.copy()
#X2['Glucose<125'] = X2['Glucose']<125
#X2['Glucose<125'] = X2['Glucose<125'].astype(int)
#X_train,X_test,y_train,y_test = train_test_split(X2, y, random_state=0,stratify=y)
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)
scaler = StandardScaler()
scaler.fit(X_train,y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
param_grid = {'n_estimators': [100,200,300,400,500], 'max_depth': [2,3,4,5]}
gbrt = GradientBoostingClassifier(random_state=0)
grid_gbrt = GridSearchCV(gbrt, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid_gbrt.fit(X_train_scaled, y_train)
gbrt = GradientBoostingClassifier(random_state=0, max_depth=grid_gbrt.best_params_['max_depth'],
                                 n_estimators=grid_gbrt.best_params_['n_estimators'])
gbrt.fit(X_train_scaled,y_train)
# Saving model to disk
pickle.dump(gbrt, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6, 4, 5, 7, 8, 9]]))