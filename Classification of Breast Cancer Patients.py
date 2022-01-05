# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:43:44 2020

@author: Lenovo
"""
from sklearn.preprocessing import LabelEncoder
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

# untuk menghilangkan warning untuk value labeling pada console ketika menampilkan output
pandas.options.mode.chained_assignment = None  # default='warn'

# membaca file dan direktori tempat iris.data.csv disimpan
direktori = "C:/Users/Lenovo/Documents/Penting/Kuliah/Data Mining/breast-cancer.data"

# memberi nama variabel
names = ['Class', 'Age', 'Menopause', 'Tumor-Size', 'Inv-Nodes', 
         'Node-Caps', 'Deg-Malig', 'Breast', 'Breast-Quad','Irradiant']

# membaca data dengan library panda
datacancer = pandas.read_csv(direktori, names=names, na_values=['?'])

# menghilangkan missing values berupa '?' pada dataset
new_data = datacancer.dropna(axis = 0, how ='any')

# melakukan labeling pada atribut yang harus dirubah
le = LabelEncoder()

le.fit(new_data['Class'])
new_data['Class'] = le.transform(new_data['Class'])

le.fit(new_data['Age'])
new_data['Age'] = le.transform(new_data['Age'])

le.fit(new_data['Menopause'])
new_data['Menopause'] = le.transform(new_data['Menopause'])

le.fit(new_data['Tumor-Size'])
new_data['Tumor-Size'] = le.transform(new_data['Tumor-Size'])

le.fit(new_data['Inv-Nodes'])
new_data['Inv-Nodes'] = le.transform(new_data['Inv-Nodes'])

le.fit(new_data['Node-Caps'])
new_data['Node-Caps'] = le.transform(new_data['Node-Caps'])

le.fit(new_data['Breast'])
new_data['Breast'] = le.transform(new_data['Breast'])

le.fit(new_data['Breast-Quad'])
new_data['Breast-Quad'] = le.transform(new_data['Breast-Quad'])

le.fit(new_data['Irradiant'])
new_data['Irradiant'] = le.transform(new_data['Irradiant'])

# menyiapkan fitur untuk digunakan dalam training dan testing
x = new_data.drop('Class',axis=1)
y = new_data.Class

# membagi data training dan data testing dengan test size 20% dan training size 80%
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size = 0.2, random_state = 0)

print('\nKLASIFIKASI DENGAN DECISION TREE\n')

# proses modeling Decision tree dengan kedalaman 4
clf = DecisionTreeClassifier(max_depth = 4)
clf = clf.fit(x_train,y_train)

y_predict = clf.predict(x_test)

# melihat hasil kelas dataset setelah diprediksi dengan Decision Tree
y_predict = clf.predict(x_test)
print('prediksi untuk X Decision Tree = ',y_predict)

# menghitung akurasi klasifikasi Decision Tree dengan penggunaan metrics
accDCT = metrics.accuracy_score(y_test, y_predict)*100
print("Accuracy Decision Tree : %.2f" %accDCT,'%')

# Proses visualisasi Decision Tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (16,9), dpi=300)

tree.plot_tree(clf,
              filled=True, 
              rounded=True, 
              fontsize=8)

# menyimpan gambar hasil visualisasi Decision Tree
fig.savefig('tree-breast-cancer.png')