from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import threading

print("Démarrage du programme")
debut = time.time()
print("Lecture du fichier train")
train = pd.read_csv('data/train.csv')
print("Lecture du fichier test")
test = pd.read_csv('data/test.csv')
KNN = KNeighborsClassifier(5)
print("Création data label")
train_x = train.iloc[0:,1:]
train_y = train.iloc[0:,0]
r = []


print("Etape 1")
x_train,x_test,y_train,y_test=train_test_split(train_x.iloc[0:20000,],train_y.iloc[0:20000,],test_size=0.25)
KNN.fit(x_train, y_train)
for i in range(0, 20000):
    if (time.time() - debut) % 10 == 0 :
        print(str(i*10) + " secondes depis le début")
    r.append(KNN.predict(test.iloc[i,:].values.reshape(1, -1))[0])

print("Etape 2")
x_train,x_test,y_train,y_test=train_test_split(train_x.iloc[20000:41999,],train_y.iloc[20000:41999,],test_size=0.25)
KNN.fit(x_train, y_train)
for i in range(20000, 41999):
    if (time.time() - debut) % 10 == 0 :
        print(str(i*10) + " secondes depis le début 2 ")
    r.append(KNN.predict(test.iloc[i,:].values.reshape(1, -1))[0])




print("Ecriture dans le fichier !")
fichier = pd.read_csv('data/sample_submission.csv')
my_submission = pd.DataFrame({'ImageId': fichier.ImageId, 'Label' : r})
my_submission.to_csv('data\submission.csv', index=False)

print("Temps " + str(time.time() - debut))
