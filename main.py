import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

iris = load_iris()
x = iris.data[:,2:4]
y = (iris.target == 0).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

model = SVC(kernel='linear',C=1.0)
model.fit(x_train_s,y_train)
pred = model.predict(x_test_s)

print("Accuray: ",round (accuracy_score(y_test,pred),4))
print("Precsion: ",round (precision_score(y_test,pred),4))
print("F1: ",round (f1_score(y_test,pred),4))
print("Recall: ",round (recall_score(y_test,pred),4))
print("Confusion Matrix \n: ",confusion_matrix(y_test,pred))

plt.figure(figsize=(10,7))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow')

xx,yy = np.meshgrid(
    np.linspace(x[:,0].min() - 0.5,x[:,0].max() + 0.5,300),
    np.linspace(x[:,1].min() - 0.5,x[:,1].max() + 0.5,300)
)
grid = scaler.transform(np.c_[xx.ravel(),yy.ravel()])
z = model.decision_function(grid).reshape(xx.shape)

plt.contour(xx,yy,z,levels=[-1,0,1],colors='black',linestyles=['--','-','-'])

sv = scaler.inverse_transform(model.support_vectors_)
plt.scatter(sv[:,0],sv[:,1],s=120,facecolors='none',edgecolors='black',label='Support vector')

plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Support Vector Machine')
plt.legend()
plt.tight_layout()
plt.show()