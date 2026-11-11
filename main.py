from unittest import result
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes,fetch_openml,load_iris
import matplotlib.pyplot as plt
from sklearn.externals.array_api_compat.torch import meshgrid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier,plot_tree

iris =load_iris()
x = iris.data[:,2:4]
y = (iris.target==0).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=7)

scalar = StandardScaler()
x_train_s = scalar.fit_transform(x_train)
x_test_s = scalar.transform(x_test)

model = SVC(kernel='linear',C=1.0)
model.fit(x_train_s,y_train)
prediction = model.predict(x_test_s)

plt.scatter(x[:,0],x[:,1],c=y,cmap='coolwarm')

xx,yy = meshgrid(
    np.linspace(x[:,0].min()-0.5,x[:,0].max()+0.5,300),
    np.linspace(x[:,1].min()-0.5,x[:,1].max()+0.5,300)
)
grid = scalar.transform(np.c_[xx.ravel(),yy.ravel()])
z = model.decision_function(grid).reshape(xx.shape)

plt.contour(xx,yy,z,levels=[-1,0,1],colors ='black',linestyles)

sv = scalar.inverse_transform(model.support_vectors_)
plt.scatter(sv[:,0],sv[:,1],s=120,facecolor='none',edgecolor='black')



