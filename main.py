import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

data = pd.read_csv("online-retail.csv" , encoding="ISO-8859-1")

x= data.select_dtypes(include="number")
x = x.fillna(x.mean())

kmeans = KMeans(n_clusters=5,random_state=42)
y = kmeans.fit_predict(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

model = DecisionTreeClassifier(criterion="entropy",random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("Accurcay: ",accuracy_score(y_test,y_pred))
print("Feature importance: ",model.feature_importances_)

plt.figure(figsize=(10,7))
plot_tree(model,filled=True,feature_names = x.columns,rounded=True,fontsize=10)
plt.show()