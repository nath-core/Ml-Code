import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")

# Select numerical columns
X = data.select_dtypes(include="number")
X = X.fillna(X.mean())

# Segment customers
kmeans = KMeans(n_clusters=5, random_state=42)
y = kmeans.fit_predict(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree using Entropy (ID3-style)
model = DecisionTreeClassifier(
    criterion="entropy", random_state=42
)
model.fit(X_train, y_train)
p = model.predict(X_test)

# Output
print("Accuracy:", accuracy_score(y_test, p))
print("Feature Importance:", model.feature_importances_)

# Graph
plt.figure(figsize=(12, 6))
plot_tree(model, filled=True, feature_names=X.columns)
plt.show()
