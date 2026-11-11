import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load Iris dataset
iris = load_iris()
X = iris.data[:, 2:4]  # Petal length and width
y = (iris.target == 0).astype(int)  # Setosa vs Non-Setosa

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Train Linear SVM
model = SVC(kernel='linear', C=1.00)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

# 5. Evaluate model
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred), 4))
print("Recall:", round(recall_score(y_test, pred), 4))
print("F1-Score:", round(f1_score(y_test, pred), 4))
print("Confusion Matrix:  \n", confusion_matrix(y_test, pred))

# 6. Plot decision boundary and margins
#plt.figure(figsize=(8, 6))
#plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Setosa')
#plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Non-Setosa')

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='rainbow')


# Generate grid in original feature scale, then scale it
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300)
)
grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
Z = model.decision_function(grid).reshape(xx.shape)

# Decision boundary (0) and margins (-1, +1)
plt.contour(xx, yy, Z, levels=[-1, 0, 1],
            colors='black', linestyles=['--', '-', '--'])

# Highlight support vectors
sv = scaler.inverse_transform(model.support_vectors_)
plt.scatter(sv[:, 0], sv[:, 1], s=120, facecolors='none',
            edgecolors='black', label='Support Vectors')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Linear SVM: Setosa vs Non-Setosa')
plt.legend()
plt.tight_layout()
plt.show()
