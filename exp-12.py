import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load Fashion MNIST dataset
fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X = fmnist.data
y = fmnist.target.astype(int)
# 2. Downsample for speed (Optional, but highly recommended for KNN)
# KNN has O(N) prediction time. Evaluating all 70k samples takes massive CPU memory/time.
# We will use 10,000 samples for training and 2,000 for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10000, test_size=2000, random_state=42, stratify=y
)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# SVM training on the full 60,000-image dataset with non-linear kernels is very slow,
# so a stratified subsample is used to keep the lab exercise tractable.
X_sub, _, y_sub, _ = train_test_split(
X_train.reshape(len(X_train), -1), y_train,
train_size=6000, stratify=y_train, random_state=42)
# Normalize pixel values to [0, 1]
X_sub = X_sub.astype('float32') / 255.0
# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub)
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Reduce 784 pixel features to 50 principal components to speed up SVM training
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f'Explained variance retained: {pca.explained_variance_ratio_.sum():.4f}')

start = time.time()
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_pca, y_train)
time_linear = time.time() - start
y_pred_linear = svm_linear.predict(X_test_pca)

start = time.time()
svm_poly = SVC(kernel='poly', degree=3, C=1.0, gamma='scale')
svm_poly.fit(X_train_pca, y_train)
time_poly = time.time() - start
y_pred_poly = svm_poly.predict(X_test_pca)

start = time.time()
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_pca, y_train)
time_rbf = time.time() - start
y_pred_rbf = svm_rbf.predict(X_test_pca)

def evaluate(name, y_true, y_pred, train_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return [name, acc, prec, rec, f1, train_time]
results = []
results.append(evaluate('Linear', y_test, y_pred_linear, time_linear))
results.append(evaluate('Polynomial', y_test, y_pred_poly, time_poly))
results.append(evaluate('RBF', y_test, y_pred_rbf, time_rbf))
results_df = pd.DataFrame(results, columns=['Kernel', 'Accuracy', 'Precision', 'Recall', 'F1', 'Train Time (s)'])
print(results_df)
