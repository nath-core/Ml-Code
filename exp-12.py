import pandas as pd
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
X, y = data.data, data.target.astype(int)

# Select 6000 samples
X, _, y, _ = train_test_split(X, y, train_size=6000, random_state=42, stratify=y)

# Normalize and split
X = X.astype('float32') / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=50, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(f'Explained variance retained: {pca.explained_variance_ratio_.sum():.4f}')

# Train and evaluate SVM kernels
models = [('Linear', SVC(kernel='linear', C=1.0)),
          ('Polynomial', SVC(kernel='poly', degree=3, C=1.0, gamma='scale')),
          ('RBF', SVC(kernel='rbf', C=1.0, gamma='scale'))]

results = []
for name, model in models:
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test)
    results.append([name, accuracy_score(y_test, y_pred),
                    precision_score(y_test, y_pred, average='macro'),
                    recall_score(y_test, y_pred, average='macro'),
                    f1_score(y_test, y_pred, average='macro'), train_time])

# Display results
results_df = pd.DataFrame(results, columns=['Kernel', 'Accuracy', 'Precision', 'Recall', 'F1', 'Train Time (s)'])
print(results_df)
