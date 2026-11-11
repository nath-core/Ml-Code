
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load and preprocess dataset
df = pd.read_csv("../data/online-retail.csv", encoding='latin1')
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# Calculate RFM values
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSpend': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalSpend': 'MonetaryValue'
})

# Create customer tiers
threshold = rfm['Frequency'].quantile(0.75)
rfm['IsFrequentBuyer'] = (rfm['Frequency'] > threshold).astype(int)
rfm['Recency_Tier'] = pd.qcut(rfm['Recency'], 3, labels=[0, 1, 2]).astype(int)
rfm['Monetary_Tier'] = pd.qcut(rfm['MonetaryValue'], 3, labels=[0, 1, 2]).astype(int)

# Split dataset
X = rfm[['Recency_Tier', 'Monetary_Tier']]
y = rfm['IsFrequentBuyer']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Decision Tree using Entropy
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output
print("Accuracy:", accuracy_score(y_test, y_pred))

# Decision Tree Visualization
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=X.columns,
          class_names=['Not Frequent', 'Frequent'],
          filled=True, rounded=True, fontsize=10)
plt.title('Customer Segmentation Decision Tree')
plt.show()

# Feature Importance
feature_importance = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:\n", feature_importance)

feature_importance.plot(kind='bar', color='teal', figsize=(8, 4))
plt.ylabel('Feature Importance')
plt.title('RFM Feature Importance')
plt.tight_layout()
plt.show()

