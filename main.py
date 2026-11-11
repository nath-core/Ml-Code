
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
url = 'pima-indians-diabetes-data.csv'
data = pd.read_csv(url)

# Features and target
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Without scaling
unscaled_model = LogisticRegression(solver='liblinear', max_iter=1000)
unscaled_model.fit(x_train, y_train)
unscaled_predictions = unscaled_model.predict(x_test)

# With scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
scaled_model = LogisticRegression(solver='liblinear', max_iter=1000)
scaled_model.fit(x_train_scaled, y_train)
scaled_predictions = scaled_model.predict(x_test_scaled)

# Compare metrics
for model_name, predictions in [
    ('Without Scaling', unscaled_predictions),
    ('With Scaling', scaled_predictions)
]:
    print(model_name,
          'Accuracy:', round(accuracy_score(y_test, predictions), 3),
          'Precision:', round(precision_score(y_test, predictions), 3),
          'Recall:', round(recall_score(y_test, predictions), 3),
          'F1:', round(f1_score(y_test, predictions), 3))

# Accuracy comparison graph
plt.bar(
    x=["Without Scaling", "With Scaling"],
    height=[accuracy_score(y_test, unscaled_predictions),
            accuracy_score(y_test, scaled_predictions)]
)
plt.ylabel("Accuracy")
plt.title("Logistic Regression")
plt.show()
