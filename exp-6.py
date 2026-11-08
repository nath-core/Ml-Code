import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
url = 'pima-indians-diabetes-data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)
x, y = data.drop(columns='Outcome'), data['Outcome']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42

)
# Without scaling
unscaled_model = LogisticRegression(solver='liblinear', max_iter=1000)
unscaled_model.fit(x_train, y_train)
# With scaling
scaler = StandardScaler()
x_train_scaled, x_test_scaled = scaler.fit_transform(x_train), scaler.transform(x_test)
scaled_model = LogisticRegression(solver='liblinear', max_iter=1000)
scaled_model.fit(x_train_scaled, y_train)

# Compare metrics
for model_name, trained_model, test_features in [
     ('Without Scaling', unscaled_model, x_test),
    ('With Scaling', scaled_model, x_test_scaled)
]:
     predictions = trained_model.predict(test_features)
     print(model_name,
        'Accuracy:', round(accuracy_score(y_test, predictions), 3),
        'Precision:', round(precision_score(y_test, predictions), 3),
        'Recall:', round(recall_score(y_test, predictions), 3),
        'F1:', round(f1_score(y_test, predictions), 3))
# Weight coefficient comparison graph
feature_positions = np.arange(len(x.columns))
plt.figure(figsize=(10, 5))
plt.bar(feature_positions-0.2, unscaled_model.coef_[0], 0.4, label='Unscaled Weights')
plt.bar(feature_positions+0.2, scaled_model.coef_[0], 0.4, label='Scaled Weights')
plt.axhline(0, color='black', linestyle=':')
plt.xticks(feature_positions, x.columns, rotation=30)
plt.ylabel('Weight Parameter Coefficient Value')
plt.title('Weight Coefficients: Unscaled vs. Standardized Features')
plt.legend()
plt.tight_layout()
plt.show()