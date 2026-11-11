
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('auto-mpg-dataset.csv').dropna(subset=['displacement', 'mpg'])
X = data[['displacement']]
y = data['mpg']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)

# Linear Regression
linear_model = LinearRegression().fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

results = [['Linear',
            mean_squared_error(y_test, linear_predictions),
            np.sqrt(mean_squared_error(y_test, linear_predictions)),
            r2_score(y_test, linear_predictions)]]

# Polynomial Regression
for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression().fit(X_train_poly, y_train)
    predictions = model.predict(X_test_poly)

    results.append([f'Degree {degree}',
                    mean_squared_error(y_test, predictions),
                    np.sqrt(mean_squared_error(y_test, predictions)),
                    r2_score(y_test, predictions)])

# Display results
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R2'])
print(results_df.round(3))

# Regression graph
xs = pd.DataFrame(np.linspace(X.min().iloc[0], X.max().iloc[0], 300),
                  columns=['displacement'])

plt.scatter(X, y, alpha=0.2, label='Actual Data')
plt.plot(xs, linear_model.predict(xs), label='Degree 1')

for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    model = LinearRegression().fit(X_train_poly, y_train)
    plt.plot(xs, model.predict(poly.transform(xs)), label=f'Degree {degree}')

plt.xlabel("Engine Displacement")
plt.ylabel("MPG")
plt.title("Polynomial Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

