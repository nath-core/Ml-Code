import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('auto-mpg.csv').dropna(subset=['displacement', 'mpg'])
X = df[['displacement']].values
y = df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)

# Evaluation metrics
def metrics(y, pred):
    mse = mean_squared_error(y, pred)
    return mse, np.sqrt(mse), r2_score(y, pred)

# Linear Regression
lin_model = LinearRegression().fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
mse_lin, rmse_lin, r2_lin = metrics(y_test, y_pred)
print(f'Linear: MSE={mse_lin:.4f} RMSE={rmse_lin:.4f} R2={r2_lin:.4f}')

# Polynomial Regression
degrees = [1, 2, 3, 4, 6, 8]
results, models = [], {}

for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression().fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    mse, rmse, r2 = metrics(y_test, y_pred)

    results.append((d, mse, rmse, r2))
    models[d] = (poly, model)
    print(f'Degree {d}: MSE={mse:.4f} RMSE={rmse:.4f} R2={r2:.4f}')

# Model Comparison
results_df = pd.DataFrame(results, columns=['Degree', 'MSE', 'RMSE', 'R2'])
results_df.loc[-1] = ['Linear baseline', mse_lin, rmse_lin, r2_lin]
print('\n', results_df.sort_index().reset_index(drop=True))

# R² Comparison
plt.figure(figsize=(7, 4))
plt.plot(degrees, [r[3] for r in results], 'o-', color='crimson')
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Performance vs Polynomial Degree')
plt.grid(alpha=0.3)
plt.show()

# Regression Curves
xs = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
plt.figure(figsize=(9, 6))
plt.scatter(X_test, y_test, s=15, alpha=0.4, label='Test Data')
plt.plot(xs, lin_model.predict(xs), 'k--', label='Linear')

for d, color in zip([2, 4, 8], ['crimson', 'green', 'orange']):
    poly, model = models[d]
    plt.plot(xs, model.predict(poly.transform(xs)), color=color, label=f'Degree {d}')

plt.xlabel('Engine Displacement (cu. in.)')
plt.ylabel('Miles Per Gallon (MPG)')
plt.title('Polynomial vs Linear Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()