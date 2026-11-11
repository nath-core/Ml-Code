
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = load_diabetes(as_frame=True).frame
x, y = data.drop(columns='target').values, data['target'].values

# Split and scale
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train models
linear_model = LinearRegression().fit(x_train, y_train)
alpha_values = np.logspace(-3, 2, 50)
ridge_model = RidgeCV(alphas=alpha_values, cv=5).fit(x_train, y_train)
lasso_model = LassoCV(alphas=alpha_values, cv=5, max_iter=10000).fit(x_train, y_train)

print('Best Ridge alpha:', ridge_model.alpha_)
print('Best Lasso alpha:', lasso_model.alpha_)

# Evaluate models
models = {'Linear': linear_model, 'Ridge': ridge_model, 'Lasso': lasso_model}
results = []

for model_name, trained_model in models.items():
    predictions = trained_model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    results.append([model_name, mse, rmse, r2])

results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R2'])
print(results_df)

# MSE comparison graph
plt.bar(
    x=results_df['Model'],
    height=results_df['MSE']
)
plt.title("Regression Comparison")
plt.ylabel("MSE")
plt.show()

