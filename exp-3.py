import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
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
x_train, x_test = scaler.fit_transform(x_train), scaler.transform(x_test)

# Train models

linear_model = LinearRegression().fit(x_train, y_train)
alpha_values = np.logspace(-3, 2, 50)
ridge_cv_model = RidgeCV(alphas=alpha_values, cv=5).fit(x_train, y_train)
lasso_cv_model = LassoCV(alphas=alpha_values, cv=5, max_iter=10000).fit(x_train, y_train)
ridge_model = Ridge(alpha=ridge_cv_model.alpha_).fit(x_train, y_train)
lasso_model = Lasso(alpha=lasso_cv_model.alpha_, max_iter=10000).fit(x_train, y_train)

print('Best Ridge alpha:', ridge_cv_model.alpha_)
print('Best Lasso alpha:', lasso_cv_model.alpha_)

# Evaluate models

models = {'Linear': linear_model, 'Ridge': ridge_model, 'Lasso': lasso_model}
results = []

for model_name, trained_model in models.items():
    predictions = trained_model.predict(x_test)
    mse, r2 = mean_squared_error(y_test, predictions), r2_score(y_test, predictions)
    results.append([model_name, mse, np.sqrt(mse), r2])

results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R2'])
print(results_df.to_string(index=False))

# Graph 1: Coefficient comparison

feature_names = data.drop(columns='target').columns
feature_positions = np.arange(len(feature_names))
plt.bar(feature_positions-0.25, linear_model.coef_, 0.25, label='Linear')
plt.bar(feature_positions, ridge_model.coef_, 0.25, label='Ridge')
plt.bar(feature_positions+0.25, lasso_model.coef_, 0.25, label='Lasso')
plt.xticks(feature_positions, feature_names, rotation=45)
plt.ylabel('Coefficient Value')
plt.title('Linear vs Ridge vs Lasso Coefficients')
plt.legend()
plt.tight_layout()
plt.show()

