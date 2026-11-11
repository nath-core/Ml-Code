import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing(as_frame=True).frame

# Select feature and target
input_data = data['AveRooms'].values.reshape(-1, 1)
target_data = data['MedHouseVal'].values

# Remove outliers
valid_rows = input_data[:, 0] < 10
input_data, target_data = input_data[valid_rows], target_data[valid_rows]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Gradient Descent
number_of_samples = len(y_train)
intercept, slope, learning_rate = 0.0, 0.0, 0.01

for _ in range(1000):
    predictions = intercept + slope * X_train.ravel()
    errors = predictions - y_train
    intercept -= learning_rate * errors.mean()
    slope -= learning_rate * np.mean(errors * X_train.ravel())

# Normal Equation
design_matrix = np.c_[np.ones(number_of_samples), X_train]
parameters = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ y_train
normal_intercept, normal_slope = parameters

# Store both models
models = [('Gradient Descent', intercept, slope), ('Normal Equation', normal_intercept, normal_slope)]

# Display parameters
print(f'GD => theta0={intercept:.4f}, theta1={slope:.4f}')
print(f'NE => theta0={normal_intercept:.4f}, theta1={normal_slope:.4f}')

# Evaluate models
for name, t0, t1 in models:
    predictions = t0 + t1 * X_test.ravel()
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print(f'{name:20s} MSE={mse:.4f} RMSE={rmse:.4f} R2={r2:.4f}')

# Visualize regression lines
X_values = np.linspace(X_test.min(), X_test.max(), 200)
figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for axis, (name, t0, t1) in zip(axes, models):
    axis.scatter(X_test.ravel(), y_test, s=8, alpha=0.4, label='Data')
    axis.plot(X_values, t0 + t1 * X_values, color='red', linewidth=2, label='Fitted Line')
    axis.set_title('Regression')
    axis.set_xlabel('Average Rooms (Scaled)')
    axis.set_ylabel('Median House Value')
    axis.legend()
    axis.grid()

plt.suptitle('Linear Regression: Gradient Descent vs Normal Equation')
plt.tight_layout()
plt.show()
