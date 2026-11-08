import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data / 255.0, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
print("Dataset processing complete.")

history_loss = {}
test_accuracies = {}
training_times = {}
activations = ['logistic', 'tanh', 'relu']

# Train and evaluate networks
for act in activations:
    print(f"Training network using activation: {act}")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation=act, max_iter=25, random_state=42)
    start_time = time.time()
    mlp.fit(X_train, y_train)
    training_times[act] = time.time() - start_time
    history_loss[act] = mlp.loss_curve_
    test_accuracies[act] = accuracy_score(y_test, mlp.predict(X_test))

# Display accuracy and training time
for act in activations:
    name = 'Sigmoid' if act == 'logistic' else act.capitalize()
    print(f"Accuracy for {name}: {test_accuracies[act]:.4f}")
    print(f"Training time for {name}: {training_times[act]:.2f} seconds")

# Plot loss convergence
plt.figure(figsize=(10, 5))
for act in activations:
    name = 'Sigmoid' if act == 'logistic' else act.capitalize()
    plt.plot(history_loss[act], label=name)

plt.title('Loss Convergence Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.grid()
plt.show()