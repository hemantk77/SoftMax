import matplotlib
matplotlib.use('Agg') # Use the non-interactive 'Agg' backend
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot_encode(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def initialize_parameters(n_features, n_classes):
    weights = np.random.randn(n_features, n_classes) * 0.01
    bias = np.zeros((1, n_classes))
    return weights, bias


def forward_pass(X, weights, bias):
    z = X.dot(weights) + bias
    y_pred = softmax(z)
    return z, y_pred


def backward_pass(X, y_true, y_pred):
    n_samples = X.shape[0]
    dw = (1 / n_samples) * X.T.dot(y_pred - y_true)
    db = (1 / n_samples) * np.sum(y_pred - y_true, axis=0, keepdims=True)
    return dw, db


def update_parameters(weights, bias, dw, db, learning_rate):
    weights -= learning_rate * dw
    bias -= learning_rate * db
    return weights, bias


def train_softmax_classifier(X, y, learning_rate=0.01, max_iterations=1000):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    weights, bias = initialize_parameters(n_features, n_classes)
    y_encoded = one_hot_encode(y, n_classes)
    cost_history = []

    for i in range(max_iterations):
        _, y_pred = forward_pass(X, weights, bias)
        cost = cross_entropy_loss(y_encoded, y_pred)
        cost_history.append(cost)
        dw, db = backward_pass(X, y_encoded, y_pred)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate)
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.4f}")

    return weights, bias, cost_history


def predict_proba(X, weights, bias):
    _, y_pred = forward_pass(X, weights, bias)
    return y_pred


def predict(X, weights, bias):
    probabilities = predict_proba(X, weights, bias)
    return np.argmax(probabilities, axis=1)


def plot_metrics(cost_history, y_test, y_pred_custom):
    """Saves the training loss curve and confusion matrix as image files."""
    # Save the Training Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title("Training Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.savefig("softmax_loss_curve.png") # Save the plot as a file
    plt.close() # Close the figure to free up memory

    # Save the Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_custom)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("softmax_confusion_matrix.png") # Save the plot as a file
    plt.close() # Close the figure

    # Print the text-based reports to the terminal
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_custom))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_custom))


def main():
    # Load the dataset
    df = pd.read_csv("multinomial_classification_dataset.csv")
    X = df.drop('label', axis=1).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'].values)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    print("Training softmax classifier...")
    weights, bias, cost_history = train_softmax_classifier(
        X_train_scaled, y_train, learning_rate=0.1, max_iterations=1000
    )

    # Predict and evaluate
    print("\nEvaluating...")
    y_pred_custom = predict(X_test_scaled, weights, bias)
    accuracy = accuracy_score(y_test, y_pred_custom)
    print(f"Custom Softmax Accuracy: {accuracy:.4f}")

    # Plot and report
    plot_metrics(cost_history, y_test, y_pred_custom)


if __name__ == "__main__":
    main()
