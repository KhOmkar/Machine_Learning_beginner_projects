"""
Simple Linear Regression Model
A beginner-friendly program that demonstrates how to train a linear regression model
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def generate_sample_data():
    """Generate sample data for linear regression demonstration"""
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y


def train_linear_regression(X_train, y_train):
    """Train a linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, r2


def plot_results(X_train, y_train, X_test, y_test, model):
    """Plot the training data, test data, and regression line"""
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.6)
    
    # Plot test data
    plt.scatter(X_test, y_test, color='green', label='Test Data', alpha=0.6)
    
    # Plot regression line
    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('linear_regression_plot.png')
    print("Plot saved as 'linear_regression_plot.png'")
    plt.close()


def main():
    """Main function to run the linear regression example"""
    print("=" * 50)
    print("Simple Linear Regression Model")
    print("=" * 50)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data()
    print(f"   Generated {len(X)} data points")
    
    # Split data into training and testing sets
    print("\n2. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train the model
    print("\n3. Training the linear regression model...")
    model = train_linear_regression(X_train, y_train)
    print("   Model trained successfully!")
    
    # Display model parameters
    print("\n4. Model Parameters:")
    print(f"   Coefficient (slope): {model.coef_[0][0]:.4f}")
    print(f"   Intercept: {model.intercept_[0]:.4f}")
    print(f"   Equation: y = {model.coef_[0][0]:.4f}x + {model.intercept_[0]:.4f}")
    
    # Evaluate the model
    print("\n5. Evaluating the model...")
    y_pred, mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Plot results
    print("\n6. Creating visualization...")
    plot_results(X_train, y_train, X_test, y_test, model)
    
    print("\n" + "=" * 50)
    print("Linear Regression completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
