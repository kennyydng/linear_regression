#!/usr/bin/env python3
"""
Training program for linear regression model.
Reads dataset and performs linear regression to find theta0 and theta1.
Saves the trained parameters to a file for use in prediction.
"""

import csv
import sys


def read_data(filename):
    """Read the dataset from CSV file."""
    mileages = []
    prices = []
    
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                mileages.append(float(row['km']))
                prices.append(float(row['price']))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except KeyError:
        print("Error: CSV file must have 'km' and 'price' columns.")
        sys.exit(1)
    except ValueError:
        print("Error: Invalid data in CSV file.")
        sys.exit(1)
    
    return mileages, prices


def normalize_data(data):
    """Normalize data to improve gradient descent convergence."""
    if not data:
        return data, 0, 1
    
    mean = sum(data) / len(data)
    std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    
    if std == 0:
        return [0] * len(data), mean, 1
    
    normalized = [(x - mean) / std for x in data]
    return normalized, mean, std


def denormalize_theta(theta0, theta1, mean_x, std_x, mean_y, std_y):
    """Convert normalized theta values back to original scale."""
    original_theta1 = theta1 * (std_y / std_x)
    original_theta0 = mean_y - original_theta1 * mean_x
    return original_theta0, original_theta1


def train_model(mileages, prices, learning_rate=0.01, iterations=1000):
    """
    Train linear regression model using gradient descent.
    Returns theta0 and theta1 parameters.
    """
    # Normalize data for better convergence
    norm_mileages, mean_x, std_x = normalize_data(mileages)
    norm_prices, mean_y, std_y = normalize_data(prices)
    
    # Initialize parameters
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)
    
    # Gradient descent
    for i in range(iterations):
        # Calculate predictions
        predictions = [theta0 + theta1 * x for x in norm_mileages]
        
        # Calculate errors
        errors = [pred - actual for pred, actual in zip(predictions, norm_prices)]
        
        # Update parameters
        theta0_gradient = sum(errors) / m
        theta1_gradient = sum(error * x for error, x in zip(errors, norm_mileages)) / m
        
        theta0 -= learning_rate * theta0_gradient
        theta1 -= learning_rate * theta1_gradient
        
        # Print progress every 100 iterations
        if (i + 1) % 100 == 0:
            mse = sum(e ** 2 for e in errors) / m
            print(f"Iteration {i + 1}: MSE = {mse:.6f}")
    
    # Denormalize parameters to original scale
    theta0, theta1 = denormalize_theta(theta0, theta1, mean_x, std_x, mean_y, std_y)
    
    return theta0, theta1


def save_model(theta0, theta1, filename='model.txt'):
    """Save trained model parameters to file."""
    try:
        with open(filename, 'w') as file:
            file.write(f"{theta0}\n{theta1}\n")
        print(f"\nModel saved to {filename}")
        print(f"theta0 = {theta0}")
        print(f"theta1 = {theta1}")
    except IOError:
        print(f"Error: Could not write to file '{filename}'.")
        sys.exit(1)


def main():
    """Main training function."""
    print("Training linear regression model...")
    
    # Read dataset
    dataset_file = 'data.csv'
    mileages, prices = read_data(dataset_file)
    
    print(f"Loaded {len(mileages)} data points from {dataset_file}")
    
    # Train model
    theta0, theta1 = train_model(mileages, prices)
    
    # Save model
    save_model(theta0, theta1)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
