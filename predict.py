#!/usr/bin/env python3
"""
Prediction program for car price estimation.
Prompts user for mileage and predicts price using trained model.
"""

import sys
import os


def load_model(filename='model.txt'):
    """Load trained model parameters from file."""
    if not os.path.exists(filename):
        print("Warning: Model file not found. Using default values (theta0=0, theta1=0).")
        print("Please run train.py first to train the model.")
        return 0.0, 0.0
    
    try:
        with open(filename, 'r') as file:
            theta0 = float(file.readline().strip())
            theta1 = float(file.readline().strip())
        return theta0, theta1
    except (IOError, ValueError):
        print("Error: Could not read model file. Using default values.")
        return 0.0, 0.0


def predict_price(mileage, theta0, theta1):
    """Predict price based on mileage using linear regression formula."""
    return theta0 + theta1 * mileage


def get_mileage_input():
    """Prompt user for mileage input."""
    while True:
        try:
            mileage_str = input("Enter mileage (km): ")
            mileage = float(mileage_str)
            
            if mileage < 0:
                print("Error: Mileage cannot be negative. Please try again.")
                continue
            
            return mileage
        except ValueError:
            print("Error: Please enter a valid number.")
        except EOFError:
            print("\nInput cancelled.")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nInput cancelled.")
            sys.exit(0)


def main():
    """Main prediction function."""
    # Load trained model
    theta0, theta1 = load_model()
    
    # Get mileage from user
    mileage = get_mileage_input()
    
    # Predict price
    price = predict_price(mileage, theta0, theta1)
    
    # Display result
    print(f"Estimated price: {price:.2f}")


if __name__ == "__main__":
    main()
