# linear_regression
The aim of this project is to introduce you to the basic concept behind machine learning

## Overview
This project implements a simple linear regression model to predict car prices based on mileage. It consists of two main programs:

1. **train.py** - Trains the model using a dataset
2. **predict.py** - Predicts car prices for given mileage values

## Usage

### Training the Model

First, train the model using the provided dataset:

```bash
python3 train.py
```

This will:
- Read the dataset from `data.csv`
- Perform linear regression using gradient descent
- Save the trained parameters (theta0 and theta1) to `model.txt`
- Display progress and final parameters

### Making Predictions

After training, you can predict car prices:

```bash
python3 predict.py
```

The program will:
- Load the trained model from `model.txt`
- Prompt you to enter a mileage value (in km)
- Display the estimated price

Example:
```
Enter mileage (km): 100000
Estimated price: 8351.51
```

## Dataset

The `data.csv` file contains sample data with two columns:
- `km` - Mileage in kilometers
- `price` - Car price

## Algorithm

The implementation uses:
- **Linear Regression**: price = theta0 + theta1 × mileage
- **Gradient Descent**: Iterative optimization algorithm
- **Data Normalization**: Improves convergence speed and stability

## Requirements

- Python 3.x
- No external dependencies (uses only standard library)
