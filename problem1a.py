"""
STK-MAT3700 Assignment 1 - Problem 1a
Load stock data from CSV, calculate returns and volatility, plot distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load data from CSV
print("Loading stock data from CSV...")
data = pd.read_csv('data/finance_data.csv', sep=';', decimal=',')

# Clean column names and set index
data.columns = ['Trading_day', 'Energy', 'Agro', 'Tech', 'Pharma', 'SoftDrink']
data = data.set_index('Trading_day')

print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"Date range: Trading days 1 to {len(data)}")

# Calculate daily returns
returns = data.pct_change().dropna()
print(f"Returns shape: {returns.shape}")

# Calculate annualized statistics (assuming 252 trading days per year)
annual_returns = returns.mean() * 252
annual_volatilities = returns.std() * np.sqrt(252)

print("\n=== PROBLEM 1a RESULTS ===")
print("\nExpected Returns (Annualized):")cls

for col in data.columns:
    print(f"{col}: {annual_returns[col]:.4f} ({annual_returns[col]*100:.2f}%)")

print("\nVolatilities (Annualized):")
for col in data.columns:
    print(f"{col}: {annual_volatilities[col]:.4f} ({annual_volatilities[col]*100:.2f}%)")

# Plot distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(data.columns):
    ax = axes[i]
    asset_returns = returns[col].dropna()
    
    # Plot histogram
    ax.hist(asset_returns, bins=30, density=True, alpha=0.7, color='skyblue', label='Empirical')
    
    # Plot normal distribution
    mu, sigma = asset_returns.mean(), asset_returns.std()
    x = np.linspace(asset_returns.min(), asset_returns.max(), 100)
    normal_dist = norm.pdf(x, mu, sigma)
    ax.plot(x, normal_dist, 'r-', linewidth=2, label='Normal fit')
    
    ax.set_title(f'{col} Returns Distribution')
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

print("\nDistribution plots created!")
print("Blue = Empirical data, Red = Normal distribution fit")