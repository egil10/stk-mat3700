"""
STK-MAT3700 Assignment 1 - Problem 1: Markowitz Portfolio Optimization
Parts a, b, c, d - Complete solution with plots saved as PDF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import os

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Load data from CSV
print("Loading stock data from CSV...")
data = pd.read_csv('data/finance_data.csv', sep=';', decimal=',')

# Clean column names and set index
data.columns = ['Trading_day', 'Energy', 'Agro', 'Tech', 'Pharma', 'SoftDrink']
data = data.set_index('Trading_day')

print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Calculate daily returns
returns = data.pct_change().dropna()
print(f"Returns shape: {returns.shape}")

# Calculate annualized statistics
annual_returns = returns.mean() * 252
annual_volatilities = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252
corr_matrix = returns.corr()

print("\n=== PROBLEM 1a: Expected Returns and Volatility ===")
print("\nExpected Returns (Annualized):")
for col in data.columns:
    print(f"{col}: {annual_returns[col]:.4f} ({annual_returns[col]*100:.2f}%)")

print("\nVolatilities (Annualized):")
for col in data.columns:
    print(f"{col}: {annual_volatilities[col]:.4f} ({annual_volatilities[col]*100:.2f}%)")

# Plot distributions for 1a
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
plt.savefig('plots/problem1a_distributions.pdf')
plt.show()

print("\n=== PROBLEM 1b: Correlation and Covariance ===")
print("\nCorrelation Matrix:")
print(corr_matrix.round(4))

print("\nCovariance Matrix:")
print(cov_matrix.round(6))

# Plot correlation matrix
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
import seaborn as sns
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Asset Return Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/problem1b_correlation.pdf')
plt.show()

# Portfolio optimization functions
def calculate_portfolio_stats(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

def optimize_portfolio(mean_returns, cov_matrix, target_return=None, risk_free_rate=0.02):
    n_assets = len(mean_returns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x * mean_returns) - target_return})
        result = minimize(portfolio_volatility, 
                        x0=np.array([1/n_assets] * n_assets),
                        args=(cov_matrix,),
                        method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(negative_sharpe_ratio,
                        x0=np.array([1/n_assets] * n_assets),
                        args=(mean_returns, cov_matrix, risk_free_rate),
                        method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

def get_efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    efficient_portfolios = []
    
    for target_ret in target_returns:
        result = optimize_portfolio(mean_returns, cov_matrix, target_return=target_ret)
        if result.success:
            weights = result.x
            portfolio_return, portfolio_volatility = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
            efficient_portfolios.append({
                'weights': weights,
                'return': portfolio_return,
                'volatility': portfolio_volatility
            })
    
    return efficient_portfolios

def find_minimum_variance_portfolio(mean_returns, cov_matrix):
    n_assets = len(mean_returns)
    
    def portfolio_volatility_func(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    result = minimize(portfolio_volatility_func,
                    x0=np.array([1/n_assets] * n_assets),
                    method='SLSQP',
                    bounds=tuple((0, 1) for _ in range(n_assets)),
                    constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}])
    
    if result.success:
        weights = result.x
        portfolio_return, portfolio_vol = calculate_portfolio_stats(weights, mean_returns, cov_matrix)
        return {
            'weights': weights,
            'return': portfolio_return,
            'volatility': portfolio_vol
        }
    return None

print("\n=== PROBLEM 1c: Efficient Frontier ===")

# Generate efficient frontier
efficient_portfolios = get_efficient_frontier(annual_returns, cov_matrix)
min_var_portfolio = find_minimum_variance_portfolio(annual_returns, cov_matrix)

if min_var_portfolio:
    print(f"Minimum Variance Portfolio:")
    print(f"  Expected Return: {min_var_portfolio['return']:.4f} ({min_var_portfolio['return']*100:.2f}%)")
    print(f"  Volatility: {min_var_portfolio['volatility']:.4f} ({min_var_portfolio['volatility']*100:.2f}%)")
    print(f"  Weights:")
    for i, col in enumerate(data.columns):
        print(f"    {col}: {min_var_portfolio['weights'][i]:.4f} ({min_var_portfolio['weights'][i]*100:.2f}%)")

# Plot efficient frontier
plt.figure(figsize=(12, 8))

# Plot efficient frontier
returns_ef = [p['return'] for p in efficient_portfolios]
volatilities_ef = [p['volatility'] for p in efficient_portfolios]
plt.plot(volatilities_ef, returns_ef, 'b-', linewidth=2, label='Efficient Frontier')

# Plot individual assets
individual_returns = annual_returns.values
individual_volatilities = np.sqrt(np.diag(cov_matrix.values))

plt.scatter(individual_volatilities, individual_returns, 
           s=100, c='red', alpha=0.7, label='Individual Assets')

# Add asset labels
for i, col in enumerate(data.columns):
    plt.annotate(col, (individual_volatilities[i], individual_returns[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# Plot minimum variance portfolio
if min_var_portfolio:
    plt.scatter(min_var_portfolio['volatility'], min_var_portfolio['return'],
               s=150, c='green', marker='*', label='Min Variance Portfolio')

plt.xlabel('Volatility (σ)')
plt.ylabel('Expected Return (μ)')
plt.title('Efficient Frontier and Individual Assets')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/problem1c_efficient_frontier.pdf')
plt.show()

print("\n=== PROBLEM 1d: 4-Asset vs 5-Asset Comparison ===")

# Remove one asset (SoftDrink) and compare
assets_4 = data.columns[:-1]  # Remove last asset
returns_4 = returns[assets_4]
mean_returns_4 = returns_4.mean() * 252
cov_matrix_4 = returns_4.cov() * 252

print(f"Removing {data.columns[-1]} from portfolio...")
print(f"Remaining assets: {list(assets_4)}")

# Generate efficient frontier for 4 assets
efficient_portfolios_4 = get_efficient_frontier(mean_returns_4, cov_matrix_4)
min_var_portfolio_4 = find_minimum_variance_portfolio(mean_returns_4, cov_matrix_4)

if min_var_portfolio_4:
    print(f"Minimum Variance Portfolio (4 assets):")
    print(f"  Expected Return: {min_var_portfolio_4['return']:.4f} ({min_var_portfolio_4['return']*100:.2f}%)")
    print(f"  Volatility: {min_var_portfolio_4['volatility']:.4f} ({min_var_portfolio_4['volatility']*100:.2f}%)")
    print(f"  Weights:")
    for i, col in enumerate(assets_4):
        print(f"    {col}: {min_var_portfolio_4['weights'][i]:.4f} ({min_var_portfolio_4['weights'][i]*100:.2f}%)")

# Plot comparison
plt.figure(figsize=(12, 8))

# 5-asset frontier
returns_5 = [p['return'] for p in efficient_portfolios]
volatilities_5 = [p['volatility'] for p in efficient_portfolios]
plt.plot(volatilities_5, returns_5, 'b-', linewidth=2, label='5 Assets')

# 4-asset frontier
returns_4 = [p['return'] for p in efficient_portfolios_4]
volatilities_4 = [p['volatility'] for p in efficient_portfolios_4]
plt.plot(volatilities_4, returns_4, 'r--', linewidth=2, label='4 Assets')

# Plot minimum variance portfolios
if min_var_portfolio:
    plt.scatter(min_var_portfolio['volatility'], min_var_portfolio['return'],
               s=150, c='blue', marker='*', label='Min Var (5 assets)')

if min_var_portfolio_4:
    plt.scatter(min_var_portfolio_4['volatility'], min_var_portfolio_4['return'],
               s=150, c='red', marker='*', label='Min Var (4 assets)')

plt.xlabel('Volatility (σ)')
plt.ylabel('Expected Return (μ)')
plt.title('Efficient Frontier: 5 Assets vs 4 Assets')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/problem1d_comparison.pdf')
plt.show()

print("\n=== SUMMARY ===")
print("All plots saved to plots/ directory:")
print("- problem1a_distributions.pdf")
print("- problem1b_correlation.pdf") 
print("- problem1c_efficient_frontier.pdf")
print("- problem1d_comparison.pdf")
print("\nProblem 1 completed successfully!")