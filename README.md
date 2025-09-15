# STK-MAT3700/4700 Assignment 1: Portfolio Optimization and Black-Scholes Options Pricing

This repository contains the complete solution for the mandatory assignment in STK-MAT3700/4700.

## Assignment Overview

The assignment consists of two main problems:

### Problem 1: Markowitz Portfolio Optimization
- Download and analyze 5 stock price series
- Calculate expected returns and volatilities
- Plot empirical vs normal distributions
- Calculate correlation and covariance matrices
- Generate efficient frontier
- Find minimum variance portfolio
- Compare 5-asset vs 4-asset portfolios

### Problem 2: Black-Scholes Options Pricing
- Implement Black-Scholes formula
- Calculate option prices for different volatilities and strikes
- Analyze implied volatility patterns
- Compare theoretical vs realistic market behavior

## Files Included

- `stk_mat3700_assignment1.py` - Complete Python script with all solutions
- `stk_mat3700_assignment1.ipynb` - Jupyter notebook version for interactive analysis
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Setup Instructions

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Assignment:**
   
   **Option A: Python Script**
   ```bash
   python stk_mat3700_assignment1.py
   ```
   
   **Option B: Jupyter Notebook**
   ```bash
   jupyter notebook stk_mat3700_assignment1.ipynb
   ```

## Required Packages

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scipy` - Optimization and statistical functions
- `yfinance` - Stock data download

## Data Sources

The script automatically downloads stock data from Yahoo Finance for the following symbols:
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google/Alphabet)
- AMZN (Amazon)
- TSLA (Tesla)

You can modify the `symbols` list in the script to use different stocks if desired.

## Key Features

### Portfolio Optimization
- Automatic data download and preprocessing
- Risk-return analysis with visualizations
- Efficient frontier generation
- Minimum variance portfolio calculation
- Portfolio comparison (5 vs 4 assets)

### Black-Scholes Implementation
- Complete Black-Scholes formula implementation
- Option pricing for multiple scenarios
- Implied volatility calculation using bisection method
- Volatility smile analysis
- Theoretical vs realistic market behavior comparison

## Output

The script generates:
1. **Statistical tables** with asset returns, volatilities, and correlations
2. **Distribution plots** comparing empirical vs normal distributions
3. **Correlation heatmaps** showing asset relationships
4. **Efficient frontier plots** with individual assets and minimum variance portfolio
5. **Option pricing tables** for different volatilities and strikes
6. **Implied volatility plots** showing theoretical vs realistic patterns

## Customization

You can easily customize the analysis by modifying:
- Stock symbols in the `symbols` list
- Date range for data download
- Risk-free rate for option pricing
- Number of portfolios on the efficient frontier
- Volatility scenarios for option pricing

## Academic Integrity

This solution is provided as a reference for understanding the concepts. Please ensure you:
- Understand all calculations and methodologies
- Can explain the results and interpretations
- Use this as a learning tool, not for direct submission
- Follow your institution's academic integrity policies

## Contact

For questions about this implementation, please refer to the course materials or consult with your instructor.

---

**Note:** This assignment requires internet connection for downloading stock data. If you encounter issues with data download, you can use the provided sample data or modify the script to use local data files.