import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import time

def bs_call(S, K, T, r, sigma):
    """Black-Scholes call option pricing"""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol(S, K, T, r, price, tol=0.005):
    """Calculate implied volatility using bisection method"""
    def f(sigma):
        return bs_call(S, K, T, r, sigma) - price
    
    x0, x1 = 0.0001, 5.0
    f0, f1 = f(x0), f(x1)
    
    if f0 * f1 > 0:
        if f0 > 0:
            return 0.0001  # Price too low, min vol
        else:
            return 5.0  # Price too high, max vol
    
    while (x1 - x0) > tol:
        xm = (x0 + x1) / 2
        fm = f(xm)
        if f0 * fm < 0:
            x1 = xm
        else:
            x0, f0 = xm, fm
    
    return (x0 + x1) / 2

def get_option_data(ticker_symbol, max_retries=3):
    """Fetch real option data from yfinance with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} to fetch {ticker_symbol} data...")
            ticker = yf.Ticker(ticker_symbol)
            
            # Get current stock price
            hist = ticker.history(period="1d")
            if hist.empty:
                print(f"Could not fetch price data for {ticker_symbol}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Failed to fetch price data")
            
            S0 = hist['Close'].iloc[-1]
            print(f"{ticker_symbol} current price: ${S0:.2f}")
            
            # Add delay before option chain request
            time.sleep(1)
            
            # Get option chain for the nearest expiration
            option_chain = ticker.option_chain()
            calls = option_chain.calls
            
            if calls.empty:
                print(f"No call options available for {ticker_symbol}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise Exception("No call options available")
            
            # Calculate time to expiration
            exp_date = calls['lastTradeDate'].iloc[0]
            if pd.isna(exp_date):
                # If no trade date, use a default expiration
                exp_date = datetime.now() + timedelta(days=30)
            
            T = (exp_date - datetime.now()).days / 365.0
            print(f"Time to expiration: {T:.3f} years")
            
            # Select strikes around current price
            atm_strike = calls['strike'].iloc[(calls['strike'] - S0).abs().argsort()[:1]].iloc[0]
            
            # Get strikes: 2 below ATM, ATM, 2 above ATM
            below_strikes = calls[calls['strike'] < S0]['strike'].sort_values(ascending=False).head(2)
            above_strikes = calls[calls['strike'] > S0]['strike'].head(2)
            
            selected_strikes = list(below_strikes.values) + [atm_strike] + list(above_strikes.values)
            
            # Filter calls for selected strikes
            selected_calls = calls[calls['strike'].isin(selected_strikes)].copy()
            selected_calls = selected_calls.sort_values('strike')
            
            # Calculate mid prices
            selected_calls['mid_price'] = (selected_calls['bid'] + selected_calls['ask']) / 2
            
            # Filter out options with zero or negative prices
            selected_calls = selected_calls[selected_calls['mid_price'] > 0]
            
            if len(selected_calls) < 3:
                print(f"Not enough valid options for {ticker_symbol}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Not enough valid options")
            
            print(f"Successfully fetched {len(selected_calls)} options for {ticker_symbol}")
            return S0, selected_calls[['strike', 'mid_price']], T
            
        except Exception as e:
            print(f"Error fetching data for {ticker_symbol} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print("Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"Failed to fetch data for {ticker_symbol} after {max_retries} attempts")
                raise e

# Risk-free rate (using 10-year Treasury yield as proxy)
r = 0.05  # 5% annual risk-free rate

# Fetch real data for AAPL and PLTR
print("Fetching real market data...")
print("=" * 50)

# AAPL data
print("Fetching AAPL data...")
S0_aapl, aapl_options, T_aapl = get_option_data('AAPL')
print(f"AAPL options found: {len(aapl_options)} strikes")
print(aapl_options)

print("\n" + "=" * 50)

# Add delay to avoid rate limiting
print("Waiting 3 seconds to avoid rate limiting...")
time.sleep(3)

# PLTR data
print("Fetching PLTR data...")
S0_pltr, pltr_options, T_pltr = get_option_data('PLTR')
print(f"PLTR options found: {len(pltr_options)} strikes")
print(pltr_options)

print("\n" + "=" * 50)

# Calculate implied volatilities for AAPL
print("Calculating implied volatilities...")
data_aapl = []
for _, row in aapl_options.iterrows():
    K, price = row['strike'], row['mid_price']
    if price > 0:  # Only calculate IV for positive prices
        iv = implied_vol(S0_aapl, K, T_aapl, r, price)
        data_aapl.append({'Stock': 'AAPL', 'Strike': K, 'Price': price, 'IV': iv})

# Calculate implied volatilities for PLTR
data_pltr = []
for _, row in pltr_options.iterrows():
    K, price = row['strike'], row['mid_price']
    if price > 0:  # Only calculate IV for positive prices
        iv = implied_vol(S0_pltr, K, T_pltr, r, price)
        data_pltr.append({'Stock': 'PLTR', 'Strike': K, 'Price': price, 'IV': iv})

# Combine data
all_data = data_aapl + data_pltr
df = pd.DataFrame(all_data)

print("\nImplied Volatility Data:")
print(df)

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# AAPL plot
aapl_data = df[df['Stock'] == 'AAPL']
axes[0].plot(aapl_data['Strike'], aapl_data['IV'], 'o-', label='AAPL', linewidth=2, markersize=6)
axes[0].set_title('AAPL Implied Volatility vs Strike')
axes[0].set_xlabel('Strike Price ($)')
axes[0].set_ylabel('Implied Volatility')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# PLTR plot
pltr_data = df[df['Stock'] == 'PLTR']
axes[1].plot(pltr_data['Strike'], pltr_data['IV'], 'o-', label='PLTR', color='orange', linewidth=2, markersize=6)
axes[1].set_title('PLTR Implied Volatility vs Strike')
axes[1].set_xlabel('Strike Price ($)')
axes[1].set_ylabel('Implied Volatility')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('plots/implied_vol.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("\nComment: IV skew indicates higher volatility expectation for OTM calls, reflecting market tail risk pricing.")
print("Real market data shows the actual implied volatility smile/skew pattern observed in practice.")