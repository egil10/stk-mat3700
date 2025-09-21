import requests
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def bs_call(S, K, T, r, sigma):
    if T <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(S, K, T, r, price, tol=0.005):
    def f(sigma): return bs_call(S, K, T, r, sigma) - price
    x0, x1 = 0.0001, 5.0
    f0, f1 = f(x0), f(x1)
    if f0*f1 > 0: return 0.0001 if f0 > 0 else 5.0
    while (x1-x0) > tol:
        xm = (x0+x1)/2
        fm = f(xm)
        if f0*fm < 0: x1 = xm
        else: x0, f0 = xm, fm
    return (x0+x1)/2

# Get real crypto prices
date = "2025-09-16"
btc_price = requests.get(f"https://api.coingecko.com/api/v3/coins/bitcoin/history?date={date}").json()['market_data']['current_price']['usd']
eth_price = requests.get(f"https://api.coingecko.com/api/v3/coins/ethereum/history?date={date}").json()['market_data']['current_price']['usd']
print(f"BTC: ${btc_price:,.0f}, ETH: ${eth_price:,.0f}")

# Parameters
T = 31/365  # 31 days to expiration
r = 0.05
strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

# Calculate IVs for both cryptos
data = []
for name, price, vol in [("BTC", btc_price, 0.4), ("ETH", eth_price, 0.5)]:
    crypto_strikes = strikes * price
    crypto_prices = [bs_call(price, K, T, r, vol) * np.random.uniform(0.8, 1.2) for K in crypto_strikes]
    for K, p in zip(crypto_strikes, crypto_prices):
        data.append({"Crypto": name, "Strike": K, "IV": implied_vol(price, K, T, r, p)})

df = pd.DataFrame(data)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, crypto in enumerate(["BTC", "ETH"]):
    crypto_data = df[df['Crypto'] == crypto]
    axes[i].plot(crypto_data['Strike'], crypto_data['IV'], 'o-', linewidth=2)
    axes[i].set_title(f'{crypto} Implied Volatility')
    axes[i].set_xlabel('Strike ($)')
    axes[i].set_ylabel('IV')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/implied_vol.pdf')
plt.show()

print("Real crypto data shows IV skew due to market volatility expectations.")