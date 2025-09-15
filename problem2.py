"""
STK-MAT3700 Assignment 1 - Problem 2: Black-Scholes Options Pricing
Parts a, b, c - Complete solution with plots saved as PDF
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import os

# Create plots directory
os.makedirs('plots', exist_ok=True)

class BlackScholes:
    """Black-Scholes option pricing implementation"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put
    
    @staticmethod
    def implied_volatility_call(S, K, T, r, market_price, initial_guess=0.2):
        def objective(sigma):
            return BlackScholes.call_price(S, K, T, r, sigma) - market_price
        
        try:
            implied_vol = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return implied_vol
        except ValueError:
            return BlackScholes._bisection_implied_vol(S, K, T, r, market_price)
    
    @staticmethod
    def _bisection_implied_vol(S, K, T, r, market_price, tolerance=0.005):
        x0, x1 = 0.001, 2.0
        f0 = BlackScholes.call_price(S, K, T, r, x0) - market_price
        f1 = BlackScholes.call_price(S, K, T, r, x1) - market_price
        
        if f0 * f1 > 0:
            x1 = 5.0
            f1 = BlackScholes.call_price(S, K, T, r, x1) - market_price
        
        while abs(x1 - x0) > tolerance:
            xm = (x0 + x1) / 2
            fm = BlackScholes.call_price(S, K, T, r, xm) - market_price
            
            if fm * f0 < 0:
                x1 = xm
                f1 = fm
            else:
                x0 = xm
                f0 = fm
        
        return (x0 + x1) / 2

print("=== PROBLEM 2: BLACK-SCHOLES OPTIONS PRICING ===")

# Parameters for option pricing
S0 = 100  # Current stock price
r = 0.05  # Risk-free rate (5% annually)
T_values = [1/12, 3/12, 6/12]  # 1 month, 3 months, 6 months
sigma_values = [0.10, 0.30, 0.50]  # 10%, 30%, 50% volatility
K_values = [S0 * (1 + k) for k in [-0.4, -0.2, 0, 0.2, 0.4]]  # ±40%, ±20%, ATM

print(f"\nParameters:")
print(f"Stock Price (S0): ${S0}")
print(f"Risk-free Rate (r): {r:.1%}")
print(f"Strike Prices: {[f'${k:.0f}' for k in K_values]}")
print(f"Time to Expiry: {[f'{T*12:.0f} months' for T in T_values]}")
print(f"Volatilities: {[f'{σ:.0%}' for σ in sigma_values]}")

print(f"\nRisk-free rate choice: {r:.1%} - This represents a reasonable current market rate")
print("for government bonds, which is typically used as the risk-free rate in option pricing.")

# Problem 2a: Calculate option prices
print("\n=== PROBLEM 2a: Black-Scholes Option Pricing ===")

# Create option pricing table
option_prices = {}

for sigma in sigma_values:
    option_prices[sigma] = {}
    for T in T_values:
        option_prices[sigma][T] = {}
        for K in K_values:
            call_price = BlackScholes.call_price(S0, K, T, r, sigma)
            option_prices[sigma][T][K] = call_price

print("\nDetailed Option Prices:")
for sigma in sigma_values:
    print(f"\nVolatility σ = {sigma:.0%}:")
    for T in T_values:
        print(f"  T = {T*12:.0f} months:")
        for K in K_values:
            price = option_prices[sigma][T][K]
            moneyness = "ITM" if K < S0 else "ATM" if K == S0 else "OTM"
            print(f"    K = ${K:.0f} ({moneyness}): ${price:.2f}")

# Plot option prices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, sigma in enumerate(sigma_values):
    ax = axes[i]
    for T in T_values:
        strikes = []
        prices = []
        for K in K_values:
            strikes.append(K)
            prices.append(option_prices[sigma][T][K])
        
        ax.plot(strikes, prices, 'o-', label=f'{T*12:.0f} months', linewidth=2, markersize=6)
    
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Call Option Price')
    ax.set_title(f'Option Prices (σ = {sigma:.0%})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=S0, color='red', linestyle='--', alpha=0.7, label='Current Price')

plt.tight_layout()
plt.savefig('plots/problem2a_option_prices.pdf')
plt.show()

print("\n=== PROBLEM 2b: Implied Volatility Analysis ===")

# Simulate market prices using Black-Scholes with a 'true' volatility
true_volatility = 0.25
market_prices = {}
implied_vols = {}

print(f"True volatility used: {true_volatility:.1%}")
print("Note: For this example, we'll simulate market prices using Black-Scholes")
print("with a 'true' volatility of 25% to demonstrate the concept.")

# Simulate market prices for 3-month options
T_market = T_values[1]  # 3-month options

for K in K_values:
    market_price = BlackScholes.call_price(S0, K, T_market, r, true_volatility)
    market_prices[K] = market_price
    implied_vol = BlackScholes.implied_volatility_call(S0, K, T_market, r, market_price)
    implied_vols[K] = implied_vol

print("\nMarket Prices and Implied Volatilities (3-month options):")
for K in K_values:
    moneyness = "ITM" if K < S0 else "ATM" if K == S0 else "OTM"
    print(f"K = ${K:.0f} ({moneyness}): Market Price = ${market_prices[K]:.2f}, Implied Vol = {implied_vols[K]:.1%}")

# Plot implied volatility smile
plt.figure(figsize=(10, 6))
strikes = list(K_values)
implied_vol_list = [implied_vols[K] for K in strikes]

plt.plot(strikes, implied_vol_list, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=true_volatility, color='red', linestyle='--', alpha=0.7, label=f'True Volatility ({true_volatility:.1%})')
plt.axvline(x=S0, color='green', linestyle='--', alpha=0.7, label='ATM Strike')

plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility Smile (Simulated)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/problem2b_implied_volatility.pdf')
plt.show()

print("\n=== PROBLEM 2c: Analysis ===")

print("\nIf the market used Black-Scholes pricing, implied volatilities would be:")
print("✓ Constant across all strikes (flat line)")
print("✓ Equal to the true underlying volatility")
print("✓ No volatility smile or skew would be observed")

print("\nIn reality, markets show volatility smiles/skews due to:")
print("• Fat tails in return distributions (leptokurtosis)")
print("• Market crash fears (volatility increases for OTM puts)")
print("• Supply and demand imbalances")
print("• Model limitations of Black-Scholes assumptions")
print("• Jump risk and other market microstructure effects")

# Create a comparison plot showing theoretical vs realistic volatility smile
plt.figure(figsize=(12, 8))

# Theoretical flat line (Black-Scholes)
strikes_theoretical = np.linspace(60, 140, 20)
implied_vol_flat = np.full_like(strikes_theoretical, true_volatility)
plt.plot(strikes_theoretical, implied_vol_flat, 'b-', linewidth=2, label='Black-Scholes (Theoretical)')

# Realistic volatility smile (simulated)
strikes_realistic = np.linspace(60, 140, 20)
# Simulate a realistic smile: higher vol for OTM options
implied_vol_smile = true_volatility + 0.05 * np.exp(-0.5 * ((strikes_realistic - S0) / 20) ** 2)
plt.plot(strikes_realistic, implied_vol_smile, 'r--', linewidth=2, label='Real Market (Typical Smile)')

plt.axvline(x=S0, color='green', linestyle=':', alpha=0.7, label='ATM Strike')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Theoretical vs Realistic Implied Volatility Patterns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/problem2c_comparison.pdf')
plt.show()

print("\n=== SUMMARY ===")
print("All plots saved to plots/ directory:")
print("- problem2a_option_prices.pdf")
print("- problem2b_implied_volatility.pdf")
print("- problem2c_comparison.pdf")
print("\nProblem 2 completed successfully!")