# Black-Scholes Call Option Pricing in Python

This project implements the Black-Scholes model in Python to price European call options (contracts exercisable only at expiration). It's commonly used as an approximation for American calls when early exercise is unlikely.

## Black-Scholes PDE

The Black-Scholes partial differential equation is:

$$
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$

Where:
- $V$: Option price  
- $S$: Underlying asset price  
- $\sigma$: Volatility  
- $r$: Risk-free interest rate  
- $t$: Time to expiration  

## Model Assumptions

- Constant $r$ and $\sigma$  
- Asset follows geometric Brownian motion  
- No dividends, taxes, or transaction costs  
- Unlimited shorting and borrowing at the risk-free rate  
- Frictionless markets (no arbitrage)

## Analytical Solution (European Call)

$$
C(S, t) = S \cdot N(d_1) - K e^{-r \tau} \cdot N(d_2)
$$

With:

$$
d_1 = \frac{\ln(S / K) + (r + \frac{1}{2} \sigma^2) \tau}{\sigma \sqrt{\tau}}, \quad
d_2 = d_1 - \sigma \sqrt{\tau}
$$

Where:
- $\tau = T - t$: Time to maturity  
- $T$: Expiration time  
- $K$: Strike price  
- $N(x)$: CDF of the standard normal distribution
