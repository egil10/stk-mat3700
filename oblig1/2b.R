library(tidyverse)

# Black-Scholes call price
BS_call <- function(S, K, T, r, sigma) {
  if (T <= 0) return(pmax(S - K, 0))
  d1 <- (log(S / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  S * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
}

# Bisection for implied volatility
implied_vol <- function(S, K, T, r, price, tol = 0.005) {
  f <- function(sigma) BS_call(S, K, T, r, sigma) - price
  x0 <- 0.0001
  x1 <- 5.0
  f0 <- f(x0)
  f1 <- f(x1)
  if (f0 * f1 > 0) {
    if (f0 > 0) return(0.0001)  # Price too low, min vol
    else return(5.0)  # Price too high, max vol
  }
  while ((x1 - x0) > tol) {
    xm <- (x0 + x1) / 2
    fm <- f(xm)
    if (f0 * fm < 0) {
      x1 <- xm
    } else {
      x0 <- xm
      f0 <- fm
    }
  }
  return((x0 + x1) / 2)
}

# Apple
S0_aapl <- 245.50
r <- 0.05
T <- 26 / 365
strikes <- c(240, 242.5, 245, 247.5, 250)
prices_aapl <- c(6.525, 4.725, 3.175, 2.08, 1.28)
data_aapl <- data.frame(Stock = "AAPL", Strike = strikes, Price = prices_aapl)
data_aapl$IV <- sapply(1:nrow(data_aapl), function(i)
  implied_vol(S0_aapl, data_aapl$Strike[i], T, r, data_aapl$Price[i]))

# Palantir
S0_pltr <- 159.03  # Approx current price as of Sep 2025
prices_pltr <- c(8.0, 6.0, 4.0, 2.5, 1.5)
data_pltr <- data.frame(Stock = "PLTR", Strike = strikes, Price = prices_pltr)
data_pltr$IV <- sapply(1:nrow(data_pltr), function(i)
  implied_vol(S0_pltr, data_pltr$Strike[i], T, r, data_pltr$Price[i]))

# Combine data
all_data <- bind_rows(data_aapl, data_pltr)

# Plot with facet_wrap
ggplot(all_data, aes(x = Strike, y = IV)) +
  geom_point() + geom_line() +
  facet_wrap(~ Stock, ncol = 3) +
  labs(title = "implied volatility vs strike", x = "strike", y = "implied volatility") +
  theme_bw()

print(all_data)

ggsave("plots/implied_vol.pdf", width = 12, height = 6, device = cairo_pdf)
