
library(tidyverse)

# Black-Scholes call option price function
BS_call <- function(S0, K, T, r, sigma) {
  d1 <- (log(S0 / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  S0 * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
}

# parameters
S0 <- 100  
r <- 0.05  
sigmas <- c(0.1, 0.3, 0.5)
Ts <- c(1/12, 3/12, 6/12)  
strike_mult <- c(0.6, 0.8, 1.0, 1.2, 1.4)
Ks <- S0 * strike_mult

# compute prices for all combinations
combs <- expand.grid(sigma = sigmas, T = Ts, K = Ks)
combs$price <- mapply(BS_call, S0, combs$K, combs$T, r, combs$sigma)

# plot price vs strike for each T, faceted by sigma
ggplot(combs, aes(x = K, y = price, color = factor(T))) +
  geom_line() +
  facet_wrap(~ sigma) +
  labs(title = "Call Option Prices", x = "strike K", y = "price", color = "T (years)") +
  theme_bw()

# print prices
print(combs) %>% head(5)

ggsave("plots/blackscholes.pdf", width = 12, height = 6, device = cairo_pdf)
